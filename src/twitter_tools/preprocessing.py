#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 10:19:45 2020

@author: piotr
"""
import stanza
import os
import pandas as pd
import datetime
import enchant
from tqdm import tqdm
from utils import clean_tweets, tag_retweets, check_errors
import json
import collections
import argparse
from nltk.metrics.distance import edit_distance
import logging
import pdb
import sys


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



class Lemmatizer:
    def __init__(self, stopwords = None, use_gpu = True, **kwargs):
        
        
        self.nlp_pl = stanza.Pipeline("pl", processors = "tokenize,mwt,pos,lemma", 
                                      use_gpu = use_gpu, tokenize_no_ssplit = True, 
                                      **kwargs)
        self.dict_pl = enchant.Dict("pl_PL")
        self.logger = logging.getLogger(__name__)

        
        if stopwords:
            self.stopwords = []
            with open(stopwords, "r") as f:
                for line in f:
                    self.stopwords.append(line.strip("\n"))
        else:
            self.stopwords = []
        
        
    def lemmatize(self, texts:list):
        """
        Lemmatize string.
        """
        
        doc = self.bind(texts)
        doc = self.nlp_pl(doc) 
        
        
        
        
        result = []
        for sent in doc.sentences:
            result.append([word.lemma for word in sent.words if word.lemma not in self.stopwords])
        self.logger.info(f"Lemmatized {len(result)} sentences")
        return result

           
    def bind(self, texts:list):
        # Bind texts for stanza pipeline
        return "\n\n".join(texts)
    
    
    
class SpellChecker:
    
    def __init__(self, threshold = 0.2):
        self.dict_pl = enchant.Dict("pl_PL")
        self.dict_en = enchant.Dict("en_EN")
        self.threshold = threshold
        

    def spellcheck(self, text:list):
        """
        Return checked list of words
        """
        return [self.check(word).lower() for word in text]
    
    
    def check(self, word:str):
        
        pl = self.dict_pl.suggest(word)

        if pl and pl[0].lower() == word.lower():
            return word
        else:
            en = self.dict_pl.suggest(word)
            if en and en[0].lower() == word.lower():
                return word
            
            elif pl and edit_distance(pl[0], word)/len(word) < self.threshold:
                return pl[0]
            
            elif en and edit_distance(en[0], word)/len(word) < self.threshold:
                return en[0]
            else:
                return word
    
    
    
    
class Preprocessor:
    def __init__(self, keep_cols = None, chunksize = 100000, record_path = None, **kwargs):
        
        self.cols = ["full_text","created_at","id_str"]

        if keep_cols:
            self.cols = list(set(self.cols + keep_cols))
        
        self.chunksize = chunksize #chunk size for loader
        
        self.lem = Lemmatizer(**kwargs)
        
        self.logger = logging.getLogger(__name__)
        
        
        self.record_path = record_path
        
        if record_path and os.path.isfile(record_path):
            self.record = json.load(open(self.record_path, "r"))
            self.record = collections.defaultdict(lambda: 0, self.record)
        else:
            self.record = collections.defaultdict(lambda: 0)
            
        self.logger.info(f"Preprocessor initialized with chunk size {chunksize}")
        
        
    def preprocess(self, source_file, target_dir, split_dates = True):
        
        #load file header
        names = pd.read_csv(source_file, dtype = str, nrows = 1, index_col = 0)
        names = names.columns
        
        #read the data by chunk
        for df in tqdm(pd.read_csv(source_file, iterator = True, dtype = str, 
                                   skiprows = self.record[source_file] + 1,
                                   chunksize = self.chunksize, index_col = 0, 
                                   names = names)):
            
            
            df = check_errors(df) #error check (column format etc)
            
            if df.empty:
                continue
            
            df = df[self.cols] #keep important columns
            

            
            df = tag_retweets(df) #tag retweets
            


            #clean mentions, interp, etc.:
            df["preprocessed"] = df["full_text"].apply(lambda x: clean_tweets(x)) 
            
            
            
            #drop empty rows before lemmatizing:
            indx = (df["preprocessed"] != "")  & (df["preprocessed"].notna())
            df = df[indx]
            
            

            
            #lemmatize - use a dictionary of unique tweets to save time
            tweet_dict = dict().fromkeys(df["preprocessed"].tolist()) 
            lemmas = self.lem.lemmatize(list(tweet_dict)) #lemmatization
            tweet_dict = {k:lemmas[i] for i, k in enumerate(tweet_dict)}
            df["preprocessed"] = df["preprocessed"].map(tweet_dict) #replacement
            df["preprocessed"] = df["preprocessed"].apply(lambda x: [word.lower() for word in x])
            
            
            df["day"] = pd.to_datetime(df["created_at"], 
                                       format = "%a %b %d %H:%M:%S +0000 %Y").dt.date #get day
            
            if split_dates:
            
                for day, subdf in df.groupby(df["day"]): #split by day
                    
                    
                    #get target directory:
                    target = os.path.split(source_file)[-1].split(".")[0]
                    target += "_" + datetime.datetime.strftime(day, "%d_%m_%Y") + ".csv"
                    target = os.path.join(target_dir, target)
                    
                    if os.path.isfile(target):
                        subdf.to_csv(target, mode = "a", header = False) #append if exists
                        
                    else:
                        subdf.to_csv(target, mode = "w") #else write to new
                        
                    self.logger.info(f"Wrote {subdf.shape[0]} rows to file {target}")
            else:
                
                target = os.path.split(source_file)[-1].split(".")[0]
                target += ".csv"
                target = os.path.join(target_dir, target)
                if os.path.isfile(target):
                    df.to_csv(target, mode = "a", header = False) #append if exists
                        
                else:
                    df.to_csv(target, mode = "w") #else write to new
                    
                self.logger.info(f"Wrote {subdf.shape[0]} rows to file {target}")
            self.record[source_file] += self.chunksize
            self.logger.info(f"Cumulative row count {self.record[source_file]}")
            
            if self.record_path:
                json.dump(self.record, open(self.record_path, "w"))
            
            
if __name__ == "__main__":
    error_counter = 0
    parser = argparse.ArgumentParser()
    parser.add_argument("source_file", help = "Path to source csv")
    parser.add_argument("target_dir", help = "Directory to store output in")
    parser.add_argument("--keep_cols", help = "List of columns to keep in the output")
    parser.add_argument("--stopwords", help = "Path to stopwords")
    parser.add_argument("--record_path", help = "Path to json record")
    parser.add_argument("--chunksize", help = "Chunk size")
    parser.add_argument("--pos_batch_size", help = "Batch size for stanza pos tagger")
    args = parser.parse_args()
    args.keep_cols = json.loads(args.keep_cols)
    args.chunksize = int(args.chunksize)
    args.pos_batch_size = int(args.pos_batch_size)
    while True:
        try:
            cleaner = Preprocessor(keep_cols = args.keep_cols, 
                                   record_path = args.record_path, 
                                   stopwords = args.stopwords, 
                                   chunksize = args.chunksize, 
                                   pos_batch_size = args.pos_batch_size)
            cleaner.preprocess(source_file = args.source_file, 
                               target_dir = args.target_dir)
            break
        except Exception as e:
            print(e)
            error_counter += 1
            args.chunksize -= error_counter * 100
            if error_counter > 5 or args.chunksize <= 100:
                break
        
    

