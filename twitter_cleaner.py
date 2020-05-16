#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 14:13:44 2020

@author: piotr
"""


import csv
import itertools
import operator
from tqdm import trange
import gc
import os
from twitter_tools import preprocess_tweets
import morfeusz2
import collections
import re
import json
import sys
import argparse
from datetime import datetime
import pdb

class TweetCleaner():


    def __init__(self, path_config):
        """
        

        Parameters
        ----------
        path_config : string
            Path to the config file containing the record of previous runs.

        Returns
        -------
        None.

        """

        if not os.path.isfile(path_config):
            self.config = collections.defaultdict(lambda: 0)
        else:
            self.config = collections.defaultdict(lambda: 0, json.load(open(path_config, "r")))

        self.path_config = path_config



    def clean(self, path_source, path_target, path_stopwords = None, cols = None, chunk_size = 100000):
        """
        

        Parameters
        ----------
        path_source : string
            path of the csv file to be preprocessed
        path_target : string
            destination path of the preprocessed data.
        cols : list, optional
            Column names to used. if None, default columns needed for the scraper to run.
        path_stopwords : string, optional
            Path to the stopwords file. The default is None - no stopword removal performed
        chunk_size : integer, optional
            How many rows should be preprocessed at one run. The default is 100000.

        Returns
        -------
        None.

        """
        
        if not cols:
            cols = ["","created_at","id_str","user-id_str", "user-followers_count", 
                    "full_text", "retweeted_status-full_text","retweeted_status-id_str"]

        old_rows = self.config[path_source]

        with open(path_source,"r") as file:
            reader = csv.DictReader(file)
            if old_rows != 0:
                print("Going through already cleaned rows")
                for _ in trange(old_rows):
                    next(reader)
            n_rows = old_rows
            
            #new added rows:
            for _ in itertools.count():

                #read chunk
                res = list(itertools.islice(reader, chunk_size))
                
                if not res:
                    break
                
                #count
                length_in = len(res)
                print(f"Read {length_in} tweets")
                n_rows += length_in

                           
                #container for variables         
                dict_vars = dict() 

                #merge dictionaries into one
                for d in res:
                    for k, v in d.items():
                        if k in cols: #if in columns selected
                            dict_vars.setdefault(k, []).append(v)
                dict_vars["index"] = dict_vars.pop("")
                
                del(res)
                gc.collect()
                
                
                #check for data issues:
                dict_vars, dict_vars_probl = self.check_errors(dict_vars)
                
                    
                #preprocess ids:
                dict_vars = self.preprocess_text(dict_vars, path_stopwords = path_stopwords)
                
                #pdb.set_trace()
                



                #write preprocessed
                fmt_from = "%a %b %d %H:%M:%S +0000 %Y"
                fmt_to = "%Y_%m_%d"
                dict_vars["day"] = list(map(lambda x: datetime.strptime(x, fmt_from).strftime(fmt_to), dict_vars["created_at"]))
                dict_vars = [dict(zip(dict_vars, i)) for i in zip(*dict_vars.values())]
                dict_vars = sorted(dict_vars, key = lambda x: x["day"])
                
                
                #write tweets from each date to a separate file
                for g, elem in itertools.groupby(dict_vars, lambda x: x["day"]):
                    path = path_target.split(".")[0] + "_" + g + ".csv"
                    self.write_file(list(elem), path)
                    
                del(dict_vars, dict_vars_probl)
                gc.collect()
                    
        #DONE
        print(f"Went through {n_rows} rows")      
        self.config[path_source] = n_rows
        json.dump(self.config, open(self.path_config, "w"))
        
        
        
    def check_errors(self, dict_vars):
        """
        

        Parameters
        ----------
        dict_vars : dictionary
         Key is column name and value is list of values.

        Returns
        -------
        dict_vars_good : dictionary
            Dictionary with dropped corrupt records.
        dict_vars_probl : dictionary
            Dictionary of corrupt records.

        """
        
        
        length_in = len(dict_vars["index"])
        
        #set all to non-problematic by default
        probl = [False for _ in range(length_in)]
        
        
        #date is not text:
        redate = "[A-Z][a-z]{2} [A-Z][a-z]{2} \d{2} \d{2}:\d{2}:\d{2} \+\d{4} \d{4}"
        test = list(map(lambda x: not re.match(redate, str(x)), dict_vars["created_at"]))
        #print(f"Number of date errors : {len([i for i in test if i == True])}")
        probl = [probl[i] or test[i] for i in range(len(probl))]
        
        
        #columns that should be strings of numbers aren't:
        should_be_numeric = ["index","id_str","user-id_str","retweeted_status-id_str"]
        for col in should_be_numeric:
            test = list(map(lambda x: not(str(x).isnumeric() or not str(x)), dict_vars[col]))
            #print(f"Number of {col} errors : {len([i for i in test if i == True])}")
            probl = [probl[i] or test[i] for i in range(len(probl))]
        
        
        #convert boolean to indices:
        nprobl_ind = [i for i, boolean in enumerate(probl) if boolean == False]
        probl_ind = [i for i, boolean in enumerate(probl) if boolean == True]
        print(f"Number of problematic ids {len(probl_ind)}")
        
        #separate problematic and non-problematic:
        getter = operator.itemgetter(*nprobl_ind)
        dict_vars_good = {key:getter(val) for key, val in dict_vars.items()}
        
        if probl_ind: #if there were any errors, return them
            getter = operator.itemgetter(*probl_ind)
            dict_vars_probl = {key:getter(val) for key, val in dict_vars.items()}
        else:
            dict_vars_probl = dict()
            
        
        return dict_vars_good, dict_vars_probl
    
    
    
    def morfeusz_lemmatize(self, tweet, morf):
        """
        Perform Polish language lemmatization using  Morfeusz lexical analyser

        Parameters
        ----------
        tweet : string.
        morf : morfeusz2 object.

        Returns
        -------
        res : Lemmatized text.

        """
        res = []
        counter = 0
        for elem in morf.analyse(tweet): #iteratre over morfeusz output
            #get only one lemma available per word - currently obtains the first one
            if elem[0] == counter:
                res.append(re.search(r"\w+",elem[2][1]).group(0))
                counter += 1
        res = " ".join(res)
        return res


    def preprocess_text(self, dict_vars, path_stopwords = None):
        """
        
        Run preprocessing on text

        Parameters
        ----------
        dict_vars : dictionary
            Key is column name and value is list of values.
        path_stopwords : string, optional
            path to the stopwords file.

        Returns
        -------
        dict_vars : dictionary
            Preprocessed data in the same format.

        """

        #stopwords https://github.com/bieli/stopwords.git

        length_in = len(dict_vars["index"])

        #mark retweets:
        dict_vars["retweet"] = ["Yes" if len(id_str) > 0 else "No" for id_str in dict_vars["retweeted_status-id_str"]]

        #replace retweets with full text:
        dict_vars["full_text"] = ["RT " + dict_vars["retweeted_status-full_text"][i] if dict_vars["retweet"][i] == "Yes" else dict_vars["full_text"][i] for i in range(length_in)]

        #don't need that anymore:
        dict_vars.pop("retweeted_status-full_text")

        #apply preprocessing:
        dict_vars["preprocessed"] = preprocess_tweets(dict_vars["full_text"])

        morf = morfeusz2.Morfeusz()
        for i, tweet in enumerate(dict_vars["preprocessed"]):
            dict_vars["preprocessed"][i] = self.morfeusz_lemmatize(tweet, morf)


        if path_stopwords:
            with open(path_stopwords,"r") as f:
                stopwords = f.read().splitlines() 
                
            dict_vars["preprocessed"] = [tweet.split(" ") for tweet in dict_vars["preprocessed"]]
            
            for i, tweet in enumerate(dict_vars["preprocessed"]):
                dict_vars["preprocessed"][i] = [word for word in tweet if word not in stopwords]
    
            dict_vars["preprocessed"] = [" ".join(tweet) for tweet in dict_vars["preprocessed"]]

        print(f"Preprocessed {length_in} tweets")

        return dict_vars
    
    
    
    
    
    def write_file(self, data, path):
        length_out = len(data)
        write_mode = "a" if os.path.isfile(path) else "w"
        with open(path, write_mode) as f:
            writer = csv.DictWriter(f, fieldnames = list(data[0].keys()))
            if write_mode == "w":
                writer.writeheader()
                print(f"Creating file {path}")
            writer.writerows(data)
        print(f"Wrote {length_out} tweets to file {path}")
    
    
if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("path_config")
        parser.add_argument("path_source")
        parser.add_argument("path_target")
        parser.add_argument("--stopwords")
        args = parser.parse_args()
        cleaner = TweetCleaner(args.path_config)
        cleaner.clean(args.path_source, args.path_target, path_stopwords = args.stopwords)
    except Exception as e:
        print(e)
    
    