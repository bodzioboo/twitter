#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 14:13:44 2020

@author: piotr
"""


import csv
import itertools
import operator
import gc
import os
from twitter_tools import preprocess_tweets
import morfeusz2
import collections
import re
import json

class TweetCleaner():


    def __init__(self, path_config):

        if not os.path.isfile(path_config):
            self.config = collections.defaultdict(lambda: 0)
        else:
            self.config = collections.defaultdict(lambda: 0, json.load(open(path_config, "r")))

        self.path_config = path_config



    def clean(self, path_source, path_target, cols, path_stopwords, chunk_size = 100000):

        old_rows = self.config[path_source]

        with open(path_source,"r") as file:
            reader = csv.DictReader(file)
            n_rows = 0
            for _ in itertools.count():
                
                while n_rows < old_rows - 1: #while rows preprocessed previously
                    next(reader)
                    n_rows += 1
                    if n_rows % chunk_size == 0:
                        print(f"Empty run {n_rows}")
                break
            
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
                
                #write problematic ids to a separate file for manual examination:
                if False:
                    path_problems = "_problems.".join(path_source.split("."))
                    length_out = len(dict_vars_probl["index"])
                    write_mode = "a" if os.path.isfile(path_problems) else "w"
                    with open(path_problems, write_mode) as f:
                        writer = csv.writer(f)
                        if write_mode == "w":
                            writer.writerow(list(dict_vars.keys()))
                        writer.writerows(list(map(lambda x: operator.itemgetter(i)(x), dict_vars_probl.values())) for i in range(length_out))
                    

                #preprocess ids:
                dict_vars = self.preprocess_text(dict_vars, path_stopwords = path_stopwords)



                #count preprocessed
                length_out = len(dict_vars["index"])
                write_mode = "a" if os.path.isfile(path_target) else "w"
                with open(path_target, write_mode) as f:
                    writer = csv.writer(f)
                    if write_mode == "w":
                        writer.writerow(list(dict_vars.keys()))
                    writer.writerows(list(map(lambda x: operator.itemgetter(i)(x), dict_vars.values())) for i in range(length_out))
                print(f"Wrote {length_out} tweets to file {path_target}")
                del(dict_vars, dict_vars_probl)
                gc.collect()
                    
        #DONE
        print(f"Went through {n_rows} rows")      
        self.config[path_source] = n_rows
        json.dump(self.config, open(self.path_config, "w"))
        
        
        
    def check_errors(self, dict_vars):
        
        
        length_in = len(dict_vars["index"])
        
        #set all to non-problematic by default
        probl = [False for _ in range(length_in)]
        
        
        #date is not text:
        test = list(map(lambda x: str(x).isnumeric(), dict_vars["created_at"]))
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
        res = []
        counter = 0
        for elem in morf.analyse(tweet): #iteratre over morfeusz output
            #get only one lemma available per word - currently obtains the first one
            if elem[0] == counter:
                res.append(re.search(r"\w+",elem[2][1]).group(0))
                counter += 1
        res = " ".join(res)
        return res


    def preprocess_text(self, dict_vars, path_stopwords):

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



        with open(path_stopwords,"r") as f:
            stopwords = f.read().splitlines() 
        dict_vars["preprocessed"] = [tweet.split(" ") for tweet in dict_vars["preprocessed"]]
        for i, tweet in enumerate(dict_vars["preprocessed"]):
            dict_vars["preprocessed"][i] = [word for word in tweet if word not in stopwords]

        dict_vars["preprocessed"] = [" ".join(tweet) for tweet in dict_vars["preprocessed"]]

        print(f"Preprocessed {length_in} tweets")

        return dict_vars