#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 14:43:53 2020

@author: piotr
"""

import os
from tweepy.cursor import Cursor
import pandas as pd
import json
from tqdm import tqdm 
from collections import defaultdict



class Scraper:
    """
    This is a generic class containing some default methods for different scrapers
    """
    
    def __init__(self, api, lang, path, count, limit = 0):
        """
        

        Parameters
        ----------
        api : tweepy API object

        lang : string
            Tweepy language abbreviation. Only tweets in this language will be collected.
            
        path: string
            Path to the file to collect to (WITHOUT EXTENSION, as these are added by methods)
            
        count : int 
            Count argument passed to the tweepy.api method used by scraper
            
        limit : int
            Limit for the Cursor for each call of the tweepy.api method used by scraper

        Returns
        -------
        None.

        """
        
        self.limit = limit 
        self.lang = lang
        self.api = api
        self.count = count
        self.errors = dict()
        self.path_json = path + ".json"
        self.path_csv = path + ".csv"
        if os.path.exists(self.path_json): #create output file if doesn't exist
            self.tweets_dict = json.load(open(self.path_json, "r"))
            self.tweets_dict = defaultdict(lambda: dict(), self.tweets_dict)
        else:
            self.tweets_dict = defaultdict(lambda: dict()) #defaultdict of dictionaries
        
        #id of last tweet scraped
        if len(self.tweets_dict) > 0:
            self.lastid = max([tweet_id for user_dict in self.tweets_dict.values() for tweet_id in user_dict.keys()])
        else:
            self.lastid = None
        
    def scrape(self, api_method, path, **kwargs):
        """
        Generic method for scraping. Takes an api method and uses Tweepy Cursor to collect the data.

        Parameters
        ----------
        api_method : tweepy.api method
            Which method to use.
        path : string
            path to dump the json file with results.
        **kwargs : 
            keyword args to be passed to the tweepy.api method

        Returns
        -------
        None.

        """
        count = 0
        for tweet in tqdm(Cursor(api_method, since_id = self.lastid, 
                                 tweet_mode = "extended", **kwargs).items(self.limit)): #iterate over limit
            #check if language matches and tweet has not been collected
            if tweet.lang == self.lang and tweet.id_str not in self.tweets_dict[tweet.user.id_str].keys(): 
                self.tweets_dict[tweet.user.id_str][tweet.id_str] = tweet._json
                count += 1
            else:
                continue
            json.dump(self.tweets_dict, open(self.path_json,"w")) #save afterwards
        print("Scraped {} new tweets".format(count))
        
        
        
    def writeCSV(self, tweets_dict, cols = ["id_str","user-id_str"], path_csv = None):
        
        """
        This method converts list of .json files collected by the scraper into 
        a pandas DataFrame and saves it as a csv file.
        

        Parameters
        ----------
        tweets_dict : list of dictionaries
            Output of a scraping method.
        cols : List, optional
            Columns to be saved in the output. The default is ["id","user-id"].
        path_csv: string, optional 
            Path for the file to be saved in. If None, uses default filepath of the Class

        Returns
        -------
        df : pandas DataFrame
            Pandas dataframe containing selected data from scraping results.

        """
        
        
        
        #allow custom csv path
        if path_csv is None:
            path_csv = self.path_csv
            
            
        if os.path.exists(path_csv): 
            df = pd.read_csv(path_csv,index_col = 0)
        else:
            #make sure id_str and user-id_str are in columns
            cols = list(set(["id_str","user-id_str"] + cols))
            #create empty df
            df = pd.DataFrame(columns = cols,dtype = "object")
            
        #convert to list of all dictionary items. Keep only new tweets. 
        tweets = [tweet_dict for user_dict in tweets_dict.values() for tweet_dict in user_dict.values()]
        tweets = [tweet for tweet in tweets if tweet["id_str"] not in pd.unique(df["id_str"]).astype("str").tolist()]
    
        for tweet in tweets:
            
            
            #empty dict to store the desired data from each tweet in the json file
            tmp = dict()
        
            for col in cols: #iterate over values we want to gather
                            
                
                #keys to get from dict (for example, "user-id", will result in ["user","id"])
                keys = col.split('-')
                
                try:
                    values = tweet[keys.pop(0)]
                    while keys:
                        values = values[keys.pop(0)] #recursively replace the values until no keys left
                except:
                    continue
                    
                tmp[col] = values
                
            #add to dataframe
            df = df.append(tmp, ignore_index = True) #add to column

            
        df.to_csv(path_csv) #save 
        return df