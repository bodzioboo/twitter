#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 13:33:21 2020

@author: piotr
"""

import os
from tweepy.cursor import Cursor
import pandas as pd
import traceback
import numpy as np
import time

#class to get data of followers of a certain Twitter profile

class FollowerScraper:

    
    
    def __init__(self,screen_name, api, lang, cols, count):
        """
        Parameters
        ----------
        screen_name : name of the profile
        api :  Tweepy api object
        lang : Tweepy language abbreviation
        cols : columns to collect - follows the structure of Tweepy JSON files, where 
        "-" stands for dictionary depth level, 
        i.e. "entities-hashtags" will return hashtag values from dict entities
        count : number of users to extract data from

        Returns
        -------
        None.

        """
        
        self.ids = api.followers_ids(screen_name,count = count) #get user ids
        self.cols = np.unique(["id"] + cols) #set columns
        self.lang = lang
        self.api = api
    
    def scrape(self, filename):
        """
        

        Parameters
        ----------
        filename : name of the .csv file to save the data in

        Returns
        -------
        df : dataframe with the followers of a profile

        """
        if os.path.exists(filename): #create output file if doesn't exist
            df = pd.read_csv(filename,index_col = 0)
            row = df.shape[0]+1
        else:
            df = pd.DataFrame(columns = self.cols)
            row = 0
        for user_id in self.ids: #iterate over users
            try:
                for tweet in Cursor(self.api.user_timeline, user_id, rt = True).items(): #iterate over last 20 tweets of user
                    if tweet.id not in df["id"] and tweet.lang == self.lang: #check if tweet is not included
                        for col in self.cols: #iterate over values we want to gather
                            #get to the sufficient depth:
                            tmp = col.split('-') 
                            val = tweet._json[tmp[0]]
                            for i in range(1,len(tmp)):
                                try:
                                    val = val[tmp[i]]
                                except:
                                    continue
                            df.at[row, col] = val #add to column
                        row += 1
                    else:
                        continue
            except Exception:
                #traceback.print_exc()
                df.to_csv(filename) #save
                return df
        df.to_csv(filename) #save
        return df