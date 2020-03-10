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

    
    
    def __init__(self,screen_name, api, lang, cols, count, limit):
        """
        Parameters
        ----------
        screen_name : name of the profile
        
        api :  Tweepy api object
        
        lang : Tweepy language abbreviation
        
        cols : columns to collect - follows the structure of Tweepy JSON files, where 
        "-" stands for dictionary depth level, 
        i.e. "entities-hashtags" will return hashtag values from dict entities. 
        
        count : number of users to extract data from
        limit: how many tweets per user should the Cursor iterate over

        Returns
        -------
        None.

        """
        
        self.ids = api.followers_ids(screen_name,count = count) #get user ids
        self.cols = np.unique(["id"] + ["user-id"] + cols) #set columns
        self.limit = limit #how many tweets per user to get
        self.lang = lang
        self.api = api
        self.errors = []
    
    def scrape(self, filename):
        """
        

        Parameters
        ----------
        filename : name of the .csv file to save the data in

        Returns
        -------
        df : dataframe with the followers of a profile

        """
        
        #set up saving the data:
        if os.path.exists(filename): #create output file if doesn't exist
            df = pd.read_csv(filename,index_col = 0)
            old_row = df.shape[0]+1
        else:
            df = pd.DataFrame(columns = self.cols,dtype = "object")
            old_row = 0
        new_row = 0
        
        
        for user_id in self.ids: #iterate over users
            print("scraping id",user_id)
            
            
            #skip protected users
            if self.api.get_user(user_id).protected:
                continue
            
            #get last id of tweet to limit the scope of the query:
            try:
                lastid = df[df["user-id"] == user_id].id.min()
            except:
                lastid = None
                
            #scrape. If something goes wrong, swtich to next id
            try:
                for tweet in Cursor(self.api.user_timeline, user_id, 
                                    rt = True, tweet_mode = "extended", 
                                    id_since = lastid).items(self.limit): #iterate over limit
                    if tweet.lang == self.lang: #check if language matches
                        for col in self.cols: #iterate over values we want to gather
                            
                            #get to the sufficient depth:
                            tmp = col.split('-')
                            try:
                                val = tweet._json[tmp[0]] #get attribute
                            except:
                                continue
                            for i in range(1,len(tmp)):
                                try:
                                    val = val[tmp[i]]
                                except:
                                    continue
                                
                            try:
                                df.at[old_row + new_row, col] = val #add to column
                            except Exception as e:
                                self.errors.append(str(e))
                                continue
                        new_row += 1
                    else:
                        continue
                df.to_csv(filename) #save on each id
            except Exception as e:
                print("Scraping failed")
                self.errors.append(str(e))
                continue
        df.to_csv(filename) #save
        print("Scraped {} tweets".format(new_row))
        return df