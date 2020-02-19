#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 13:10:41 2019

@author: piotr
"""

import os
from tweepy.cursor import Cursor
import pandas as pd
import traceback
import numpy as np
import time

#Scrape fronda followers:

class FollowerScraper:
    
    def __init__(self,screen_name, api, lang, cols, count):
        self.ids = api.followers_ids(screen_name,count = count)
        self.cols = np.unique(["id"] + cols)
        self.lang = lang
        self.api = api
    
    def scrape(self, filename):
        if os.path.exists(filename):
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
                traceback.print_exc()
                df.to_csv(filename) #save
                return df
        df.to_csv(filename) #save
        return df
    

def convert_date(twitter_date, fmt = '%Y-%m-%d %H:%M:%S'):
    date_converted = [time.strftime(fmt, time.strptime(date,'%a %b %d %H:%M:%S +0000 %Y')) for date in twitter_date]
    return np.array(date_converted)

def convert_hashtag(twitter_hashtags):
    hashtags = []
    for elem in twitter_hashtags:
        if elem != []:
            text = [hashtag['text'] for hashtag in elem]
            text = " ".join(text)
        else:
            text = " "
        hashtags.append(text)
    return np.array(hashtags)


def convert_mention(twitter_mentions):
    pass
                
    

