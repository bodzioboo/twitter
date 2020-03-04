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
                
    

