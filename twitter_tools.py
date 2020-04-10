#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 13:10:41 2019

@author: piotr
"""

import os
import pandas as pd
import traceback
import numpy as np
import time
import re

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
                
    

def tweet_stats(twitter_text):
    
    """
    Function for counting stuff in tweets. Takes in a list, returns a dict.
    
    """
    
    
    tweets = list(twitter_text)
    
    counts = dict()
    
    #count URls
    re_url = re.compile(r'(?:https?\:\/\/)?(?:www\.)?[a-zA-Z0-9-]+\.(?:(?:[a-zA-Z0-9-]+\.)*)?[a-z]{2,4}(?:(?:\/\S+)*)?')
    urls = [re_url.findall(x) for x in tweets]
    counts["urls"] = [len(x) for x in urls]
    
    #remove urls (so they don't interfere with other counts)
    tweets = [tweet.lower() for tweet in tweets] #lowercase
    tweets = [re.sub(r'(?:https?\:\/\/)?(?:www\.)?[a-zA-Z0-9-]+\.(?:(?:[a-zA-Z0-9-]+\.)*)?[a-z]{2,4}(?:(?:\/\S+)*)?',' ',tweet) for tweet in tweets]
    #remove extra whitespaces:
    tweets = [re.sub(r' +',' ',tweet) for tweet in tweets]
    
    #get counts of emojis and emoticons
    re_emoji = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251" 
    "]+"
)
    emojis = [re_emoji.findall(x) for x in tweets]
    counts["emoti"] = [len(x) for x in emojis]
    
    re_emoticons_pos = re.compile(r"[oO>]?[;:=Xx8]+'?\-?[)>)pPdD3\]\}\*]+")
    emoticons_pos = [re_emoticons_pos.findall(x) for x in tweets]
    counts["emoti_pos"] = [len(x) for x in emoticons_pos]
    
    re_emoticons_neg = re.compile(r"[oO>]?[:;=]+'?\-?[(<\\\/\[(\{]+")
    emoticons_neg = [re_emoticons_neg.findall(x) for x in tweets]
    counts["emoti_neg"] = [len(x) for x in emoticons_neg]

    re_emoticons_neut = re.compile(r"[oO>]?[:;=]+'?\-?[oO]+")
    emoticons_neut = [re_emoticons_neut.findall(x) for x in tweets]
    counts["emoti_neut"] = [len(x) for x in emoticons_neut]
    
    #get hashtags counts
    re_hashtag = re.compile(r'\#[a-zA-Z0-9]+')
    hashtags = [re_hashtag.findall(x) for x in tweets]
    counts["hashtags"] = [len(x) for x in hashtags]
    
    #get mention counts
    re_mention = re.compile(r'\@\w+(?!\.\w+)\b')
    mentions = [re_mention.findall(x) for x in tweets]
    counts["mentions"] = [len(x) for x in mentions]
    
    #get number counts
    re_number = re.compile(r"(\S+)?\d+(\S+)?")
    numbers = [re_number.findall(x) for x in tweets]
    counts["numbers"] = [len(x) for x in numbers]
    
    return counts





def preprocess_tweets(twitter_text,url_token = False, emoji_token = False, numbers_token = False, hashtag_token = False, mention_token = False):
    tweets = list(twitter_text) #convert to list
    tweets = [tweet.lower() for tweet in tweets] #lowercase
    
    url = emoji = emoticon_pos = emoticon_neg = emoticon_neut = numbers = hashtag = mention = ' ' 
    
    if url_token:
        url = " urltoken "
    
    if emoji_token:
        emoji = " emojitoken "
        emoticon_pos = " emoticon_pos "
        emoticon_neg = " emoticon_neg "
        emoticon_neut = " emoticon_neut "
    
    if numbers_token:
        numbers = " numbertoken "
    
    if hashtag_token:
        hashtag = " hashtagtoken "
    
    if mention_token:
        mention = " mentiontoken "
    
    
    #urls
    url_regex = r"(?:https?\:\/\/)?(?:www\.)?[a-zA-Z0-9-]+\.(?:(?:[a-zA-Z0-9-]+\.)*)?[a-z]{2,4}(?:(?:(\/|\?)\S+)*)?"
    tweets = [re.sub(url_regex, url ,tweet) for tweet in tweets]

        
        
    #emoticons and emojis
    re_emoji = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251" 
    "]+")
    
    tweets = [re.sub(re_emoji, emoji, tweet) for tweet in tweets]
    tweets = [re.sub(r"[oO>]?[;:=Xx8]+'?\-?[)>)pPdD3\]\}\*]+", emoticon_pos, tweet) for tweet in tweets]
    tweets = [re.sub(r"[oO>]?[:;=]+'?\-?[(<\\\/\[)\{]+", emoticon_neg, tweet) for tweet in tweets]
    tweets = [re.sub(r"[oO>]?[:;=]+'?\-?[oO]+", emoticon_neut, tweet) for tweet in tweets]
    
    
    #remove words of length == 1
    tweets = [re.sub(r'\b(\w)\b',' ',tweet) for tweet in tweets]
    
    #digits
    tweets = [re.sub(r"(\S+)?\d+(\S+)?", numbers ,tweet) for tweet in tweets]
        
        
    #rt flags
    tweets = [re.sub(r"^rt",' ', tweet) for tweet in tweets]
    
    #newline
    tweets = [re.sub("\\n",' ', tweet) for tweet in tweets]
    
    #hashtags
    tweets = [re.sub(r'\#[a-zA-Z0-9]+', hashtag, tweet) for tweet in tweets]
    
    #usermentions
    tweets = [re.sub(r'\@\w+(?!\.\w+)\b', mention, tweet) for tweet in tweets]
    
    #non alphanumeric
    tweets = [re.sub(r"\&amp"," ",tweet) for tweet in tweets] #bug in twitter - &amp appearing
    tweets = [re.sub(r"\W+"," ",tweet) for tweet in tweets]
    #tweets = [re.sub(r"[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~\…\–]+"," ",tweet) for tweet in tweets]
    #underscores
    tweets = [re.sub(r"_+"," ",tweet) for tweet in tweets]
    
    
    #remove extra whitespaces:
    tweets = [re.sub(r' +',' ',tweet) for tweet in tweets]
    tweets = [tweet.strip() for tweet in tweets]
    
    return tweets