#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:47:38 2020

@author: piotr
"""

import sys
import pickle
import os
sys.path.append('/home/piotr/projects/twitter/src')
from twitter_tools.scrapers import TwitterSampler
from twitter_tools.config import consumer_key, consumer_secret
from tweepy import API, AppAuthHandler
auth = AppAuthHandler(consumer_key = consumer_key, consumer_secret = consumer_secret)
api = API(auth, wait_on_rate_limit = True, wait_on_rate_limit_notify = True)







if __name__ == "__main__":
    
    path_sample = "/home/piotr/projects/twitter/data/sample"
    path_network = "/home/piotr/projects/twitter/data/network"
    
    scraper = TwitterSampler(api, path_log = "network_scraping.log")
    
    
    
    gov_sample = pickle.load(open(os.path.join(path_sample, "gov_sample.pickle"), "rb"))
    _ = scraper.getFriends(gov_sample, os.path.join(path_network, "gov_network_friends.json"))
    
    
    
    opp_sample = pickle.load(open(os.path.join(path_sample, "opp_sample.pickle"), "rb"))
    _ = scraper.getFriends(opp_sample, os.path.join(path_network, "opp_network_friends.json"))
