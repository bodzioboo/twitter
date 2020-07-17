#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 16:59:14 2020

@author: piotr
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 13:02:58 2020

@author: piotr
"""

import datetime
import pickle
import argparse
import os
import sys
sys.path.append("..")
from tweepy import API, AppAuthHandler
from twitter_tools.config import consumer_key, consumer_secret
from twitter_tools.scrapers import FollowerScraper
import logging
import pdb


if __name__ == "__main__":
    
    
    
    path = "/home/piotr/projects/twitter"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("date", help = "YYYY/mm/dd format")
    parser.add_argument("-s", "--shutdown", action = "store_true")
    parser.add_argument("-what", type = str, default = 'both')
    args = parser.parse_args()
    scrape_from = datetime.datetime.strptime(args.date,"%Y/%m/%d")
    
    
    
    auth = AppAuthHandler(consumer_key, consumer_secret)
    api = API(auth, wait_on_rate_limit = True, wait_on_rate_limit_notify = True) 
    
    #initialize scraper
    twitter_columns = pickle.load(open(os.path.join(path, 'data/twitter_columns.pickle'),"rb"))
    scraper = FollowerScraper(api = api, path_log = "logs/scraper_new.log")
    
    
    if args.what == 'gov' or args.what == 'both':
    
    
        #government
        gov_ids = pickle.load(open(os.path.join(path, "data/sample/gov_sample.pickle"),"rb"))
        scraper.scrape(gov_ids, path = os.path.join(path, "data/scraped/gov_tweets"),
                   min_date = scrape_from, date_split = True, 
                   limit = 50, cols = twitter_columns)
    if args.what == 'opp' or args.what == 'both':
        
        #opposition:
        opp_ids = pickle.load(open(os.path.join(path, "data/sample/opp_sample.pickle"),"rb"))
        scraper.scrape(opp_ids, path = os.path.join(path, "data/scraped/opp_tweets"),
                   min_date = scrape_from, date_split = True,
                   limit = 50, cols = twitter_columns)
    
    #shutdown
    if args.shutdown:
        os.system("shutdown")