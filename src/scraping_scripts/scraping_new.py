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
from tweepy import API, AppAuthHandler
from twitter_tools.config import consumer_key, consumer_secret
from twitter_tools.scrapers import FollowerScraper

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("date", help = "YYYY/mm/dd format")
    parser.add_argument("-s", "--shutdown", action = "store_true")
    args = parser.parse_args()
    scrape_from = datetime.datetime.strptime(args.date,"%Y/%m/%d")
    
    
    
    auth = AppAuthHandler(consumer_key, consumer_secret)
    api = API(auth, wait_on_rate_limit = True, wait_on_rate_limit_notify = True) 
    
    #initialize scraper
    twitter_columns = pickle.load(open("/home/piotr/data/twitter_columns.pickle","rb"))
    scraper = FollowerScraper(api = api, path_log = "scraper_new.log")
    
    
    #government
    gov_ids = pickle.load(open("/home/piotr/data/sample/gov_ids.pickle","rb"))
    scraper.scrape(gov_ids[:5000],path = "/home/piotr/data/scraped/gov_tweets",
               min_date = scrape_from,
               limit = 50, cols = twitter_columns)
    
    opp_ids = pickle.load(open("/home/piotr/data/sample/opp_ids.pickle","rb"))
    scraper.scrape(opp_ids[:5000], path = "/home/piotr/data/scraped/opp_tweets",
               min_date = scrape_from, 
               limit = 50, cols = twitter_columns)
    
    
    
    
    #shutdown
    if args.shutdown:
        os.system("shutdown")