#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 10:29:16 2020

@author: piotr
"""

#class to get data of followers of a certain Twitter profile
from .Scraper import Scraper

class FollowerScraper(Scraper):

    
    def __init__(self, screen_name, api, lang, path, count, limit):
        """
        

        Parameters
        ----------
        screen_name: string
            Twitter username beginning with @
            
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
        super().__init__(api, lang, path, count, limit)
        self.ids = [str(user_id) for user_id in api.followers_ids(screen_name,count = count)] #get user ids
        
    
    def scrape(self, return_df = True, **kwargs):
        """
        Scrape tweets from followers of the account and store in json file.

        Parameters
        ----------
            
        return_df: boolean.
            Deafault value is True. Whether to return (and save) dataframe. If not, returns dictionary.
            
        **kwargs:
            for now it takes in column names to pass to the writeCSV method

        Returns
        -------
        Pandas Data Frame or Dictionary, depending on the return_df argument.

        """
        
        
        for user_id in tqdm(self.ids): #iterate over users
            
            try:
                #skip protected users
                if self.api.get_user(user_id).protected:
                    continue
            except Exception as e:
                self.errors[user_id] = str(e)
                continue

            
                
            #scrape. If something goes wrong, swtich to next id
            try:
                super().scrape(api_method = self.api.user_timeline, path = self.path_json,
                               user_id = user_id, rt = True)
            except Exception as e:
                print("Scraping failed")
                self.errors[user_id] = str(e)
                continue
            
        #DONE:
        if return_df:
            #write csv and return dataframe
            tweets = super().writeCSV(self.tweets_dict, **kwargs)
            return tweets
        else:
            return self.tweets_dict