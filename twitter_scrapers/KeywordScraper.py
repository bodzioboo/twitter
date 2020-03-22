#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 14:45:30 2020

@author: piotr
"""
from .Scraper import Scraper

class KeywordScraper(Scraper):
    
    def __init__(self, searchwords, api, lang, path, count, limit):
        """
        

        Parameters
        ----------
        searchwords: string
            words/hashtags to scrape. For example "#twitter OR #DonaldTrump" for either
            #twitter or #DonaldTrump
            
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
        super().__init__(api,  lang, path, count, limit)
        self.searchwords = searchwords #get user ids
        
        
    def scrape(self, return_df = True, **kwargs):
        
        """
        Scrape tweets using a keyword and store in json file.
    
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
        super().scrape(self.api.search, self.path_json, q = self.searchwords, 
                            rt = True)
            
        #DONE:
        if return_df:
            #write csv and return dataframe
            tweets = super().writeCSV(self.tweets_dict, **kwargs)
            return tweets
        else:
            return self.tweets_dict
        
    
    
        