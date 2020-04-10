#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 10:29:16 2020

@author: piotr
"""

import os
import pandas as pd
import numpy as np
import dask.array as da
import json
import datetime
import time
import logging
import gc
import itertools
import functools
import collections
from tqdm import tqdm 
import random
from tweepy import Cursor



class Scraper:
    """
    This is a generic class containing some methods for scrapering tweets
    """
    
    def __init__(self, api, path_log):
        """
        

        Parameters
        ----------
        api : tweepy.api
            initialized Tweepy API 
        path_log : str
            Path to the log file.

        Returns
        -------
        None.

        """
        
        #Log setup
        self.api = api
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(path_log)
        formatter = logging.Formatter('%(asctime)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info("Initialized")
        
            
            
    def scrape(self, api_method, record = None, min_date = None, limit = 0, **kwargs):
        """
        Generic method for scraping. Takes an api method and 
        uses Tweepy Cursor to collect the data.

        Parameters
        ----------
        api_method : tweepy.api method
            Which API method to use in Cursor
        record : dict
            Record of tweets scraped in the past. Keys are user ids, values are lists
            of twitter ids
        min_date: datetime object
            What date to scrape from.
        limit: int
            limit for the cursor method. How many tweets to retrieve per user.
        **kwargs : 
            keyword args to be passed to the tweepy.api method

        Returns
        -------
        tweets_jsons: list
            list of tweepy dictionaries
        record: dict
            lists of scraped tweets for each user
        count: int
            Number of tweets scraped

        """
        #sinceid determines since when to search
        if (record is None or not record):
            record = []
            sinceid = None

        else:
            sinceid = max([int(value) for value in record])
         
        #maxid determines when to search to
        #updated after each iteration in first scrape so that the cursor goes
        #deeper whenever it's called
        maxid = None
            
        tweets_jsons = [] #store tweet data
        count = 0 #store tweets obtained
        if min_date is None: 
            min_date = datetime.datetime.now() - datetime.timedelta(hours = 36)
        tweet_date = min_date
        
        error_count = 0
        
        
        while tweet_date >= min_date:
        #keep scraping until a tweet reaches the minimum date specified
            try: 
                cursor = Cursor(api_method,tweet_mode = "extended", 
                                    since_id = sinceid,
                                    max_id = maxid,
                                    **kwargs).items(limit)
                new = 0
                for tweet in cursor: #iterate over limit
                    tweet_date = tweet.created_at
                    if tweet.id_str not in record and tweet.created_at >= min_date:
                        tweets_jsons.append(tweet._json)
                        record.append(tweet.id_str)
                        new += 1
                count += new
                self.logger.info("Scraped {} new tweets".format(new))
        
                
                if new == 0:
                    #if there were no tweets to be evaluated, break while and return
                    self.logger.info("No more tweets to be scraped.")
                    break
        
                if sinceid is None:
                    #update max id - only during first scrape
                    maxid = tweet.id_str
        
            except Exception as e:
                #on error, try again three times and if still no luck, finish scraping
                if error_count < 3:
                    self.logger.exception("Error: \n{}. Waiting 60 seconds".format(e.args[0]))
                    error_count += 1
                    time.sleep(60)
                    continue
                else:
                    self.logger.exception("Error: \n{}. Failed to scrape.".format(e.args[0]))
                    break
                
                
            
        gc.collect()
        return tweets_jsons, record, count
    
    
    #utility function for recursively traversing the json file into a one level dictionary
    def _traverse_json(self, oldDict, newDict = dict()):
        """    
        This method returns a Twitter json dict with depth of one. Allows to write the dict
        into a CSV file.
        
        Parameters
        ----------
        oldDict : DICTIONARY
            Twitter json dictionary to traverse
        newDict : DICTIONARY
            should be explicitly passed empty

        Returns
        -------
        newDict:
            traversed dictionary

        """
        for k in list(oldDict):
            if type(oldDict[k]) != dict:
                newDict[k] = oldDict.pop(k)
            else:
                tmp = k
                oldDict[k] = {tmp + "-" + key:value for key, value in oldDict[k].items()}
                self._traverse_json(oldDict[k],newDict)
        return newDict
    
    
    def _writeCSV(self, tweets_jsons, path_csv, cols = None):
        """
        This method converts list of traversed .json files 
        collected by the scraper into 
        a pandas DataFrame and writes it to a csv file.
    
    
        Parameters
        ----------
        tweets_dict : list of dictionaries
            Output of a scraping method.
    
        path_csv: string
            path to write the csv file to
        Returns
        -------
        None
    
        """
        assert(type(tweets_jsons) == list)
        if not tweets_jsons:
            self.logger.info("List empty. Not writing Anything")
            return None
        
        
        #store tweets
        json_traversed = []
    
        #"traverse" the json dictionary to convert to data frame
        for tweet_json in tweets_jsons:
            json_traversed.append(self._traverse_json(tweet_json, dict()))
        df = pd.DataFrame(json_traversed, dtype = str) #convert to dataframe with default string type
        
        
    
        if not os.path.isfile(path_csv):
            #create new dataframe
            if cols:
                target = pd.DataFrame(columns = cols, dtype = str)
            else:
                target = pd.DataFrame(columns = df.columns, dtype = str)
    
            #write new csv:
            self.logger.info("Creating file {}".format(path_csv))
            target.to_csv(path_csv, header = True)
        else:
            #read the target:
            target = pd.read_csv(path_csv, nrows = 0, index_col = 0,
                                 header = 0, dtype = str)  
    
        
        self.logger.info("Appending to file {}. Got {} columns. Writing {}.".format(path_csv, len(df.columns), len(target.columns)))
        df = df.reindex(columns = target.columns) #coerce to target shape
        df.to_csv(path_csv, mode = "a", header = False)
        
        
        
    def _writeTXT(self, tweets_jsons, path_txt):
        """
        This method writes list Twitter API json dictionaries 
        to txt file as strings. 

        Parameters
        ----------
        tweets_jsons : list of dictionaries
            output of the scraping method
        path_txt : string
            path to write the txt file to

        Returns
        -------
        None.

        """
        assert(type(tweets_jsons) == list)
        if not tweets_jsons:
            return None
        mode = "a" if os.path.isfile(path_txt) else "w"
        
        with open(path_txt, mode) as f:
            for js in tweets_jsons:
                f.write(json.dumps(js) + "\n")
    
    def _TXT2CSV(self, path_txt, path_csv, n = 10000, **kwargs):
        """
        This method converts the txt file of json strings into a csv file
        

        Parameters
        ----------
        path_txt : string
            path to read txt from
        path_csv : string
            path to write csv to
        n : integer, optional
            Number of jsons per call of the csv writing method. 
            The default is 10000.
        **kwargs : arguments passed to the _writeCSV method. 
        Returns
        -------
        None.

        """
        with open(path_txt, "r") as f:
            for _ in itertools.count():
                tweets_jsons = [x.strip() for x in itertools.islice(f, n)]
                if not tweets_jsons:
                    break
                self._writeCSV(tweets_jsons, path_csv, **kwargs)
                
                
            
        


        
        
class FollowerScraper(Scraper):
    
    """
    This class includes tools to scrape Followers of a list of Twitter profiles.
    """

    
    def __init__(self, api, path_log):
        """
        

        Parameters
        ----------
        api : tweepy API object

            
        path_log: string
            path for the log file.

        Returns
        -------
        None.

        """
        super().__init__(api, path_log)
        
        
                    
    
    def getFollowers(self, profiles, path):
        """
        Gets ids of profile followers.
        

        Parameters
        ----------
        profiles : list of strings
            Profile names to scrape from
        path: string
            path to store result in

        Returns
        -------
        Dictionary of profiles associated with their followers IDs.

        """
        
        if os.path.exists(path):
            followers = json.load(open(path,"r"))
            if set(profiles) != set(followers.keys()):
                raise ValueError("Profiles don't match the target dictionary")
        else:
            followers = dict()
            for profile in profiles:
                followers[profile] = []
                while True:
                    try:
                        for follower_id in tqdm(Cursor(self.api.followers_ids, screen_name = profile).pages(),position=0, leave=True):
                            followers[profile].extend(follower_id)
                    except Exception as e:
                        self.logger.exception("Error {}".format(e.args[0]))
                        print("Error. Sleeping 60 seconds")
                        time.sleep(60)
                        continue
                    break
                self.logger.info("Scraped followers of profile {}".format(profile))
                json.dump(followers, open(path,"w")) #save
        return followers
            
            
    def sampleFollowers(self, followers, num_users, min_usr_tweets = 1, 
                        last_date_active = None, lang = None):
        """
        

        Parameters
        ----------
        followers : list
            DESCRIPTION.
        num_users: int
            number of users to sample
        min_user_tweets: int
            criterion: minimun number of tweets to include
        last_date_active : datetime object
            criterion: minimum last activity of the user profile
        path: string
            path to store sample in
            
        

        Returns
        -------
        sample : list
            sample of users from followers that specify the criteria.

        """
        if last_date_active is None:
            last_date_active == datetime.datetime.now() - datetime.timedelta(days = 1)

        
        #remove duplicates and shuffle
        followers = list(set(followers))
        sample = list() #store sample
        print("Sampling users")
        
        #while sample not complete
        while len(sample) < num_users and len(followers) > 0:
            print("\rUsers sampled {}. Population remaining {}".format(len(sample),len(followers)), end = "")
            #sample random index
            random_index = random.randint(0, len(followers))
            try:
                random_user = self.api.get_user(followers.pop(random_index))
            #ensure conditions are met:
                if (not random_user.protected and 
                    random_user.statuses_count > min_usr_tweets and 
                    random_user.status.created_at > last_date_active and 
                    random_user.lang in lang):
                    sample.append(random_user.id) #add id to sample - check this
            except Exception as e:
                self.logger.exception("Exception in sampling. Error: {}".format(e.args[0]))
                time.sleep(60)
                continue
        self.logger.info("Finished sampling. Sampled {} users".format(len(sample)))
        return sample
    
    
    
    #for each element of the superset, determine in how many subsets it appears
    def _howMany(self,superset,subsets):
        """
        

        Parameters
        ----------
        superset : list of unique twitter IDs
        subsets : list of lists of twitter IDs

        Returns
        -------
        count : np.array of counts for each value of the superset

        """
        superset = np.array(superset)[:,np.newaxis] #convert to Nx1 nparray
        superset = da.from_array(superset, chunks = (2000,1)) #convert to dask
        count = np.zeros(superset.shape[0]).astype("int8")
        for subset in tqdm(subsets):
            tmp = da.from_array(subset)
            count += (superset == tmp).sum(axis = 1).compute().astype("int8")
            gc.collect()
        return count
    
    
    #function to filter out elements of superset that occur in at most/least n subsets  
    def subsetFollowers(self, followers1, followers2, least, most):
        """
        Given two populations of Twitter profile IDs, this method returns
        two non-overlapping sub-populations that meet the specified subsetting conditions. 
        The conditions determine minimum number of recurrence in its own population and
        maximum number of recurrences in the other population. 
        A population is defined as a number of profile IDs and IDs of the followers
        of these profiles.


        Parameters
        ----------
        followers1 : dict 
            Output of the getFollowers method. dict of followers of profiles (key - followed profiles; values - list of 
            associated IDs). 
        followers2 : dict
            Output of the getFollowers method. dict of followers of profiles (key - followed profiles; values - list of 
            associated IDs). 
        least : int
            minimum number of times a given ID occurs in its own group
        most : int
            maximum number of times a given ID occurs in the other group

        Returns
        -------
        LIST
            IDs from followers1 that met the subsetting conditions.
        LIST
            IDs from followers2 that met the subsetting conditions.

        """
        
        #get list of subsets and superset of both
        f1_superset = list(set(itertools.chain.from_iterable(followers1.values())))
        f1_subsets = list(followers1.values())
        f2_superset = list(set(itertools.chain.from_iterable(followers2.values())))
        f2_subsets = list(followers2.values())
        
        #get counts
        f1_in_f1 = self._howMany(f1_superset, f1_subsets) #how many gov profiles followed by each gov follower
        f1_in_f2 = self._howMany(f1_superset, f2_subsets) #how many opp profiles followed by each gov follower
        f2_in_f2 = self._howMany(f2_superset, f2_subsets) #how many opp profiles followed by each opp follower
        f2_in_f1 = self._howMany(f2_superset, f1_subsets) #how many gov profiles followed by each opp follower
        
        f1_ind = ((f1_in_f1 >= least) & (f1_in_f2 <= most))
        f2_ind = ((f2_in_f2 >= least) & (f2_in_f1 <= most))
        
        
        return np.array(f1_superset)[f1_ind].tolist(), np.array(f2_superset)[f2_ind].tolist()
    
    
    def getFollowersData(self, followers, path):
        """
        Scrape data from list of IDs using the Tweepy API.get_user method and
        write them to CSV

        Parameters
        ----------
        followers : list
            list of user IDs
        path : string
            path to CSV file.
        Returns
        -------
        None.

        """
        followers_data = []
        for follower in tqdm(followers):
            try:
                followers_data.append(self.api.get_user(follower)._json)
            except Exception as e:
                self.logger.exception("Exception in scraping at ID {}. Error: {}".format(follower,e.args[0]))
                time.sleep(60)
                continue
        self._writeCSV(followers_data, path)
                
        
           
        
        
        
    
    def scrape(self, sample, path, min_date = None, limit = 0, cols = None, csv = True):
        """
        Scrape tweets from a list of user IDs bweteen a specified date to now and
        write them to output file.

        Parameters
        ----------
        sample: list of str
            List of user IDs
        path: string
            path to store record and output
        min_date: datetime object
            what date to scrape from
        limit: int
            how many tweets to scrape in one iteration.
        cols: list of str or None
            which column to scrape from each user
        
            
        
        Returns
        -------
        None

        """
        
        #the record dict is used to keep track of tweet IDs scraper for each user
        path_record = path + ".json"
        if os.path.exists(path_record):
            record_dict = json.load(open(path_record, "r"))
            record_dict = collections.defaultdict(lambda: None, record_dict)
        else:
            record_dict = collections.defaultdict(lambda: None)
            
        #define a writer partial function to write to either csv or txt    
        if csv:
            writer = functools.partial(super()._writeCSV, path_csv = path + ".csv", cols = cols)
        else:
            writer = functools.partial(super()._writeTXT, path_txt = path + ".txt")
            
        total_new = 0 #count new tweets scraped
        
        #iterator over users provided
        for user_id in tqdm(sample,position=0, leave=True): #iterate over users
            
            try:
                tweets, record, count = super().scrape(api_method = self.api.user_timeline, 
                                                       record = record_dict[str(user_id)],
                                                       min_date = min_date,
                                                       limit = limit, user_id = user_id,
                                                       rt = True)
                
                self.logger.info("Scraped user ID {}. Number of tweets {}.".format(user_id, count))
                total_new += count
                writer(tweets) #write
                record_dict[user_id] = record 
                json.dump(record_dict, open(path_record, "w")) #save record
            
            #log and go to next id on exceptions
            except Exception as e:
                self.logger.exception("Failed on user {}. Error \n{}.".format(user_id, e.args[0]))
                continue
                
            
        #DONE:
        print("Total tweets scraped {}".format(total_new))
        json.dump(record_dict, open(path_record, "w"))
        self.logger.info("Finished scraping. Total {}".format(total_new))
        
        
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
  



