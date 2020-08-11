#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 10:29:16 2020

@author: piotr
"""

import os
import pandas as pd
import json
import datetime
import time
import logging
import operator
import gc
import itertools
import functools
import collections
from tqdm import tqdm 
from tweepy import Cursor
import numpy as np
import dask.array as da
from .utils import check_errors





class Scraper:
    """
    This is a generic class containing some methods for scrapering tweets
    """
    
    def __init__(self, api, path_log:str = None):
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
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        fmt = logging.Formatter('%(asctime)s: %(message)s')
        sh = logging.StreamHandler()
        sh.setLevel(logging.ERROR)
        sh.setFormatter(fmt)
        self.logger.addHandler(sh)
        if path_log:
            fh = logging.FileHandler(path_log)
            fh.setLevel(logging.INFO)
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)
        self.logger.info("Initialized")
        
            
            
    def scrape(self, api_method, path:str, record:dict = None, 
               min_date:datetime.datetime = None, 
               limit:int = 0, output:str = "csv", cols:list = None, 
               verbose:bool = False, date_split:bool = False,
               **kwargs):
        """
        Generic method for scraping. Takes an api method and 
        uses Tweepy Cursor to collect the data.

        Parameters
        ----------
        api_method : tweepy.api method
            Which API method to use in Cursor
        path: str
            directory to write results in
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
            
        #define a writer partial function to write to either csv or txt    
        if output == "csv":
            writer = functools.partial(self._writeCSV, path_csv = path + ".csv", cols = cols)
        elif output == "txt":
            writer = functools.partial(self._writeTXT, path_txt = path + ".txt")
        else:
            raise ValueError("Invalid output string specified")
            
        
        if not record:
            record = []
            sinceid = None

        else:
            sinceid = max([int(value) for value in record])
         
        #maxid determines when to search to
        #updated after each iteration in first scrape so that the cursor goes
        #deeper whenever it's called
        maxid = None
        
        count = 0 #store tweets obtained
        error_count = 0 #count errors
            
        
        #boundary dates:
        if min_date is None: 
            min_date = datetime.datetime.now() - datetime.timedelta(hours = 36)
        tweet_date = min_date
        
            

        
        
        while tweet_date >= min_date:
            tweets_jsons = [] #store tweet data
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
                
                if verbose:
                    print(f"\rNumber of tweets scraped {count}", end = "")
                
                if new != 0:
                    self.logger.info("Scraped {} new tweets".format(new))
                    writer(tweets_jsons, date_split = date_split) #write to file
                
                elif new == 0:
                    #if there were no tweets to be evaluated, break while and return
                    self.logger.info("No more tweets to be scraped.")
                    break
        
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
        return record, count
    
    
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
    
    
    def _writeCSV(self, tweets_jsons, path_csv, cols = None, date_split = False):
        """
        

        Parameters
        ----------
        tweets_jsons : list of dict
            list of Twitter API jsons
        path_csv : str
            path to write output to
        cols : list, optional
            columns to write. The default is None.
        date_split : bool, optional
            Should files be split by date. The default is False.

        Returns
        -------
        None.

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
        
        df = check_errors(df)
        
        if date_split: #split by date
            
            date = pd.to_datetime(df.created_at).dt.date #get date series
            date = date.apply(lambda x: x.strftime('%Y_%m_%d'))
            for dt, df in df.groupby(date):
                
                folder, file = os.path.split(path_csv)
                file_date = file.split('.')[0] + '_' + dt + '.csv'
                subpath = os.path.join(folder, file_date)
                
                if not os.path.isfile(subpath):
                    #create new dataframe
                    if cols:
                        target = pd.DataFrame(columns = cols, dtype = str)
                    else:
                        target = pd.DataFrame(columns = df.columns, dtype = str)
            
                    #write new csv:
                    self.logger.info("Creating file {}".format(subpath))
                    target.to_csv(subpath, header = True)
                else:
                    #read the target:
                    target = pd.read_csv(subpath, nrows = 0, index_col = 0,
                                         header = 0, dtype = str)  
            
                
                self.logger.info("Appending to file {}. Got {} columns. Writing {}.".format(subpath, len(df.columns), len(target.columns)))
                df = df.reindex(columns = target.columns) #coerce to target shape
                df.to_csv(subpath, mode = "a", header = False)
                
        
        else: #write to one file
    
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
        
        
    def restoreRecord(self, path):
        """
        Used to restore scraping record if the json file is corrupt

        Parameters
        ----------
        path : path to the csv file containing scraped tweets

        Returns
        -------
        record : dict.

        """
        data = pd.read_csv(path, dtype = str, index_col = 0, 
                             usecols = ["Unnamed: 0", "id_str","user-id_str"])
        data = data.loc[data.index.map(lambda x: str(x).isnumeric()).tolist()]
        #zip user-tweet id pairs
        zipper = zip(data["user-id_str"].tolist(),data["id_str"].tolist())
        #sort by user id
        zipper = sorted(zipper, key = lambda x: str(x[0]))
        #create record dictionary
        record = dict()
        #iteratre over grouped entities and 
        for k, g in itertools.groupby(list(zipper), lambda x: x[0]):
            record[k] = list(map(operator.itemgetter(1), g))
        return record

                
        
           
        
        
        
    
    def scrape(self, sample, path, min_date = None, 
               limit = 0, cols = None, output = "csv", 
               date_split = False, **kwargs):
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
            which column to scrape from each user. Works only with output = "csv"
        output: string
            type of file to write to. 
            "csv" indicates a csv file with columns indicated by the cols argument; 
            "txt" indicates json strings in a csv file
        
            
        
        Returns
        -------
        None

        """
        
        #the record dict is used to keep track of tweet IDs scraper for each user
        path_record = path + ".json"
        if os.path.exists(path_record):
            
            #if json throws error, recover record using the csv file:
            try:
                record_dict = json.load(open(path_record, "r"))
            except json.JSONDecodeError:
                self.logger.exception("Recovering record dictionary")
                record_dict = self.restoreRecord(path + ".csv")
            record_dict = collections.defaultdict(lambda: None, record_dict)
        else:
            record_dict = collections.defaultdict(lambda: None)
            
        total_new = 0 #count new tweets scraped
        
        #iterator over users provided
        for user_id in tqdm(sample,position=0, leave=True): #iterate over users
            
            try:
                record, count = super().scrape(api_method = self.api.user_timeline, 
                                                       user_id = user_id, 
                                                       rt = True,
                                                       path = path, 
                                                       record = record_dict[str(user_id)],
                                                       min_date = min_date,
                                                       limit = limit, 
                                                       output = output, 
                                                       cols = cols, 
                                                       date_split = date_split)
                
                self.logger.info("Scraped user ID {}. Number of tweets {}.".format(user_id, count))
                total_new += count
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
        
        
        
        
        
        



class KeywordsScraper(Scraper):
    
    def scrape(self, keywords, path, min_date = None, 
               limit = 0, cols = None, output = "csv"):
        """
        

        Parameters
        ----------
        keywords : string
            keywords to scrape
        path : string
            path to scrape to without extension
        
        **kwargs : kwargs for the  generic scraper method

        Returns
        -------
        None.

        """
        
        #the record is used to keep track of tweet IDs 
        path_record = path + "_record.txt"
        if os.path.exists(path_record):
            with open(path_record,"r") as f:
                record = f.read().splitlines()
        else:
            record = []
            
        
            
        try:
            record, count = super().scrape(api_method = self.api.search, 
                                                   q = keywords,
                                                   rt = True,
                                                   path = path, 
                                                   record = record,
                                                   min_date = min_date,
                                                   limit = limit, 
                                                   output = output, 
                                                   cols = cols, 
                                                   verbose = True)
            #store record
            with open(path_record,"w") as f:
                f.write([rec + "\n" for rec in record], "w")
                
        
        #log and go to next id on exceptions
        except Exception as e:
            self.logger.exception("Failed. Error \n{}.".format(e.args[0]))
                
            
        #DONE:
        print("Total tweets scraped {}".format(count))
        self.logger.info("Finished scraping. Total {}".format(count))        
        
        
        

        
        
        
class TwitterSampler(Scraper):
    
            
        
        
    def getFollowers(self, profiles, path):
        """
        Gets ids of profile followers.
        

        Parameters
        ----------
        profiles : list of strings
            Profile ids
        path: string
            path to store result in

        Returns
        -------
        Dictionary of profiles associated with their followers IDs.

        """
        self.logger.info(f"Scraping to file {path}")
        if os.path.exists(path):
            followers = json.load(open(path,"r"))
        else:
            followers = dict()
        for profile in tqdm(profiles):
            if profile in followers:
                continue
            followers[profile] = []
            while True:
                try:
                    for follower_id in Cursor(self.api.followers_ids, 
                                                   user_id = profile).pages():
                        followers[profile].extend(follower_id)
                except Exception as e:
                    self.logger.exception("Error {}".format(e.args[0]))
                    print("Error. Sleeping 60 seconds")
                    time.sleep(60)
                break
            self.logger.info("Scraped followers of profile {}".format(profile))
            json.dump(followers, open(path,"w")) #save
        return followers
    
    
    def getFriends(self, profiles, path):
        
        
        """
        Gets ids of profile followers.
        

        Parameters
        ----------
        profiles : list of strings
            Profile ids
        path: string
            path to store result in

        Returns
        -------
        Dictionary of profiles associated with their followers IDs.

        """
        self.logger.info(f"Scraping to file {path}")
        if os.path.exists(path):
            friends = json.load(open(path,"r"))
        else:
            friends = dict()
        for profile in tqdm(profiles):
            if profile in friends:
                continue
            friends[profile] = []
            while True:
                try:
                    for friend_id in Cursor(self.api.friends_ids, 
                                                   user_id = profile).pages():
                        friends[profile].extend(friend_id)
                except Exception as e:
                    self.logger.exception(f"Error {e.args[0]} Sleeping 60 seconds")
                    time.sleep(60)
                break
            self.logger.info("Scraped friends of profile {}".format(profile))
            json.dump(friends, open(path,"w")) #save
        return friends
        
            
            
    
    
    
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
        
        
        
        
        
        
        
        
        
        
  



