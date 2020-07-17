#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 11:28:25 2020

@author: piotr
"""

import os
import numpy as np
import sys
sys.path.append("..")
path = "/home/piotr/projects/twitter"
from twitter_tools.utils import filter_data
import json
import argparse
from functools import partial


        
        
if __name__ == "__main__":

    
    def filter_fun(data, prop_pol, drop = None, retweets = True, drop_duplicates = True, 
                   keep_cols = ["lemmatized", "id_str", "source", "day"]):
        """
        Filter function for the data reader.

        Parameters
        ----------
        data : pd.DataFrame.
        prop_pol : float
            Proportion of Polish tokens to keep.
        drop : dict, optional
            Key is name of column. Values list of values to be dropped. The default is None.
        retweets : bool, optional
            Keep retweets?. The default is True.
        drop_duplicates : bool, optional
            Drop duplicates (per day)?. The default is True.
        keep_cols : list, optional
            Which columns to keep. The default is ["lemmatized", "id_str", "source", "day"].

        Returns
        -------
        pd.DataFrame

        """
        if drop: #use a dict to drop problematic observations
            for k, v in drop.items():    
                    data = data[data[k].apply(lambda x: x not in v)]
        data = data[data["polish"].astype(float) >= prop_pol] #at least some proportion of Polish tokens
        if not retweets:
            data = data[np.logical_not(data["retweet"])]
        if drop_duplicates:
            data.drop_duplicates(inplace = True)
        return data[keep_cols]
        
    parser = argparse.ArgumentParser()
    parser.add_argument("path_source", help = "Path to source files")
    parser.add_argument("path_target", help = "Path to target file")
    parser.add_argument("nfiles", help = "Number of files", type = int)
    parser.add_argument("prop_pol", help = "Minimum proporiton of Polish per Tweet", type = float)
    parser.add_argument("--keep_cols", help = "Comma-separated columns to keep", type = str)
    parser.set_defaults(keep_cols = "lemmatized,id_str,source,day")
    parser.add_argument("--drop", help = "Json file with drop dictionary")
    parser.set_defaults(None)
    parser.add_argument('--use_retweets', dest='retweets', action='store_true')
    parser.add_argument('--drop_retweets', dest='retweets', action='store_false')
    parser.add_argument('--drop_duplicates', dest='drop_duplicates', action='store_true')
    parser.add_argument('--use_duplicates', dest='drop_duplicates', action='store_false')
    parser.set_defaults(retweets = False)
    parser.set_defaults(drop_duplicates = True)
    args = parser.parse_args()
    
    if args.drop is not None:
        args.drop = json.load(open(args.drop, "r"))
    
    fltr = partial(filter_fun, prop_pol = args.prop_pol, retweets = args.retweets,
                   keep_cols = [args.keep_cols.split(",")])
    filter_data(path_source = args.path_source, path_target = args.path_target, nfiles = args.nfiles, filter_fun = fltr)