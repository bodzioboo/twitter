#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 14:32:52 2020

@author: piotr
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 11:03:15 2020

@author: piotr
"""

import warnings
warnings.filterwarnings('ignore')
import os
import pickle
import json
import sys
sys.path.append("..")
from twitter_models.embeddings import KTopicModel
import argparse
import pandas as pd
from itertools import product
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
import logging

        
if __name__ == "__main__":
    
    #logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    
    #arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument("path_data", help = "Path to file")
    parser.add_argument("path_results", help = "Path to store results in")
    parser.add_argument("path_embeddings", help = "Path to word embeddings")
    parser.add_argument("path_params", help = "Json file with parameters")
    parser.add_argument("col_text", help = "Name of the text column")
    parser.add_argument("--batch_size", help = "Batch size for predicting clusters", 
                        type = int, default = 100000)
    parser.add_argument("--train_size", help = "Train size for the model", 
                        type = float, default = 0.1)
    parser.set_defaults(pickle = False)
    args = parser.parse_args()
    

    #set up search grid
    params = json.load(open(args.path_params, "r"))
    keys = params.keys(); combs = product(*params.values()) #keys + all combinations of parameters
    grid = [dict(zip(keys, elem)) for elem in combs] #zip these toegether into a list of dicts
    folder_grid, file_grid = os.path.split(args.path_params) #path for storing the JSON with results analysis
    path_results = os.path.join(folder_grid, file_grid.split(".")[0] + "_results_" + ".json")
    
    for i, param in tqdm(enumerate(grid)): 
        logger.info(f"Evaluating parameters {param}")
        ktp = KTopicModel(k = param["k"], a = param["a"], path_embeddings = args.path_embeddings)
        data = pd.read_csv(args.path_data, dtype = str, index_col = 0)
        
        #drop duplicates and nas
        data.drop_duplicates(subset = [args.col_text], inplace = True)
        data.dropna(inplace = True)
        
        #get splits
        splitter = StratifiedShuffleSplit(n_splits = 1, train_size = args.train_ize, random_state = 1234)
        tr, ts = next(splitter.split(data, data['day'] + data['source']))
        
        
        #keep important TRAINING cols:
        data = data[args.col_text].iloc[tr]

        #fit the model
        ktp.fit(data.iloc[tr])
        #evaluate:
        grid[i]['silhuette'] = ktp.evaluate(data, sample_size = 10000, n_samples = 30)
        grid[i]['inhertia'] = ktp.cluster.inertia_
        #save:
        pickle.dump(ktp, open(os.path.join(args.path_results, f'cluster_k_{ktp.k}_a_{ktp.a}.p'), 'wb'))
        json.dump(grid, open(os.path.join(args.path_results, 'results.json'), 'w'))