#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 11:03:15 2020

@author: piotr
"""

import os
import pickle
import json
import sys
sys.path.append("..")
from twitter_models.embeddings import KTopicModel
from twitter_tools.utils import batch
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("path_data", type = str, help = "Path to file")
    parser.add_argument("path_results", type = str, help = "Path to store results in")
    parser.add_argument("path_embeddings", type = str, help = "Path to word embeddings", default = None)
    parser.add_argument("col_text", help = "Name of the text column")
    parser.add_argument("k", help = "Number of clusters", type = int)
    parser.add_argument("a", help = "Inverse weighting of words", type = float, default = 0.0001)
    parser.add_argument("train_size", help = "Size of the train for the clusterer", type = float)
    parser.add_argument("--pickle_model", action = "store_true", dest = "pickle", help = "Pickle the trained model?")
    parser.add_argument("--pretrained", type = str, default = None, help = "Path to pretrained model")
    parser.set_defaults(pickle = False)
    args = parser.parse_args()
    
    #load data:
    data = pd.read_csv(args.path_data, dtype = str, index_col = 0)
    data.drop_duplicates(subset = [args.col_text], inplace = True)
    data.dropna(inplace = True)
    
    
    #load pretrained or train
    if args.pretrained:
        ktp = pickle.load(open(args.pretrained))
        data = data[args.col_text]
    else:
        ktp = KTopicModel(k = args.k, path_embeddings = args.path_embeddings)
        splitter = StratifiedShuffleSplit(n_splits = 1, train_size = args.train_size, random_state = 1234)
        tr, ts = next(splitter.split(data, data['day'] + data['source']))
        data = data[args.col_text]
        ktp.fit(data.iloc[tr])
        
    
        
    clusters = []
    for dat in batch(data, 500000):
        clusters.extend(ktp.predict(dat, refit = True))
    with open(os.path.join(args.path_results, f'cluster_k_{ktp.k}_a_{ktp.a}.json'), 'w') as f:
        json.dump(dict(zip(data, clusters)), f)
    if args.pickle:
        pickle.dump(ktp, open(os.path.join(args.path_results, f'cluster_k_{ktp.k}_a_{ktp.a}.p'), 'wb'))