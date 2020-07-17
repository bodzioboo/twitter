#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:43:14 2020

@author: piotr
"""

import sys
sys.path.append("..")
from twitter_models.embeddings import OutlierDetector
from twitter_tools.utils import batch
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedShuffleSplit
import argparse
import pandas as pd
import pickle
import json
import pdb

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("path_data", type = str, help = "Path to file")
    parser.add_argument("path_results", type = str, help = "Path to results file")
    parser.add_argument("path_embeddings", type = str, help = "Path to word embeddings", default = None)
    parser.add_argument("col_text", help = "Name of the text column")
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
        detector = pickle.load(open(args.pretrained))
        data = data[args.col_text].astype(str)
    else:
        detector = OutlierDetector(args.path_embeddings, IsolationForest(n_jobs = 4))
        splitter = StratifiedShuffleSplit(n_splits = 1, train_size = args.train_size, random_state = 1234)
        tr, ts = next(splitter.split(data, data['day'] + data['source']))
        data = data[args.col_text].astype(str)
        detector.fit(data.iloc[tr])
        
    outliers = []
    for dat in batch(data, 500000):
        out = detector.predict(dat)
        outliers.extend(out.tolist())
    with open(args.path_results, 'w') as f:
        json.dump(dict(zip(data, outliers)), f)
    