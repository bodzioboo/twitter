#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 18:54:47 2020

@author: piotr
"""


import warnings
warnings.filterwarnings('ignore')
import os
import sys
import pickle
import itertools
from functools import partial
import gc
import numpy as np
sys.path.append("/home/piotr/projects/twitter/src")
from twitter_tools.utils import read_files, clean_tweets
from twitter_models.embeddings import KTopicModel
from tqdm import tqdm

grid = {"min_length":[10, 5, 0], "a":[1e-2, 1e-3, 1e-5], "k":[5, 10, 20]}
keys = grid.keys(); combs = itertools.product(*grid.values()) #keys + all combinations of parameters
params = [dict(zip(keys, elem)) for elem in combs] #zip these toegether into a list of dicts
path_results = "/home/piotr/projects/twitter/notebooks/nlp/results/topics"
path_clean = "/home/piotr/projects/twitter/data/clean" #data path for reader
path_embeddings = "/home/piotr/nlp/cc.pl.300.vec"

#define filter function for data reader
def keep_n(data, n):
    data["full_text"] = data["full_text"].apply(lambda x: clean_tweets(x).lower()) #clean and lowercase
    data = data.loc[(data.full_text.apply(lambda x: len(x.split()) >= n)) & (np.logical_not(data.english))] #drop
    return data


min_length = 10  #None #store values from previous iteration. If they are the sampe, skip steps to save computation time
np.random.seed(1234)
for i, param in enumerate(params): #Iteratre over parameter space
    if param["min_length"] != min_length: #if previous minmum length wasn't the same, reload the data:
        text = list()
        for dat in tqdm(read_files(path_clean, 85, filter_fun = partial(keep_n, n = param["min_length"]))):
            text.extend(dat["full_text"].tolist()) #text to list
        del(dat)
        gc.collect()
        text = list(set(text)) # keep unique
        text = np.random.choice(text, int(0.5*len(text))) #sampl
        
    print(f"Total data length {len(text)}")
    ktp = KTopicModel(k = param["k"], batch_size = 1000, path_embeddings = path_embeddings, a = param["a"], dask_split = 100)
    ktp.fit(text)
      
    params[i]["fit"] = {"silhuette":ktp.evaluate(text, sample_size = 10000, n_samples = 30),
                        "intertia":ktp.cluster.inertia_} #store evaluation metrics
    ktp.embed = dict(ktp.embed) #cannot be serialized with a default dict
    pickle.dump(ktp, open(os.path.join(path_results, f'model_k_{param["k"]}_n_{param["min_length"]}_a_{param["a"]}.p'), 'wb'))
    min_length = param["min_length"]
    pickle.dump(params, open(os.path.join(path_results, "results.p"), "wb"))