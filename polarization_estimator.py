#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 19:57:21 2020

@author: piotr
"""

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import operator
import itertools
import scipy as sp
import collections
from tqdm import trange
import random

class ModelPolarization:
    def __init__(self, parties, speakers, text):
        text = self.vectorize_data(text, ngram_range = (2,2), limit = 10) #vectorize
        self.party = np.unique(parties) #store party names

        #get counts of each bigram for each speaker:
        zipper = sorted(zip(parties, speakers, text), key = lambda x:(x[0], x[1]))
        self.counts = dict()
        for g, elem in itertools.groupby(zipper,lambda x: (x[0],x[1])):
            self.counts[g] = sp.sparse.vstack(list(i for _, _, i in elem))
        self.counts = {k:np.array(v.sum(axis = 0)).flatten() for k, v in self.counts.items() if v.sum() > 0}


    def vectorize_data(self, data, ngram_range, limit):
        vectorizer = CountVectorizer(ngram_range = ngram_range)
        vectorizer.fit(data)
        res = vectorizer.transform(data)
        indices = np.where(np.array(res.sum(axis = 0)).flatten() > limit)[0]
        #names = vectorizer.get_feature_names()
        res = res[:,indices]
        #names = operator.itemgetter(*indices)(names)

        return res

    def plug_in(self, counts = None, confidence_interval = None):
        if not counts:
            counts = self.counts
        #Party-wise total phrase counts
        party_counts = collections.defaultdict(lambda: [])
        for g, elem in itertools.groupby(zip(counts.keys(), counts.values()), lambda x: x[0][0]):
            party_counts[g].extend(list(i for _,i in elem))

        #empirical party frequencies
        party_freq = {k:np.vstack(v).sum(axis = 0) for k, v in party_counts.items()}
        party_freq = {k:v/v.sum() for k, v in party_freq.items()}

        #posterior belief that an observer with neutral prior assigns the true party:
        posterior = party_freq[self.party[0]]/(party_freq[self.party[0]] + party_freq[self.party[1]])


        results = dict()


        #partisanship plug-in estimator
        results["point"] = np.sum(0.5*party_freq[self.party[0]] * posterior + 0.5*party_freq[self.party[1]]*(1 - posterior))

        if confidence_interval:
            results["samples"] = []

            for i in range(confidence_interval):
                popsize = len(self.counts.keys())
                sampsize = int(round(popsize/10))
                inds = random.sample(range(popsize),sampsize)
                sample_keys = [val for i, val in enumerate(self.counts.keys()) if i in inds]
                sample = dict(zip(sample_keys, operator.itemgetter(*sample_keys)(self.counts)))
                results["samples"].append(self.plug_in(counts = sample, confidence_interval = None))

            return results

        return results["point"]


    def leave_out(self, counts = None, confidence_interval = None):

        if not counts:
            counts = self.counts

        c_mat = sp.sparse.csr_matrix(np.vstack(list(counts.values())))
        parties = np.array(list(party for party, id in counts.keys()))
        indg = np.array(parties) == self.party[0]
        indo = np.array(parties) == self.party[1]

        #get posterior belief of neutral observer for each phrase
        posterior = collections.defaultdict(lambda: [])
        zipper = zip(counts.keys(), c_mat)
        for i, (g, val) in enumerate(itertools.groupby(zipper, lambda x:x[0])):
            ind = np.arange(c_mat.shape[0]) != i
            qg = np.sum(c_mat[ind & indg], axis = 0)
            qo = np.sum(c_mat[ind & indo], axis = 0)
            posterior[g[0]].extend(np.array(qg/(qo+qg)))
        posterior = {k:np.nan_to_num(np.vstack(v),copy = False,nan = 0.0) for k, v in posterior.items()}


        #get speaker phrase frequencies
        speaker_freq = collections.defaultdict(lambda: [])
        for g, elem in itertools.groupby(zip(parties, counts.values()), lambda x: x[0]):
            speaker_freq[g].extend(list(i/i.sum() for _, i in elem))
        speaker_freq = {k:np.vstack(v) for k, v in speaker_freq.items()}

        #results
        results = dict()



        gov = 0.5*(1/len(speaker_freq[self.party[0]]))*np.sum(speaker_freq[self.party[0]] * posterior[self.party[0]])
        opp = 0.5*(1/len(speaker_freq[self.party[1]]))*np.sum(speaker_freq[self.party[1]] * (1 - posterior[self.party[1]]))
        results["point"] = gov + opp

        if confidence_interval:
            results["samples"] = []

            for i in range(confidence_interval):
                popsize = len(self.counts.keys())
                sampsize = int(round(popsize/10))
                #pdb.set_trace()
                inds = random.sample(range(popsize),sampsize)
                sample_keys = [val for i, val in enumerate(self.counts.keys()) if i in inds]
                sample = dict(zip(sample_keys, operator.itemgetter(*sample_keys)(self.counts)))
                results["samples"].append(self.leave_out(counts = sample, confidence_interval = None))

            results["lower"] = np.quantile(results["samples"], 0.025)
            results["upper"] = np.quantile(results["samples"], 0.975)

            return results




        return results["point"]
