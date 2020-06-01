#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 19:57:21 2020

@author: piotr
"""

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import itertools
import scipy as sp
import collections
import random

class ModelPolarization:
    def __init__(self, parties, speakers, text):
        text = self.vectorize_data(text, ngram_range = (2,2), limit = 10) #vectorize
        self.party = np.unique(parties) #store party names
        self.counts = self.ngram_counts(parties, speakers, text)

        


    def vectorize_data(self, data, ngram_range, limit):
        
        """
        
        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        ngram_range : TYPE
            DESCRIPTION.
        limit : TYPE
            DESCRIPTION.

        Returns
        -------
        res : TYPE
            DESCRIPTION.

        """
        
        vectorizer = CountVectorizer(ngram_range = ngram_range)
        vectorizer.fit(data)
        res = vectorizer.transform(data)
        indices = np.where(np.array(res.sum(axis = 0)).flatten() > limit)[0]
        res = res[:,indices]

        return res
    
    
    def ngram_counts(self, parties, speakers, text_vectorized):
        
        """
        Get speaker - count vector pairs for each speaker.

        Parameters
        ----------
        parties : LIST
            party names
        speakers : LIST
            speakers ids.
        text_vectorized : LIST
            vectorized text

        Returns
        -------
        counts : dict

        """
        
        #get counts of each bigram for each speaker:
        zipper = sorted(zip(parties, speakers, text_vectorized), key = lambda x:(x[0], x[1]))
        counts = dict()
        for g, elem in itertools.groupby(zipper,lambda x: (x[0],x[1])):
            counts[g] = sp.sparse.vstack(list(i for _, _, i in elem))
        counts = {k:np.array(v.sum(axis = 0)).flatten() for k, v in counts.items() if v.sum() > 0}
        
        return counts
        
    
    
    def sample_speakers(self, proportion = 0.1):
        """
        Subsampling procedure for estimating confidence intervals.

        Parameters
        ----------
        proportion : TYPE, optional
            DESCRIPTION. The default is 0.1.

        Returns
        -------
        sample : TYPE
            DESCRIPTION.

        """
        popsize = int(len(self.counts.keys()) / 2) #half of the population - so sample is equal
        sampsize = int(round(popsize * proportion))
        
        #get random draw of 0.5 sample size from first party
        inds = random.sample(range(popsize), sampsize)
        sample = dict([elem for i, elem in enumerate(filter(lambda x: x[0][0] == self.party[0], self.counts.items())) if i in inds])
        
        #get random draw of 0.5 sample size from second party
        inds = random.sample(range(popsize), sampsize)
        sample.update(dict([elem for i, elem in enumerate(filter(lambda x: x[0][0] == self.party[1], self.counts.items())) if i in inds]))
        
        #return the sample
        return sample
    
   
    
    def choice(self, counts):
        """
        Get empirical probabilities of phrase frequencies for each speaker.

        Parameters
        ----------
        counts : TYPE
            DESCRIPTION.

        Returns
        -------
        c_mat - sparse matrix of phrase counts [NxM]
        parties - array of party names [Nx1]
            Dimensions:
            N - the number of speakers
            M - the number of bigrams

        """
        
        
        c_mat = sp.sparse.csr_matrix(np.vstack(list(counts.values())))
        parties = np.array(list(party for party, id in counts.keys()))
        speaker_freq = c_mat/c_mat.sum(axis = 1)
        
        #get speaker phrase frequencies
        speaker_freq = collections.defaultdict(lambda: [])
        for g, elem in itertools.groupby(zip(parties, counts.values()), lambda x: x[0]):
            speaker_freq[g].extend(list(i/i.sum() for _, i in elem))
        speaker_freq = {k:np.vstack(v) for k, v in speaker_freq.items()}
        
        
        return c_mat
    
    
    def posterior(self, c_mat, counts, leave_out):
        """
        Get posterior beliefs of observer, i.e. probability of assignment to 
        a certain party.

        Parameters
        ----------
        leave_out : bool, estimate by leave-out?
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        posterior = collections.defaultdict(lambda: []) #store data
        parties = np.array(list(party for party, id in counts.keys())) #party vector
        p0ind = np.array(parties) == self.party[0] #index for one party
        p1ind = np.array(parties) == self.party[1] #index for another
        
        if leave_out:
            
            #get leave-out estimates of posterior, i.e. removing the speaker
            #for whom the posterior is computed
            posterior = []
            ind = np.arange(c_mat.shape[0])
            for i in range(c_mat.shape[0]):
                qg = np.sum(c_mat[(ind != i) & p0ind], axis = 0)
                qo = np.sum(c_mat[(ind != i) & p1ind], axis = 0)
                posterior.extend(np.array(qg/(qo+qg)))
            posterior = np.vstack(posterior)
            posterior = np.nan_to_num(posterior, copy = False, nan = 0.0)

            

            
        else:
            
            posterior = np.array(c_mat[p0ind].sum(axis = 0)/c_mat.sum(axis = 0))
            posterior = posterior.repeat(c_mat.shape[0], axis = 0)
                
            
        return posterior
    
    
    def polarization(self, counts, leave_out, absolute = True):
        """
        Get the polarization matrix.

        Parameters
        ----------
        counts : TYPE
            DESCRIPTION.
        leave_out : TYPE
            DESCRIPTION.

        Returns
        -------
        p_mat : TYPE
            DESCRIPTION.

        """
        c_mat = self.choice(counts) #get counts matrix and parties vector
        parties = np.array(list(party for party, id in counts.keys())) #party vector
        p0ind = parties == self.party[0] #index of party 0
        p1ind = np.logical_not(p0ind) #index of party 1
        posterior = self.posterior(c_mat, counts, leave_out = leave_out) #get posterior
        c_mat = np.array(c_mat/c_mat.sum(axis = 1)) #get frequencies
        p_scores = np.zeros(c_mat.shape[0])
        
        
        #transform the polarization matrix into polarization scores
        if absolute: #expressed in terms of probability of being assigned to one's party
            p_scores[p0ind] = np.sum(c_mat[p0ind] * posterior[p0ind], axis = 1)
            p_scores[p1ind] = np.sum(c_mat[p1ind] * (1 - posterior[p1ind]), axis = 1)
        else: #expressed in terms of probability of being assigned to government
            p_scores = np.sum(c_mat * posterior, axis = 1)
            
        
        return p_scores
        
    
    def estimate(self, counts = None, conf_int = None, leave_out = False):
        """
        Get aggregated polarization estimates for a given counts matrix.

        Parameters
        ----------
        counts : TYPE, optional
            DESCRIPTION. The default is None.
        conf_int : TYPE, optional
            DESCRIPTION. The default is None.
        leave_out : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        res : TYPE
            DESCRIPTION.

        """
        if not counts:
            counts = self.counts
        
        #store results
        res = dict()
        
        #get polarization matrix and party name vector
        p_scores = self.polarization(counts, leave_out)
        parties = np.array(list(party for party, id in counts.keys())) #party vector
        
        #get sample sizes for both parties
        res["n"] = {self.party[0]: np.sum(parties == self.party[0]), 
                    self.party[1]: np.sum(parties == self.party[0])}
        

        #get point estimate of polarization
        gov = 0.5*(1/res["n"][self.party[0]]) * np.sum(p_scores[parties == self.party[0]])
        opp = 0.5*(1/res["n"][self.party[1]]) * np.sum(p_scores[parties == self.party[1]])
        res["point"] = gov + opp
        
        
        #get subsampling confidence intervals
        if conf_int:
            res["samples"] = []

            for i in range(conf_int):
                sample = self.sample_speakers(0.1)
                res["samples"].append(self.estimate(counts = sample, conf_int = None,
                                                    leave_out = leave_out))

            return res

        return res
