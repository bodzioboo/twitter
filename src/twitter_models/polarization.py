#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:49:52 2020

@author: piotr
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 19:57:21 2020

@author: piotr
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict
from scipy.sparse import csr_matrix
import pdb
import logging
import pandas as pd


class ModelPolarization:
    def __init__(self, parties, limit = 10, ngram_range = (2,2), 
                 method = "count", log = logging.INFO, 
                 **kwargs):
        """
        Initialize the model

        Parameters
        ----------
        parties : list
            names of the two parties to be used in the model.
        limit : int, optional
            Minimum number of time a phrase has to be used to be included. The default is 10.
        ngram_range : tuple, optional
            Ngram range for the vectorizer. The default is (2,2).
        method : str, optional
            Vectorization method. Possible options ("count", "tfidf"). The default is "count".

        Returns
        -------
        None.

        """
        self.party = np.unique(parties) #store party names
        if method == "count":
            self.vectorizer = CountVectorizer(ngram_range = ngram_range, min_df = limit, **kwargs)
        elif method == "tfidf":
            self.vectorizer = TfidfVectorizer(ngram_range = ngram_range, min_df = limit, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log)

        self.logger.debug('Initialized')
            
            
    def estimate(self, 
                 parties:list, 
                 speakers:list, 
                 text:list, 
                 text_vectorized = None, 
                 text_id:list = None,
                 level:str = "aggregate", 
                 conf_int:int = None, 
                 sample_size:float = 0.1, **kwargs):
        """
        Main function. Estimate partisanship according to Gentzkow's model.

        Parameters
        ----------
        parties : TYPE
            Party IDs.
        speakers : TYPE
            Speakers IDs.
        text : TYPE
            Texts.
        text_id : TYPE, optional
            Unique ID of each text. The default is None. Required when
            level = 'speech'
        level : TYPE, optional
            Aggregation level. The default is "aggregate".
        conf_int : int, optional
            Number of confidence intervals. The default is None.
        sample_size : float, optional
            Sample size for confidence intervals. The default is 0.1.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        #convert to numpy:
        parties = np.array(parties, dtype = str)
        speakers = np.array(speakers, dtype = str)
        text = np.array(text, dtype = str)
        #vectorize text:
        #first run - fit; next runs (CI) - just transform
        if text_vectorized is None:
            text_vectorized = self._vectorize_text(text)
        
        if level == "aggregate": #aggregate estimates
            parties_a, speakers_a, text_vectorized_a = self._aggregate(parties, speakers, text_vectorized)
            p_scores = self._polarization(parties_a, speakers_a, text_vectorized_a, low_memory = False, **kwargs)
            #store results
            res = dict()
        
            #get sample sizes for both parties
            res["n"] = {self.party[0]: np.sum(parties_a == self.party[0]), 
                    self.party[1]: np.sum(parties_a == self.party[1])}


            #get point estimate of polarization
            gov = 0.5*(1/res["n"][self.party[0]]) * np.sum(p_scores[parties_a == self.party[0]])
            opp = 0.5*(1/res["n"][self.party[1]]) * np.sum(p_scores[parties_a == self.party[1]])
            res["point"] = gov + opp
            
            if conf_int:
                assert(sample_size is not None) 
                samples = []
                
                for i in range(conf_int):
                    samp_inds = self._sample(parties, sample_size = 0.1, stratified = True) #get sample of ids
                    speakers_s = speakers[samp_inds]
                    parties_s = parties[samp_inds]
                    text_s = text[samp_inds]
                    text_vectorized_s = text_vectorized[samp_inds]
                    #apply recursively:
                    samples.append(self.estimate(parties_s, speakers_s, text_s, 
                                                 text_vectorized = text_vectorized_s,
                                                 conf_int = None, level = "aggregate")) 
                res = self._confidence_intervals(samples, res) #compute confidence intervals
                return res
            
            return res
                    
        elif level == "speaker": #polarization for each individual user
            parties_a, speakers_a, text_vectorized_a = self._aggregate(parties, speakers, text_vectorized)
            p_scores = self._polarization(parties_a, speakers_a, text_vectorized_a, low_memory = True, **kwargs)
            
            return p_scores, speakers_a
            
            
        elif level == "speech" and text_id is not None: #polarization for each individual speech
            text_id = np.array(text_id)
            p_scores = self._polarization(parties, speakers, text_vectorized, low_memory = True, **kwargs)
            return dict(zip(text_id, p_scores))
        
            
            
    def get_posteriors(self, parties, speakers, text):
        """
        Get posterior distribution of each word to evaluate word-wise partisanship.

        """
        parties = np.array(parties, dtype = str)
        speakers = np.array(speakers, dtype = str)
        text = np.array(text, dtype = str)
        text_vectorized = self._vectorize_text(text)
        parties_a, speakers_a, text_vectorized_a = self._aggregate(parties, speakers, text_vectorized)
        posterior = self._posterior(parties_a, speakers_a, text_vectorized_a, low_memory = False)
        posterior = np.mean(posterior, axis = 0)
        posterior = dict(zip(self.vectorizer.get_feature_names(), posterior))
        
        return posterior
            
    
    def prefit(self, text):
        """
        Prefit vectorizer.
        
        """
        self.vectorizer.fit(text)
        return self
    
    def _vectorize_text(self, text):
        """
        Parameters
        ----------
        text : list of strings
        Returns
        -------
        text_vectorized: sparse matrix

        """
        self.logger.debug('Vectorizing text')
        if hasattr(self.vectorizer, 'vocabulary_'):
            text_vectorized = self.vectorizer.transform(text)
        else:
            text_vectorized = self.vectorizer.fit_transform(text)
        self.logger.debug('Done vectorizing text')
        return text_vectorized
        
    
    
    
    
    def _aggregate(self, parties, speakers, text_vectorized):
        """
        
        Aggregates the vectorized text data by speaker
        
        Parameters
        ----------
        parties : list-like object
            vector of party names.
        speakers : list-like object
            vector of speaker ids.
        text_vectorized : np.array
            vectorized text.

        Returns
        -------
        parties_u : np.array
            party names corresponding to unique speaker ids.
        speakers_u : np.array
            unique speaker ids.
        agg : np.array
            vectorized text aggregated by speaker.

        """
        
        #get unique speakers and corresponding parties:
        parties_u, speakers_u = zip(*np.unique(np.c_[parties, speakers], axis = 0))
        parties_u = np.array(parties_u); speakers_u = np.array(speakers_u)
        agg = []
        #aggregate by speaker
        for speaker in speakers_u:
            agg.append(text_vectorized[speakers == speaker].sum(axis = 0))
        agg = np.vstack(agg)
        
        #remove 0s
        nonzero = np.array(agg.sum(axis = 1) > 0).flatten()
        parties_u = parties_u[nonzero]
        speakers_u = speakers_u[nonzero]
        agg = agg[nonzero]
        agg = csr_matrix(agg)

        return parties_u, speakers_u, agg
    
    
    
    
    
    def _posterior(self, parties, speakers, text_vectorized, leave_out = True, low_memory = True):
        """
        

        Parameters
        ----------
        parties : list-like object
            vector of party names.
        speakers : list-like object
            vector of speaker ids.
        text_vectorized : np.array
            vectorized text.
        leave_out : bool, optional
            Whether to calculate the leave-out estimate. The default is True.
            If False - compute plug in estimate of posterior, i.e. same for all speakers
        low_memory: bool, optional
            control whether the computation is memory-efficient (through a for loop on a dictionary, 
            using sparse arrays or processing-time efficient (using vectorization on np.array)
            default is True

        Returns
        -------
        posteriors : np.array
            posterior probabilities of being assigned to first party for each phrase in text.

        """
        
        if leave_out:
        
            if low_memory:
                #this methodis more memory efficient, but takes longer processing time:
            
                #leave out estimate (i.e. for each speaker posterior calculated EXCLUDING his vocabulary)
                posteriors = dict()
                for speaker in np.unique(speakers): #compute for each speaker
                    ind = (speakers != speaker) #indices of all speakers but the current one
                    denominator = text_vectorized[ind].sum(axis = 0) #phrase counts for all speakers but the current one
                    numerator = text_vectorized[(parties == self.party[0]) & ind].sum(axis = 0) #same as above but only party 0
                    posteriors[speaker] = numerator/denominator #calculate posterior
                    posteriors[speaker] = np.nan_to_num(posteriors[speaker], nan = 0.5, copy = False)
                #return a dict
            else:
                
                #this method is vectorized, but takes up more memory
                self.logger.debug('Computing posteriors')
                text_vectorized = text_vectorized.toarray() #convert sparse to normal
                denominators = text_vectorized.sum(axis = 0) - text_vectorized #sum of all phrases for each excluding each
                p0 = text_vectorized.copy()
                p0[parties == self.party[1]] = 0 #subtract individual counts only when party is p0
                numerators = text_vectorized[parties == self.party[0]].sum(axis = 0) - p0
                posteriors = numerators/denominators
                posteriors = np.nan_to_num(posteriors, nan = 0.5, copy = False)
                self.logger.debug('Done computing posteriors')
                #return an array
                
        else:
            
            #plug in estimate
            denominator = text_vectorized.sum(axis = 0) #phrase counts for all speakers
            numerator = text_vectorized[parties == self.party[0]].sum(axis = 0) #phrase counts for party 0
            posteriors = defaultdict(lambda: numerator/denominator) #calculate posterior
                
            
        
        
        return posteriors
    
    
    
    
    
    
    def _polarization(self, parties, speakers, text_vectorized, 
                      low_memory = True, **kwargs):
        """
        

        Parameters
        ----------
        parties : list-like object
            vector of party names.
        speakers : list-like object
            vector of speaker ids.
        text_vectorized : np.array
            vectorized text.
        low_memory: bool, optional
            control whether the computation is memory-efficient (through a for loop on a dictionary, 
            or processing-time efficient (using vectorization).
            default is True


        Returns
        -------
        p_scores : TYPE
            DESCRIPTION.

        """
        c_mat = text_vectorized.multiply(1/text_vectorized.sum(axis = 1)) #normalize counts into individual probabilities
        c_mat = np.nan_to_num(c_mat, copy = False, nan = 0.5) #convert missing to 0.5s
        
        
        if low_memory:
            #memory-efficient version - requires id:count dictionary as input, return by the _posterior method
            #if low_memory set to True
            c_mat = c_mat.tocsr() #convert to csr
            posterior = self._posterior(parties, speakers, text_vectorized, low_memory = low_memory, **kwargs) #get posterior
            p_scores = np.zeros(c_mat.shape[0]) #store polarization scores
            
            for speaker in np.unique(speakers):
                ind = speakers == speaker
                post = np.array(posterior[speaker]).flatten()
                #depending on the party, posterior is either posterior or 1 -posterior
                if np.unique(parties[ind])[0] != self.party[0]:
                    post = 1 - post
                counts = np.array(c_mat[ind].todense())
                p_scores[ind] = np.sum(counts * post, axis = 1) #rowwise dot product
        else:
            #vectorized version - requires array as input:
            #depending on the party, posterior is either posterior or 1 -posterior
            c_mat = c_mat.toarray() #convert to csr
            posterior = self._posterior(parties, speakers, text_vectorized, low_memory = low_memory, **kwargs) #get posterior
            p_scores = np.zeros(c_mat.shape[0]) #store polarization scores
            posterior[parties != self.party[0]] = 1 - posterior[parties != self.party[0]] #SUBTRACTION OF ONE!
            p_scores = np.sum(c_mat * posterior, axis = 1) #rowwise dot product
            
            
                
        p_scores[np.array(c_mat.sum(axis = 1) == 0).flatten()] = 0.5 #if no phrase counted = 0.5 similarity (neutral)  
         
        return p_scores
    

        
    
    
    

    def _sample(self, parties, sample_size = 0.1, stratified = False):

        if stratified:
            prob_p0 = (np.sum(parties == self.party[0])/parties.shape[0])/np.sum(parties == self.party[0])
            prob_p1 = (np.sum(parties == self.party[1])/parties.shape[0])/np.sum(parties == self.party[1])
            p = np.where(parties == self.party[0], prob_p0, prob_p1)
        else:
            p = None
            
        inds = np.arange(0, len(parties))
        sample_size = round(len(inds)*sample_size) #get number
        sample_inds = np.random.choice(inds, size = sample_size, p = p, replace = False)
        
        #return the sample
        return sample_inds

    
    def _confidence_intervals(self, samples, res):
        
        #sample size and point estimate
        estimate = res["point"]
        size = res["n"][self.party[0]] + res["n"][self.party[1]]
        
        #sizes and estimates for each of the subsamples:
        sample_sizes = np.array([elem['n']['gov'] + elem['n']['opp'] for elem in samples])
        sample_estimates = np.array([elem['point'] for elem in samples])
    
        #Qkt - for each subsample deviation of subsample from the mean, normalized by sqrt of the subsample's SD
        Q_kt = np.sqrt(sample_sizes) * (sample_estimates - np.mean(sample_estimates))
        
        #store
        stats = dict()
    
        #lower ci
        stats['lower_ci'] = estimate - np.partition(Q_kt, 90)[90]/np.sqrt(size)
    
        #upper ci
        stats['upper_ci'] = estimate - np.partition(Q_kt, 10)[10]/np.sqrt(size) 
    
        #estimates
        stats['estimate'] = estimate
        
        return stats
    
    
    def estimate_topics(self, source:list, topics:list):
        dat = pd.DataFrame(dict(source = source, topic = topics))
        dat = pd.DataFrame(dat.groupby(['source', 'topic']).size()).reset_index().pivot(index = 'topic', columns = 'source')
        dat.columns = dat.columns.droplevel(0)
        dat['n'] = dat['gov'] + dat['opp'] #get number of tweets in topic on this day
        dat['prob_gov'] = dat['gov']/(dat['n']) #get probability of topic in gov
        dat['prob_opp'] = 1 - dat['prob_gov'] #get probability of topic in opp
        #get posterior probability of assigning to correct party for each topic
        dat['pola'] = dat['prob_gov']*dat['gov']/dat['n'] + dat['prob_opp']*dat['opp']/dat['n'] 
        #get weigthed average
        pola = (dat['pola']*dat['n']/dat['n'].sum()).sum()
        return pola
    
    
    

        
#test 
if __name__ == "__main__":
    """
    
    import sys
    sys.path.append("..")
    from twitter_tools.utils import read_files
    import json 
    import os
    import time
    path = "/home/piotr/projects/twitter/data/clean"
    nonpolish_ids = json.load(open(os.path.join(path, "non_polish_ids.json"), "r"))
    from functools import partial
    path_stopwords = "/home/piotr/nlp/polish.stopwords.txt"
    stopwords = []
    with open(path_stopwords, "r") as f:
        for word in f:
            if word != "nie":
                stopwords.append(word.strip("\n"))
    
    
    def data_filter(data, n):
        data.loc[:,("lemmatized")] = data.loc[:,("lemmatized")].apply(lambda x: " ".join([word for word in x.split() if word not in stopwords]))
        data[data["user-id_str"].apply(lambda x: x not in nonpolish_ids)] #exclude non-polish users
        #data = data.loc[data.polish.astype(float) >= 0.5] #more than 50% polish words
        data = data[data.lemmatized.apply(lambda x: len(x) >= n)] #longer than n
        data = data[np.logical_not(data.retweet)]
        return data
    
    for dat in tqdm(read_files(path, 2, dtype = {"user-id_str":str, "retweet": bool, 
                                                 "polish": float, "lemmatized":str}, 
                               filter_fun = partial(data_filter, n = 0))):
        pass
    mod = ModelPolarization(parties = ["gov","opp"], limit = 10)
    start = time.time()
    res = mod.estimate(dat["source"], dat["user-id_str"], dat["lemmatized"], level = "aggregate", leave_out = True, conf_int = 100)
    print(f'Evaluation time{time.time() - start}')
    print(res)
    """