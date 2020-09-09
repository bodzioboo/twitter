#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 15:49:52 2020

@author: piotr
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 19:57:21 2020

@author: piotr
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
from operator import itemgetter
from joblib import Parallel, delayed
import logging
import pandas as pd
from typing import List, Sequence, Iterable
import dask.array as da
from dask.diagnostics import ProgressBar


class ModelPolarization:
    def __init__(self, parties, limit=10, ngram_range=(2, 2),
                 method="count", log=logging.INFO, n_jobs=4,
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
        :param num_workers:

        """
        self.party = np.unique(parties)  # store party names
        if method == "count":
            self.vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=limit, **kwargs)
        elif method == "tfidf":
            self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=limit, **kwargs)
        self.n_jobs = n_jobs
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log)
        self.logger.debug('Initialized')

    def estimate(self,
                 parties: List[str],
                 speakers: List[str],
                 text: List[str],
                 text_vectorized=None,
                 text_id: list = None,
                 level: str = "aggregate",
                 conf_int: int = None,
                 sample_size: float = 0.1, **kwargs):
        """
        
        :param list of party names: 
        :param list of speaker IDs: 
        :param list of texts: 
        :param sparse array of vectorized text. Used for CI estimation.  
        :param text_id: list of text IDs. Used when level = 'speech'
        :param level: what level of aggregation to calculate polarization on 
        :param conf_int: how many confidence intervals to calculate
        :param sample_size: sample size for confidenence interval
        :param kwargs: 
        :return: 
        """

        # convert to numpy:
        parties = np.array(parties, dtype=str)
        speakers = np.array(speakers, dtype=str)
        text = np.array(text, dtype=str)
        # vectorize text:
        # first run - fit; next runs (CI) - just transform
        if text_vectorized is None:
            text_vectorized = self._vectorize_text(text)

        if level == "aggregate":  # aggregate estimates
            parties_a, speakers_a, text_vectorized_a = self._aggregate(parties, speakers, text_vectorized)
            p_scores = self._polarization(parties_a, speakers_a, text_vectorized_a, low_memory=False, **kwargs)
            # store results
            res = dict()

            # get sample sizes for both parties
            res["n"] = {self.party[0]: np.sum(parties_a == self.party[0]),
                        self.party[1]: np.sum(parties_a == self.party[1])}

            # get point estimate of polarization
            gov = 0.5 * (1 / res["n"][self.party[0]]) * np.sum(p_scores[parties_a == self.party[0]])
            opp = 0.5 * (1 / res["n"][self.party[1]]) * np.sum(p_scores[parties_a == self.party[1]])
            res["point"] = gov + opp

            if conf_int:
                assert (sample_size is not None)
                samples = Parallel(n_jobs=self.n_jobs)(delayed(self._bootstrap)(speakers, parties,
                                                                                text, text_vectorized,
                                                                                sample_size=sample_size,
                                                                                stratified=True) for _ in
                                                       range(conf_int))
                res = self._confidence_intervals(samples, res)  # compute confidence intervals
                return res

            return res

        elif level == "speaker":  # polarization for each individual user
            parties_a, speakers_a, text_vectorized_a = self._aggregate(parties, speakers, text_vectorized)
            p_scores = self._polarization(parties_a, speakers_a, text_vectorized_a, low_memory=True, **kwargs)

            return p_scores, speakers_a


        elif level == "speech" and text_id is not None:  # polarization for each individual speech
            text_id = np.array(text_id)
            p_scores = self._polarization(parties, speakers, text_vectorized, low_memory=True, **kwargs)
            return dict(zip(text_id, p_scores))

    def estimate_phrase_frequency(self, parties: List[str], speakers: List[str], text: List[str]):
        parties = np.array(parties, dtype=str)
        speakers = np.array(speakers, dtype=str)
        text = np.array(text, dtype=str)
        text_vectorized = self._vectorize_text(text)
        parties_a, speakers_a, text_vectorized_a = self._aggregate(parties, speakers, text_vectorized)
        probs = self._phrase_prob(parties_a, speakers_a, text_vectorized_a, low_memory=False)
        probs = np.mean(probs, axis=0)
        probs = dict(zip(self.vectorizer.get_feature_names(), probs))

        return probs

    def estimate_phrases(self, parties: List[str], speakers: List[str], text: List[str]):

        parties = np.array(parties, dtype=str)
        speakers = np.array(speakers, dtype=str)
        text = np.array(text, dtype=str)
        text_vectorized = self._vectorize_text(text)
        parties_a, speakers_a, text_vectorized_a = self._aggregate(parties, speakers, text_vectorized)

        with np.errstate(divide='ignore', invalid='ignore'):
            c_mat = text_vectorized_a.multiply(
                1 / text_vectorized_a.sum(axis=1))  # normalize counts into individual probabilities
            c_mat = np.nan_to_num(c_mat, copy=False, nan=0.5)  # convert missing to 0.5s
            c_mat = c_mat.toarray() # store in csr format for indexing
            probs = self._phrase_prob(parties_a, speakers_a, text_vectorized_a, low_memory=False)
            c_mat = da.from_array(c_mat, chunks=(int(round(c_mat.shape[0]/1000)), c_mat.shape[1]))
            res = c_mat[:, np.newaxis] / (1 - c_mat)[:, :, np.newaxis]
            diags = np.einsum('jii->ji', res)
            res = np.sum((res * probs[:, np.newaxis]) - diags[:, np.newaxis], axis=1)
            res = 0.5 - 0.5 * res
            with ProgressBar():
                res = res.compute(num_workers=4)
            res = np.mean(res, axis=1)
            res = np.nan_to_num(res, 0)
            phrases = self.vectorizer.get_feature_names()
            res = dict(zip(phrases, res))
            # FINAL IDEA: to reach the proper result, split and combine the party-wise polarization scores

        return res

    def estimate_phrases_emp(self, parties: List[str], speakers: List[str], text: List[str]):

        parties = np.array(parties, dtype=str)
        speakers = np.array(speakers, dtype=str)
        text = np.array(text, dtype=str)
        text_vectorized = self._vectorize_text(text)
        parties_a, speakers_a, text_vectorized_a = self._aggregate(parties, speakers, text_vectorized)
        post = self._polarization(parties_a, speakers_a, text_vectorized_a)
        phr = np.array(self.vectorizer.get_feature_names())
        post_change = dict()
        for p in tqdm(phr):
            p_ch = self._polarization(parties_a, speakers_a, text_vectorized_a[:, phr != p], low_memory=True)
            post_change[p] = post - p_ch

        return post_change


        return res

    def estimate_topics(self, source: list, topics: list):
        dat = pd.DataFrame(dict(source=source, topic=topics))
        dat = pd.DataFrame(dat.groupby(['source', 'topic']).size()).reset_index().pivot(index='topic', columns='source')
        dat.columns = dat.columns.droplevel(0)
        dat['n'] = dat['gov'] + dat['opp']  # get number of tweets in topic on this day
        dat['prob_gov'] = dat['gov'] / (dat['n'])  # get probability of topic in gov
        dat['prob_opp'] = 1 - dat['prob_gov']  # get probability of topic in opp
        # get posterior probability of assigning to correct party for each topic
        dat['pola'] = dat['prob_gov'] * dat['gov'] / dat['n'] + dat['prob_opp'] * dat['opp'] / dat['n']
        # get weigthed average
        pola = (dat['pola'] * dat['n'] / dat['n'].sum()).sum()
        return pola

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

        # get unique speakers and corresponding parties:
        parties_u, speakers_u = zip(*np.unique(np.c_[parties, speakers], axis=0))
        parties_u = np.array(parties_u)
        speakers_u = np.array(speakers_u)
        agg = []
        # aggregate by speaker
        for speaker in speakers_u:
            agg.append(text_vectorized[speakers == speaker].sum(axis=0))
        agg = np.vstack(agg)

        # remove 0s
        nonzero = np.array(agg.sum(axis=1) > 0).flatten()
        parties_u = parties_u[nonzero]
        speakers_u = speakers_u[nonzero]
        agg = agg[nonzero]
        agg = csr_matrix(agg)

        return parties_u, speakers_u, agg

    def _phrase_prob(self, parties, speakers, text_vectorized, leave_out=True, low_memory=True):
        """
        Get probabilities of phrases for each party.

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
            If False - compute plug in estimate of phrase-party probabilities, i.e. same for all speakers
        low_memory: bool, optional
            control whether the computation is memory-efficient (through a for loop on a dictionary, 
            using sparse arrays or processing-time efficient (using vectorization on np.array)
            default is True

        Returns
        -------
        probs : np.array
            probabilities of each phrase being assigned to a party

        """

        if leave_out:

            if low_memory:
                # this method is more memory efficient, but takes longer processing time:
                # leave out estimate (i.e. for each speaker party probabilities calculated EXCLUDING his vocabulary)
                probs = dict()
                for speaker in np.unique(speakers):  # compute for each speaker
                    ind = (speakers != speaker)  # indices of all speakers but the current one
                    denominator = text_vectorized[ind].sum(axis=0)  # phrase counts for all speakers but the current one
                    numerator = text_vectorized[(parties == self.party[0]) & ind].sum(
                        axis=0)  # same as above but only party 0
                    with np.errstate(divide='ignore', invalid='ignore'):
                        probs[speaker] = numerator / denominator  # calculate probabilities
                    probs[speaker] = np.nan_to_num(probs[speaker], nan=0.5, copy=False)
                # return a dict
            else:

                # this method is vectorized, but takes up more memory
                self.logger.debug('Computing phrase probabilities')
                text_vectorized = text_vectorized.toarray()  # convert sparse to normal
                denominators = text_vectorized.sum(
                    axis=0) - text_vectorized  # sum of all phrases for each excluding each
                p0 = text_vectorized.copy()
                p0[parties == self.party[1]] = 0  # subtract individual counts only when party is p0
                numerators = text_vectorized[parties == self.party[0]].sum(axis=0) - p0
                # in the leave out method, if a phrase was used only by particular speaker, the denominator is 0
                # in those cases, the produced nans should be replaced with 0.5, the neutral prior
                with np.errstate(divide='ignore', invalid='ignore'):
                    probs = numerators / denominators
                probs = np.nan_to_num(probs, nan=0.5, copy=False)
                self.logger.debug('Done computing probabilities')
                # return an array

        else:

            # plug in estimate
            denominator = text_vectorized.sum(axis=0)  # phrase counts for all speakers
            numerator = text_vectorized[parties == self.party[0]].sum(axis=0)  # phrase counts for party 0
            probs = defaultdict(lambda: numerator / denominator)  # calculate probabilities
            # return a dict

        return probs

    def _polarization(self, parties, speakers, text_vectorized,
                      low_memory=True, **kwargs):
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
        with np.errstate(divide='ignore', invalid='ignore'):
            c_mat = text_vectorized.multiply(
                1 / text_vectorized.sum(axis=1))  # normalize counts into individual probabilities
        c_mat = np.nan_to_num(c_mat, copy=False, nan=0.5)  # convert missing to 0.5s

        if low_memory:
            # memory-efficient version - requires id:count dictionary as input, return by the _phrase_prob method
            # if low_memory set to True
            c_mat = c_mat.tocsr()  # convert to csr
            probs = self._phrase_prob(parties, speakers, text_vectorized, low_memory=low_memory,
                                      **kwargs)  # get probabilities
            p_scores = np.zeros(c_mat.shape[0])  # store polarization scores

            for speaker in np.unique(speakers):
                ind = speakers == speaker
                prob = np.array(probs[speaker]).flatten()
                # depending on the party, probability is either proba or 1 - proba
                if np.unique(parties[ind])[0] != self.party[0]:
                    prob = 1 - prob
                counts = np.array(c_mat[ind].todense())
                p_scores[ind] = np.sum(counts * prob, axis=1)  # rowwise dot product
        else:
            # vectorized version - requires array as input:
            # depending on the party, prob is either prob or 1 -prob
            c_mat = c_mat.toarray()  # convert to csr
            probs = self._phrase_prob(parties, speakers, text_vectorized, low_memory=low_memory,
                                      **kwargs)  # get probs
            probs[parties != self.party[0]] = 1 - probs[parties != self.party[0]]  # SUBTRACTION OF ONE!
            p_scores = np.sum(c_mat * probs, axis=1)  # rowwise dot product

        p_scores[np.array(c_mat.sum(axis=1) == 0).flatten()] = 0.5  # if no phrase counted = 0.5 similarity (neutral)

        return p_scores

    def _sample(self, parties, sample_size=0.1, stratified=False):

        if stratified:
            prob_p0 = (np.sum(parties == self.party[0]) / parties.shape[0]) / np.sum(parties == self.party[0])
            prob_p1 = (np.sum(parties == self.party[1]) / parties.shape[0]) / np.sum(parties == self.party[1])
            p = np.where(parties == self.party[0], prob_p0, prob_p1)
        else:
            p = None

        inds = np.arange(0, len(parties))
        sample_size = round(len(inds) * sample_size)  # get number
        sample_inds = np.random.choice(inds, size=sample_size, p=p, replace=False)

        # return the sample
        return sample_inds

    def _bootstrap(self, speakers, parties, text, text_vectorized, sample_size=0.1, stratified=False):

        if stratified:
            prob_p0 = (np.sum(parties == self.party[0]) / parties.shape[0]) / np.sum(parties == self.party[0])
            prob_p1 = (np.sum(parties == self.party[1]) / parties.shape[0]) / np.sum(parties == self.party[1])
            p = np.where(parties == self.party[0], prob_p0, prob_p1)
        else:
            p = None

        inds = np.arange(len(parties))
        size = int(round(inds.shape[0] * sample_size, 0))  # get number
        sample_inds = np.random.choice(inds, size=size, p=p, replace=False)

        parties, speakers, text, text_vectorized = list(
            map(itemgetter(sample_inds), [parties, speakers, text, text_vectorized]))

        res = self.estimate(parties, speakers, text,
                            text_vectorized=text_vectorized,
                            conf_int=None, level="aggregate")

        return res

    def _confidence_intervals(self, samples, res):

        # sample size and point estimate
        estimate = res["point"]
        size = res["n"][self.party[0]] + res["n"][self.party[1]]

        # sizes and estimates for each of the subsamples:
        sample_sizes = np.array([elem['n']['gov'] + elem['n']['opp'] for elem in samples])
        sample_estimates = np.array([elem['point'] for elem in samples])

        # Qkt - for each subsample deviation of subsample from the mean, normalized by sqrt of the subsample's SD
        Q_kt = np.sqrt(sample_sizes) * (sample_estimates - np.mean(sample_estimates))

        # store
        stats = dict()

        # lower ci
        stats['lower_ci'] = estimate - np.partition(Q_kt, 90)[90] / np.sqrt(size)

        # upper ci
        stats['upper_ci'] = estimate - np.partition(Q_kt, 10)[10] / np.sqrt(size)

        # estimates
        stats['estimate'] = estimate

        return stats


# test
if __name__ == "__main__":
    from src.twitter_tools.utils import read_window, read_files
    from functools import partial
    import random
    import pickle
    import json
    import os
    from tqdm import tqdm

    PATH_STOPWORDS = '/home/piotr/nlp/polish.stopwords.txt'
    stopwords = []
    with open(PATH_STOPWORDS, 'r') as f:
        for line in f:
            word = line.strip('\n')
            if word != 'nie':
                stopwords.append(word)
    stopwords.append('mieć')

    PATH = "/home/piotr/projects/twitter/"
    PATH_DATA = os.path.join(PATH, "data/clean")
    PATH_DROP = os.path.join(PATH, 'results/cleaning/drop_tweets.json')
    drop_tweets = json.load(open(PATH_DROP, 'r'))
    PATH_DROP = os.path.join(PATH, 'results/cleaning/drop_users.json')
    drop_users = json.load(open(PATH_DROP, 'r'))
    dtypes = json.load(open(os.path.join(PATH, 'results/cleaning/dtypes.json'), 'r'))


    # filter function:
    def filter_fun(df: pd.DataFrame, drop_users: list, drop_tweets: list, drop_duplicates=True, keep_cols=None,
                   **kwargs):
        df = df[np.logical_not(df['user-id_str'].isin(drop_users))]  # drop IDs that are to be excluded
        df = df[np.logical_not(df['id_str'].isin(drop_tweets))]
        if drop_duplicates:
            df.drop_duplicates(inplace=True, **kwargs)
        if keep_cols is not None:
            df = df[keep_cols]
        return df


    gov = pickle.load(open(os.path.join(PATH, 'data/sample/gov_sample.p'), "rb"))
    opp = pickle.load(open(os.path.join(PATH, 'data/sample/opp_sample.p'), "rb"))
    parties = {k: "gov" for k in gov}
    parties.update({k: "opp" for k in opp})

    # Get random assignment
    random.seed(1234)
    random_keys = list(parties.keys())
    random.shuffle(random_keys)  # randomize keys
    random_values = list(parties.values())
    random.shuffle(random_values)  # randomize values
    randomized = dict(zip(random_keys, random_values))  # zip into dict

    ff = partial(filter_fun, drop_users=drop_users, drop_tweets=drop_tweets,
                 subset=['lemmatized'], keep_cols=['day', 'lemmatized', 'id_str', 'user-id_str', 'source'])

    START = '2020_02_23'
    END = '2020_07_23'
    data = next(read_files(PATH_DATA, batch_size=1, day_from=START, day_to=END, dtype=dtypes, filter_fun=ff))

    # fit vectorizer on all vocabulary:
    model = ModelPolarization(parties=["gov", "opp"], limit=40, ngram_range=(1, 2),
                              log=20, n_jobs=4, stop_words=stopwords)
    phrase_pol = model.estimate_phrases_emp(data['source'], data['user-id_str'], data['lemmatized'])
    print(sorted(phrase_pol.items(), key=lambda x: x[1], reverse=True))
