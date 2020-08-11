#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 15:42:37 2020

@author: piotr
"""

from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.utils.extmath import randomized_svd
import numpy as np
import dask.array as da
from nltk import FreqDist
import itertools
from sklearn.metrics import silhouette_score
import logging
from collections import defaultdict
import pdb
from nltk.cluster import KMeansClusterer
from nltk.cluster.util import cosine_distance
import gc

class SentenceEmbeddings:
    
    
    def __init__(self, path_embeddings, a = 10e-3, ncomp = 1, num_workers = 4, dask_split = 10):
        self.path_embeddings = path_embeddings #path to txt file with the embeddings
        self.a = a  #normalizing constant for weighting 
        self.num_workers = num_workers #number of workers for dask operations
        self.dask_split = dask_split #how many parts shoud task array be split into
        self.randoms = [] #store documents with randomly assigned embedding vectors
        self.logger = logging.getLogger(__name__)
        self.logger.info("Model initialized")
        self.word_embed = dict() #dict of word embeddings
        self.u = None #left singular vector of the sentence embedding matrix
        self.vocab = None #freq dist of the text
        self.ncomp = ncomp #number of components to remove in SVD
        
    def fit(self, text:list):
        """
        Obtain weights and first principal component

        Parameters
        ----------
        text : list
            tokenized text.

        Returns
        -------
        self

        """
        #PROBABILITY WEIGHTS:
        #compute inverse probability weights
        weights = self._get_weights(text)
        
        #WORD EMBEDDINGS:
        self.logger.debug("Reading sentence embeddings")
        self.word_embed.update(self._get_embeddings_words(self.vocab))
        
        #SENTENCE EMBEDDINGS:
        self.logger.debug("Computing sentence embeddings.")
        sent_embed = self._get_embeddings_sent(text, self.word_embed, weights) #get sentence embeddings
        
        if self.u is None: #this is kept in case of re-fitting (i.e. using old svd with new vocabulary)
            self.u, _, _ = randomized_svd(sent_embed.T, self.ncomp) #get left singular vector of transposed
        
        
        return self
    
    
    def transform(self, text):
        """
        Compute sentence embeddings vectors

        Parameters
        ----------
        text : list
            tokenized text.

        Returns
        -------
        sent_embed : np.array
            sentence embeddings.

        """
        weights = self._get_weights(text, refit = False)
        sent_embed = self._get_embeddings_sent(text, self.word_embed, weights)
        if self.ncomp > 0:
            sent_embed = self._remove_pc(sent_embed)
        return sent_embed
        
        
        
    def fit_transform(self, text:list, refit:bool = False):
        """
        Peform fitting and return sentence embedding vectors.

        Parameters
        ----------
        text : list
            DESCRIPTION.
        refit : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        sent_embed : TYPE
            DESCRIPTION.

        """
        self.fit(text)
        sent_embed = self.transform(text)
        return sent_embed
        
    
    def _get_embeddings_sent(self, text, embed, weights):
        
        """
        Obtain sentence embeddings in accordance with Arora et al. 2017
        """
        
        #convert to defaultdict so empty array always returned
        embed = defaultdict(lambda: np.array([]), embed) 
        
        #get embedding size to generate random vectors if no word in dictionary
        embed_size = len([elem for i, elem in enumerate(embed.values()) if i == 0].pop())
        
        sent_embed = [] #store embeddings
        self.logger.debug(f"Calculating sentence embeddings. Text size {len(text)}")
        for i, sentence in enumerate(text):
            embedding = np.array([embed[word] for word in sentence if word in embed])
            if embedding.size == 0: #in case of all words out of embedding, initialize them at random
                res = np.random.uniform(0, 1, embed_size) 
                self.randoms.append(sentence) #count random vectors
            else:
                weight = np.array([weights[word] for word in sentence if word in embed]) #get weights
                res = (embedding.T @ weight)/weight.shape[0] #weighted average of constituent vectors
            sent_embed.append(res)
        #bind into one matrix:
        sent_embed = np.array(sent_embed)
        self.logger.debug(f'Calculated sentence embeddings. Shape {sent_embed.shape}')
        
        return sent_embed
    
        
    
    def _remove_pc(self, sent_embed: np.array):
        """
        Remove the first principal component
        Parameters
        ----------
        sent_embed : np.array [n_samples, emb_dim]
            Array of sentence embeddings.

        Returns
        -------
        sent_embed : np:array [n_samples, emb_dim]
        Array of sentence embeddings after removing first principal component.

        """
        sent_embed = sent_embed.T #transpose
        emb_dim, n_samples = sent_embed.shape #get shape for chunking
        if n_samples > self.dask_split:
            j_chunk =  int(n_samples/self.dask_split) #get chunk size
        else:
            j_chunk = n_samples
        sent_embed = da.from_array(sent_embed, chunks = (emb_dim, j_chunk))

        #remove first principal component and transpose back
        sent_embed = (sent_embed - self.u @ self.u.T @ sent_embed).T
            
        #run the operations
        sent_embed = sent_embed.compute(num_workers = self.num_workers)
        
        return sent_embed

        
    def _get_weights(self, text, refit = True):
        #inverse frequency weights
        #in each case, vocab temporary variable stores vocabulary distribution
        #refit argument controls, whether weights are updated with new data
        if self.vocab is None: #if first run - get word counts
            self.vocab = FreqDist(itertools.chain.from_iterable(text)) #freq dist
        elif refit: #update word counts with new data
            new_vocab = FreqDist(itertools.chain.from_iterable(text))
            for k, v in new_vocab.items():
                self.vocab[k] += v
        
        #return normalized
        count = sum(list(self.vocab.values()))
        weights = {k:self.a/(self.a + (v/count)) for k, v in self.vocab.items()}
        median_prob = np.median(list(self.vocab.values()))
        weights = defaultdict(lambda: median_prob, weights)
        return weights
        
    def _get_embeddings_words(self, vocab):
        self.logger.info("Reading word embeddings")
        #read embeddings to dict
        embed = dict()
        with open(self.path_embeddings, "r", encoding = 'utf8') as f:
            for line in f:
                elem = line.split()
                if elem[0] in vocab.keys(): #if in frequency dist
                    embed[elem[0]] = np.array(elem[1:], dtype = np.float16)
        return embed
    
    
class OutlierDetector(SentenceEmbeddings):
    
    def __init__(self, path_embeddings, method, **kwargs):
        super().__init__(path_embeddings, **kwargs)
        self.clf = method
        
    
    def fit(self, text):
        super(OutlierDetector, self).fit(text)
        embeddings = super(OutlierDetector, self).transform(text)
        self.clf.fit(embeddings)
        return self
    
    def predict(self, text, refit = False):
        if refit:
            embeddings = super(OutlierDetector, self).fit_transform(text)
        else:
            embeddings = super(OutlierDetector, self).transform(text)
        scores = self.clf.predict(embeddings)
        return scores
        
    
    
    
class KTopicModel(SentenceEmbeddings):
    
    def __init__(self, k, path_embeddings, batch_size = 100, **kwargs):
        super().__init__(path_embeddings, **kwargs)
        self.k = k
        self.batch_size = batch_size
        self.cluster = MiniBatchKMeans(n_clusters = self.k, batch_size = self.batch_size) #init clusterer

    def __repr__(self):
        return f'KTopicEmbeddingModel (k = {self.k}, a = {self.a})'
        
    def fit(self, text):
        """
        Parameterstext
        ----------
        text : list of strings
            Text to be clustered.
        """
        
        super(KTopicModel, self).fit(text)
        sent_embed = super(KTopicModel, self).transform(text)
        
        self.cluster.fit(sent_embed) #cluster
        
        self.centroids = dict()
        
        for k, cluster in enumerate(self.cluster.cluster_centers_):
            self.centroids[k] = cluster
        
    def predict(self, text:list, refit:bool = False, return_closest:bool = False):
        """
        Predict clusters/similarity to each centroid. 

        Parameters
        ----------
        text : iterable of strings
        refit : bool, optional. The default is False.
        return_closest : bool, optional. The default is False.

        Returns
        -------
        lbl : iterable

        """
        if refit:
            sent_embed = super(KTopicModel, self).fit_transform(text)
        else:
            sent_embed = super(KTopicModel, self).transform(text)
        
        #either return the exact cluster or cosine similarity to each of the clusters
        if return_closest:
            lbl = self.cluster.predict(sent_embed)
        else:
            lbl = []
            for embed in sent_embed:
                lbl.append({k:1 - cosine_distance(embed, v) for k, v in self.centroids.items()})
        return lbl
    
    def evaluate(self, text:list, sample_size:int, refit:bool = False, **kwargs):
        """
        

        Parameters
        ----------
        text : list
            DESCRIPTION.
        sample_size : int
            DESCRIPTION.
        refit : bool, optional
            DESCRIPTION. The default is False.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        score : TYPE
            DESCRIPTION.

        """
        
        lbl = self.predict(text, refit = refit, return_closest = True)
        sent_embed = super().transform(text)
        score = silhouette_score(sent_embed, lbl, 
                                 sample_size = sample_size, 
                                 random_state = 1234, **kwargs)
        return score
    
    
class DBSCANTopics(SentenceEmbeddings):
    
    def __init__(self, eps, min_samples, path_embeddings, **kwargs):
        super().__init__(path_embeddings, **kwargs)
        self.cluster = DBSCAN(eps, min_samples, metric = 'cosine')
        
    def fit(self, text:list):
        sent_embed = super().fit_transform(text)
        self.cluster.fit(sent_embed) #cluster
    
    def predict(self, text:list):
        sent_embed = super().transform(text)
        lbl = self.cluster.predict(sent_embed) #cluster
        return lbl
    
class KTopicModelCosine(SentenceEmbeddings):
    
    def __init__(self, k, path_embeddings, **kwargs):
        super().__init__(path_embeddings, **kwargs)
        self.k = k
        self.cluster = KMeansClusterer(k, distance = cosine_distance)
        
    def fit(self, text:list):
        sent_embed = super().fit_transform(text)
        self.cluster.cluster_vectorspace(sent_embed)
        
    def predict(self, text:list, return_closest:bool = False, refit:bool = False):
        
        if refit:
            sent_embed = super().fit_transform(text)
        else:
            sent_embed = super().transform(text)
        
        #either return the exact cluster or cosine similarity to each of the clusters
        if return_closest:
            lbl = [self.cluster.classify_vectorspace(x) for x in sent_embed]
        else:
            lbl = []
            for embed in sent_embed:
                lbl.append({i:1 - cosine_distance(embed, v) for i, v in enumerate(self.cluster.means())})
        return lbl
    
    def evaluate(self, text:list, sample_size:int, refit:bool = False, **kwargs):

        
        lbl = self.predict(text, refit = refit, return_closest = True)
        weights = self._get_weights(text, refit = refit)
        sent_embed = self._get_embeddings_sent(text, self.word_embed, weights)
        score = silhouette_score(sent_embed, lbl, 
                                 sample_size = sample_size, 
                                 random_state = 1234, **kwargs)
        return score
        
    
    
class PolarizationEmbeddings(SentenceEmbeddings):
    
    def __init__(self, path_embeddings, parties, **kwargs):
        super().__init__(path_embeddings, **kwargs)
        assert (type(parties) == list and len(parties) == 2)
        self.party = parties
    
    def compute(self, text, source):
        
        sent_embed = super().fit_transform(text) #get sentence embeddings
        source = np.array(source, dtype = str) #convert source to np array
        
        gc.collect()
        
        #compute similarities
        P0 = np.array(sent_embed[source == self.party[0]], copy = True)
        P1 = np.array(sent_embed[source == self.party[1]], copy = True)
        p0_p0 = self._similarity(P0, P0)
        p0_p1 = self._similarity(P0, P1)
        p1_p0 = self._similarity(P1, P0)
        p1_p1 = self._similarity(P1, P1)
        
        #get polarization
        p0_pol = np.sum(source == self.party[0])/len(source) * (p0_p0 - p0_p1)
        p1_pol = np.sum(source == self.party[1])/len(source) * (p1_p1 - p1_p0)
        
        polarization = p0_pol + p1_pol
        
        return polarization
        
    def _similarity(self, arr1, arr2):
        A = da.from_array(arr1, chunks = (10000, arr1.shape[1]))
        B = da.from_array(arr2, chunks = (10000, arr2.shape[1]))
        sims = (A @ B.T)/(np.outer(np.linalg.norm(A, axis = 1), np.linalg.norm(B, axis = 1)))
        return sims.mean().compute(num_workers = self.num_workers)
    

    
if __name__ == "__main__":
    pass

    
    
    