#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 15:42:37 2020

@author: piotr
"""

from sklearn.cluster import MiniBatchKMeans
from sklearn.utils.extmath import randomized_svd
import numpy as np
import dask.array as da
from nltk import FreqDist
import itertools
from sklearn.metrics import silhouette_score
import logging
from collections import defaultdict
from tqdm import tqdm
import pdb

class SentenceEmbeddings:
    def __init__(self, path_embeddings, a = 10e-3, num_workers = 4, dask_split = 10, full_vocab = False):
        self.path_embeddings = path_embeddings #path to txt file with the embeddings
        self.a = a  #normalizing constant for weighting 
        self.num_workers = num_workers #number of workers for dask operations
        self.dask_split = dask_split #how many parts shoud task array be split into
        self.randoms = [] #store documents with randomly assigned embedding vectors
        self.logger = logging.getLogger("__main__")
        self.logger.info("Initialized")
        self.embed = dict()
        self.u = None #left singular vector of the sentence embedding matrix
        self.weights = dict()
        
        
    def fit(self, text):
        
        text = [elem.split() for elem in text]
        
        vocab = self.get_vocab(text) #get vocabulary freq dist
        
        self.weights.update(self.get_weights(vocab)) #get inverse probability weights
        
        if not self.embed:
        
            self.embed.update(self.get_embeddings_words(vocab)) #load embedding vectors for vocabulary
        
        sent_embed = self.get_embeddings_sent(text, self.embed, self.weights) #get sentence embeddings
        
        if self.u is None: #this is kept in case of re-fitting (i.e. using old svd with new vocabulary)
            self.u, _, _ = randomized_svd(sent_embed, 1) #get left singular vector
            
        sent_embed = self.remove_pc(sent_embed) #remove first principal component
        
        return sent_embed
        
        
    
    
    def get_embeddings_sent(self, text, embed, weights):
        
        """
        Obtain sentence embeddings in accordance with Arora et al. 2017
        """
        
        #get embedding size to generate random vectors if no word in dictionary
        embed_size = len([elem for i, elem in enumerate(embed.values()) if i == 0].pop())
        
        sent_embed = [] #store embeddings
        self.logger.info("Calculating sentence embeddings")
        for i, sentence in tqdm(enumerate(text)):
            embedding = np.array([embed[word] for word in sentence if word in embed])
            if embedding.size == 0: #in case of all words out of embedding, initialize them at random
                res = np.random.uniform(0, 1, embed_size) 
                self.randoms.append(i) #count random vectors
            else:
                weight = np.array([weights[word] for word in sentence if word in embed]) #get weights
                res = (embedding.T @ weight)/weight.shape[0] #weighted average of constituent vectors
            sent_embed.append(res)
        #bind into one matrix:
        sent_embed = np.array(sent_embed).T
        
        return sent_embed
        
    
    def remove_pc(self, sent_embed):
        """
        Remove the first principal component from the embedding matrix

        """
        n_features, n_samples = sent_embed.shape #reverse order, since it's transposed
        if n_samples >= self.dask_split:
            sent_embed = da.from_array(sent_embed, chunks = (n_samples, int(round(n_features/self.dask_split))))
        else:
            sent_embed = da.from_array(sent_embed)
        sent_embed = (sent_embed - self.u @ self.u.T @ sent_embed).T
        sent_embed = sent_embed.compute(num_workers = self.num_workers) #store first principal component
        
        return sent_embed
            

        
    def get_vocab(self, text):
        """
        Compute frequency distribution of the vocabulary

        """
        vocab = FreqDist(itertools.chain.from_iterable(text))
        return vocab

        
    def get_weights(self, vocab):
        #inverse frequency weights
        count = sum(vocab.values())
        weights = {k:self.a/(self.a + count/v) for k, v in vocab.items()}
        return weights
        
    def get_embeddings_words(self, vocab):
        #read embeddings to dict
        embed = defaultdict(lambda: np.array([])) #alwasy return empty array if no match
        with open(self.path_embeddings, "r") as f:
            for line in tqdm(f):
                elem = line.split()
                if elem[0] in vocab.keys(): #if in frequency dist
                    embed[elem[0]] = np.array(elem[1:], dtype = np.float16)
        return embed
    
    
    
class KTopicModel(SentenceEmbeddings):
    
    def __init__(self, k, path_embeddings, batch_size = 100, **kwargs):
        super().__init__(path_embeddings, **kwargs)
        self.k = k
        self.batch_size = batch_size
        
        
    def __repr__(self):
        return f'KTopicEmbeddingModel (k = {self.k}, a = {self.a})'
        
    def fit(self, text):
        """
        

        Parameters
        ----------
        text : list of strings
            Text to be clustered.

        Returns
        -------
        None.

        """
        
        sent_embed = super().fit(text)
        
        self.cluster = MiniBatchKMeans(n_clusters = self.k, batch_size = self.batch_size) #init clusterer
        
        self.cluster.fit(sent_embed) #cluster
        
        self.fitted_labels = self.cluster.labels_ #store labels
        
    def predict(self, text, refit = False):
        """
        

        Parameters
        ----------
        text : TYPE
            DESCRIPTION.
        refit : bool, optional
            If True, re-run the init (without PCA) to obtain complete 
            vocabulary and weights. The default is False.

        Returns
        -------
        lbl : TYPE
            DESCRIPTION.

        """
        
        
        
        if refit: #obtain complete sentence embeddings and vocabulary; use 
            sent_embed = super().fit(text)
        else:
            text = [elem.split() for elem in text]
            sent_embed = self.get_embeddings_sent(text, self.embed, self.weights)
            sent_embed = self.remove_pc(sent_embed)
            
        
        
        lbl = self.cluster.predict(sent_embed)
        
        return lbl 
    
    def evaluate(self, text, sample_size, n_samples = 50):
        """
        Get silhuette score by sampling

        Parameters
        ----------
        text : TYPE
            DESCRIPTION.
        sample_size : TYPE
            DESCRIPTION.
        n_samples : TYPE, optional
            DESCRIPTION. The default is 30.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        scores = []
        for _ in range(n_samples):
            sample = np.random.choice(text, sample_size)
            lbl = self.predict(sample, refit = True)
            sample = [word.split() for word in sample]
            sent_embed = self.get_embeddings_sent(sample, self.embed, self.weights).T
            try:
                scores.append(silhouette_score(sent_embed, lbl))
            except:
                continue
        return np.mean(scores)
    
        
            