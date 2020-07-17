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
from sklearn.manifold import TSNE
import seaborn as sns
import pdb
from nltk.cluster import KMeansClusterer
from nltk.cluster.util import cosine_distance


class SentenceEmbeddings:
    
    
    def __init__(self, path_embeddings, a = 10e-3, num_workers = 4, dask_split = 10):
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
        
    def fit(self, text):
        """
        Compute sentence embeddings and return them.

        Parameters
        ----------
        text : itereable of strings.

        Returns
        -------
        sent_embed : np.array

        """
        
        #TOKENIZE:
        text = [elem.split() for elem in text]
        
        #PROBABILITY WEIGHTS:
        #compute inverse probability weights
        weights = self._get_weights(text)
        
        #WORD EMBEDDINGS:
        self.word_embed.update(self._get_embeddings_words(self.vocab))
        
        #SENTENCE EMBEDDINGS:
        self.logger.debug(f"Computing sentence embeddings.")
        sent_embed = self._get_embeddings_sent(text, self.word_embed, weights) #get sentence embeddings
        
        if self.u is None: #this is kept in case of re-fitting (i.e. using old svd with new vocabulary)
            self.u, _, _ = randomized_svd(sent_embed, 1) #get left singular vector
            
        sent_embed = self._remove_pc(sent_embed) #remove first principal component
        
        
        return sent_embed
    
    def predict(self, text:list, refit:bool = False):
        """
        Compute sentence embeddings based on fitted data

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
        
        if refit: #obtain complete word embeddings for new data; 
            sent_embed = super().fit(text)
        else: #use word vectors loaded at fitting
            text = [elem.split() for elem in text]
            weights = self._get_weights(text)
            sent_embed = self._get_embeddings_sent(text, self.word_embed, weights)
            sent_embed = self._remove_pc(sent_embed)
            
        return sent_embed
    
    def plot(self, text, **kwargs):
        """
        Visualize sentence embeddings using tSNE
        """
        
        embeds = self.fit(text)
        tsne = TSNE()
        dat = tsne.fit_transform(embeds)
        sns.scatterplot(dat[:,0], dat[:,1], **kwargs)
        plt.title("t-SNE projection of sentence embeddings")
        
        
        
    
    
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
        sent_embed = np.array(sent_embed).T
        
        return sent_embed
    
        
    
    def _remove_pc(self, sent_embed):
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
        embeddings = super().fit(text)
        self.clf.fit(embeddings)
        return self
    
    def predict(self, text, refit = False):
        embeddings = super().predict(text, refit = refit)
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
        
        sent_embed = super().fit(text)
        
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
        sent_embed = super().predict(text, refit = refit)
        
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
        text = [word.split() for word in text]
        weights = self._get_weights(text, refit = refit)
        sent_embed = self._get_embeddings_sent(text, self.word_embed, weights).T
        score = silhouette_score(sent_embed, lbl, 
                                 sample_size = sample_size, 
                                 random_state = 1234, **kwargs)
        return score
    
    
class DBSCANTopics(SentenceEmbeddings):
    
    def __init__(self, eps, min_samples, path_embeddings, **kwargs):
        super().__init__(path_embeddings, **kwargs)
        self.cluster = DBSCAN(eps, min_samples, metric = 'cosine')
        
    def fit(self, text:list):
        sent_embed = super().fit(text)
        self.cluster.fit(sent_embed) #cluster
    
    def predict(self, text:list):
        sent_embed = super().fit(text)
        lbl = self.cluster.predict(sent_embed) #cluster
        return lbl
    
class KTopicModelCosine(SentenceEmbeddings):
    
    def __init__(self, k, path_embeddings, **kwargs):
        super().__init__(path_embeddings, **kwargs)
        self.k = k
        self.cluster = KMeansClusterer(k, distance = cosine_distance)
        
    def fit(self, text:list):
        sent_embed = super().fit(text)
        self.cluster.cluster_vectorspace(sent_embed)
        
    def predict(self, text:list, return_closest:bool = False, refit:bool = False):
        
        sent_embed = super().predict(text, refit = refit)
        
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
        text = [word.split() for word in text]
        weights = self._get_weights(text, refit = refit)
        sent_embed = self._get_embeddings_sent(text, self.word_embed, weights).T
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
        
        sent_embed = super().fit(text)
        p0_p0 = self._similarity(sent_embed[source == self.party[0]], sent_embed[source == self.party[0]])
        p0_p1 = self._similarity(sent_embed[source == self.party[0]], sent_embed[source == self.party[1]])
        p1_p0 = self._similarity(sent_embed[source == self.party[1]], sent_embed[source == self.party[0]])
        p1_p1 = self._similarity(sent_embed[source == self.party[1]], sent_embed[source == self.party[1]])
        
        p0_pol = np.sum(source == self.party[0])/len(source) * (p0_p0 - p0_p1)
        p1_pol = np.sum(source == self.party[1])/len(source) * (p1_p1 - p1_p0)
        
        polarization = p0_pol + p1_pol
        
        return polarization
        
    def _similarity(self, arr1, arr2):
        arr1 = da.from_array(arr1, chunks = (10000, arr1.shape[1]))
        arr2 = da.from_array(arr2, chunks = (10000, arr2.shape[1]))
        numer = arr1 @ arr2.T
        denom = np.outer(np.linalg.norm(arr1, axis = 1), np.linalg.norm(arr2, axis = 2))
        sims = np.mean(numer/denom)
        sims = sims.compute(num_workers = self.num_workers)
        return sims
    
    
    
if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    test = pd.read_csv('/home/piotr/projects/twitter/test/test_data.csv', index_col = 0)
    path_embeddings = '/home/piotr/nlp/cc.pl.300.vec'
    model = PolarizationEmbeddings(path_embeddings, parties = ['gov', 'opp'])
    res = model.compute(test.lemmatized.astype(str), test.source)

    
    
    