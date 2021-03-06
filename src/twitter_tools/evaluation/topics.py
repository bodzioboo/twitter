#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 15:11:10 2020

@author: piotr
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import chain
from nltk import FreqDist
from ..utils import batch
from ...twitter_models.embeddings import KTopicModel
from gensim.models import KeyedVectors


def plot_n_closest(ktp: KTopicModel, gensim_model: KeyedVectors, topn=20):
    """
    Function to plot n closest words in terms of word vectors
    """
    # get n closest words to each centroid:
    res = dict()
    for k, center in enumerate(ktp.cluster.means()):
        res[k] = gensim_model.similar_by_vector(center, topn=topn)

    # plot
    fig = plt.figure(figsize=(15, 10))
    for i in res:
        ax = fig.add_subplot(np.ceil(np.sqrt(len(res))), np.ceil(np.sqrt(len(res))), i + 1, )
        ax.barh([j[0] for j in res[i]], [j[1] for j in res[i]])
        ax.set_title(f'Topic {i}')
    plt.tight_layout()
    plt.show()


def predict_clusters(model, text: list, batch_size: int = 500000):
    """
    Function used to predict clusters in batches to limit memory usage.
    """
    clusters = []
    for txt in tqdm(batch(text, batch_size)):
        clusters.extend(model.predict(text))
    clusters = dict(zip(text, clusters))
    return clusters


def map_clusters(data: pd.DataFrame, cls: dict, col_id: str = 'lemmatized'):
    """
    Function to map clustering dictionary onto the data by identifying variable
    """
    data['cluster'] = data[col_id].map(cls)
    data.dropna(subset=['cluster'], inplace=True)
    data.reset_index(inplace=True, drop=True)
    return data


def eval_topics(text: list, labels: list, n: int = 100):
    """
    Function to get distributions of most common
    tokens in each topic that were not amongst the 
    most common tokens in other topics.
    """
    dists = dict()
    for i in np.unique(labels):
        txt = text[labels == i]
        freq = FreqDist(chain.from_iterable(txt))
        dists[i] = freq.most_common(n)
    dists_unique = dict()
    for i, dist in dists.items():
        # get vocabulary of all other topics but i
        rest = []
        _ = [rest.extend([el[0] for el in v]) for k, v in dists.items() if k != i]
        # filter out those not included in the vocabulary of topic i
        dists_unique[i] = {k: v for k, v in dist if k not in rest}
    return dists_unique


def clustering_summary(text: list, labels: list, source: np.array, **kwargs):
    """
    Function to create summaries of the clustering results - most common tokens
    in each topic + the split among the parties and count
    """
    dists = eval_topics(text, labels, **kwargs)
    dists_nocounts = {k: [", ".join(list(v))] for k, v in dists.items()}
    df = pd.DataFrame.from_dict(dists_nocounts, orient="index")
    sources = pd.crosstab(labels, source)
    df = pd.merge(df, sources, left_index=True, right_index=True)
    df["count"] = df[["gov", "opp"]].sum(axis=1)
    df.columns = ["tokens", "gov", "opp", "count"]
    return df, dists


def plot_words(dists, **kwargs):
    """
    Function to plot most common tokens in each topic
    in one figure.
    """
    fig_side = np.ceil(np.sqrt(len(dists)))
    fig = plt.figure(**kwargs)
    for i, (k, fdist) in enumerate(dists.items()):
        ax = fig.add_subplot(fig_side, fig_side, i + 1)
        ax.set_title(f"Topic {k}")
        ax.barh(y=list(fdist.keys())[:10], width=list(fdist.values())[:10])
    plt.tight_layout()
    plt.show()


def plot_dists(data: pd.DataFrame, topic_subset: list = None,
               topic_dict: dict = None, **kwargs):
    """
    Function to plot token distributions of the clusters.
    """

    # prepare plot
    fig, ax = plt.subplots(3, 1, **kwargs)
    ax = ax.ravel()

    if topic_subset is not None:
        data = data.loc[data.cluster.isin(topic_subset)]

    if topic_dict is not None:
        data['cluster'] = data['cluster'].astype(str) + " : " + data['cluster'].map(topic_dict)

    # OVERALL DISTRIBUTION
    tab = pd.value_counts(data.cluster, normalize=True)
    tab.plot.bar(ax=ax[0])
    # get mapping for the next plot:
    mapping = {k: i for i, (k, v) in enumerate(sorted(tab.items(), key=lambda x: x[1], reverse=True))}
    ax[0].axes.get_xaxis().set_visible(False)

    # BETWEEN-PARTY DISTRIBUTION
    crosstab = pd.crosstab(data.cluster, data.source, normalize="index")
    # order according to size:
    key = crosstab.reset_index().cluster.map(mapping)
    crosstab = crosstab.iloc[key.argsort()]
    crosstab.plot.bar(ax=ax[1], stacked=True)
    if topic_dict is None:
        ax[1].set_xlabel("Topic number")
    else:
        ax[1].set_xlabel("Topic title")
    ax[1].legend(["Government", "Opposition"])

    # TEMPORAL DISTRIBUTION
    pd.crosstab(data.day, data.cluster, normalize="index").plot.bar(rot=45, ax=ax[2])
    ticks = np.arange(0, len(data.day.unique()), 4)  # every fourth tick
    ax[2].set_xticks(ticks)
    ax[2].set_xticklabels(np.array(sorted(data.day.unique()))[ticks])

    # label
    font = {'size': 18}
    fig.text(-0.01, 0.5, 'Propotion of tweets', va='center', rotation='vertical', fontdict=font)
    # show
    plt.tight_layout()
    plt.show()
