#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 11:08:15 2020

@author: piotr
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import polyfit



def plot_aggregated(data:dict, tick_freq:int=10, trend_line = False):
    aggregated = {day:{k + "_" + k1:v1 for k, v in res.items() if k != "disaggregated" for k1, v1 in v.items()} for day, res in data.items()}
    aggregated = pd.DataFrame(aggregated).T.reset_index()
    fig, ax = plt.subplots(1,figsize = (10,10))
    date = np.array(list(data.keys()))
    ax.plot(date, aggregated["true_estimate"], lw = 3, label='Leave out estimator', color='blue')
    ax.plot(date, aggregated["random_estimate"], lw = 3, linestyle = "--", label='Random assignment', color='red')
    ax.fill_between(date, aggregated["true_upper_ci"], aggregated["true_lower_ci"], facecolor='blue', alpha=0.5)
    ax.fill_between(date, aggregated["random_upper_ci"], aggregated["random_lower_ci"], facecolor='red', alpha=0.5)
    if trend_line:
        date_num = np.arange(len(date))
        b, m = polyfit(date_num, aggregated['true_estimate'].to_numpy(), 1)
        ax.plot(date_num, b + m * date_num, '-')
    ax.legend(loc='upper left')
    ax.xaxis.set_tick_params(rotation=45)
    ax.xaxis.set_ticks(np.arange(len(date), step = tick_freq))
    ax.set_xlabel('Date')
    ax.set_ylabel('Polarization')
    ax.grid()
    plt.show()

def plot_aggregated_bytopic(bytopic:list, figsize:tuple=(10, 10), tick_freq:int = 20, 
                            topic_subset:list = None, topic_dict:dict = None):

    if topic_subset is not None:
        bytopic = {k:v for k,v in bytopic.items() if k in topic_subset}
    dim = int(np.ceil(np.sqrt(len(bytopic))))
    fig, ax = plt.subplots(dim, dim, figsize = figsize, sharey = True, sharex = True)
    ax = ax.ravel()
    for i in range(dim**2):
        if i >= dim**2 - dim:
            ax[i].set_xlabel('Date')
        if i in range(0, dim**2, dim):
            ax[i].set_ylabel('Polarization')
        ax[i].xaxis.set_tick_params(rotation=45)
        ax[i].xaxis.set_tick_params(rotation=45)
        if i+1 in bytopic.keys():
            aggregated = pd.DataFrame(bytopic[i+1].items())
            aggregated.columns = ['day','true_estimate']
            ax[i].xaxis.set_ticks(np.arange(len(aggregated['day']), step = tick_freq))
            ax[i].plot(aggregated['day'], aggregated["true_estimate"], lw = 3, label='Leave out estimator', color='blue')
            if topic_dict is not None:
                ax[i].set_title(f'Topic: {topic_dict[str(i + 1)]}')
            else:
                ax[i].set_title(f'Topic number {i + 1}')
            ax[i].grid()
    plt.tight_layout()
    plt.show()
    
    
