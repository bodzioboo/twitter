#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 11:08:15 2020

@author: piotr
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit



def plot_aggregated(date:list, 
                    true_estimate:list, random_estimate:list = None, 
                    true_ci:tuple = None, random_ci:tuple = None,
                    trend_line:bool = False, tick_freq:int = 10):
    fig, ax = plt.subplots(1,figsize = (10,10))
    ax.plot(date, true_estimate, lw = 3, label='Leave out estimator', color='blue')
    if random_estimate is not None:
        ax.plot(date, random_estimate, lw = 3, linestyle = "--", label='Random assignment', color='red')
    if true_ci is not None:
        ax.fill_between(date, true_ci.iloc[:,1], true_ci.iloc[:,0], facecolor='blue', alpha=0.5)
    if random_ci is not None:
        ax.fill_between(date, random_ci.iloc[:,1], random_ci.iloc[:,0], facecolor='red', alpha=0.5)
    if trend_line:
        date_num = np.arange(len(date))
        b, m = polyfit(date_num, true_estimate.to_numpy(), 1)
        ax.plot(date_num, b + m * date_num, '-', color = '')
    ax.legend(loc='upper left')
    ax.xaxis.set_tick_params(rotation=45)
    ax.xaxis.set_ticks(np.arange(len(date), step = tick_freq))
    ax.set_xlabel('Date')
    ax.set_ylabel('Polarization')
    ax.grid()
    return fig, ax
    
    

    


def plot_aggregated_bytopic(date:list, true_estimate:list, topics:list,
                            figsize:tuple=(10, 10), tick_freq:int = 20, 
                            trend_line:bool = False,
                            topic_subset:list = None, 
                            dims:tuple = None):
    unique_topics = sorted(np.unique(topics))
    date = np.unique(date)
    if dims is None:
        dim1 = dim2 = int(np.ceil(np.sqrt(len(unique_topics))))
    else:
        dim1, dim2 = dims
    fig, ax = plt.subplots(dim1, dim2, figsize = figsize, sharey = True, sharex = True)
    ax = ax.ravel()
    for i in range(dim1*dim2):
        if i >= dim1**dim2  - dim2:
            ax[i].set_xlabel('Date')
        if i in range(0, dim1*dim2, dim1):
            ax[i].set_ylabel('Polarization')
        ax[i].xaxis.set_tick_params(rotation=45)
        if i < len(unique_topics):
            ax[i].xaxis.set_ticks(np.arange(len(date), step = tick_freq))
            ax[i].plot(date, true_estimate[topics == unique_topics[i]], lw = 3, 
                       label='Leave out estimator', color='blue')
            if trend_line:
                date_num = np.arange(len(date))
                b, m = polyfit(date_num, true_estimate[topics == unique_topics[i]].to_numpy(), 1)
                ax[i].plot(date_num, b + m * date_num, '-', color = 'red')
            ax[i].set_title(f'Topic {unique_topics[i]}')
            ax[i].grid()
    plt.tight_layout()
    plt.show()
    
    
