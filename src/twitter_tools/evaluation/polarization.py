#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 11:08:15 2020

@author: piotr
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit
import statsmodels.api as sm


def plot_aggregated(date: list,
                    true_estimate: list, random_estimate: list = None,
                    true_ci: tuple = None, random_ci: tuple = None,
                    trend_line: bool = False, lowess: bool = False,
                    tick_freq: int = 10, ax=None, legend: bool=False):
    if ax is None:
        plt.gca()
    ax.plot(date, true_estimate, lw=3, label='Leave out estimator', color='blue')
    if random_estimate is not None:
        ax.plot(date, random_estimate, lw=3, linestyle="--", label='Random assignment', color='red')
    if true_ci is not None:
        ax.fill_between(date, true_ci.iloc[:, 1], true_ci.iloc[:, 0], facecolor='blue', alpha=0.5)
    if random_ci is not None:
        ax.fill_between(date, random_ci.iloc[:, 1], random_ci.iloc[:, 0], facecolor='red', alpha=0.5)
    if trend_line:
        date_num = np.arange(len(date))
        b, m = polyfit(date_num, true_estimate.to_numpy(), 1)
        ax.plot(date_num, b + m * date_num, '-', color='red')
    if lowess:
        date_num = np.arange(len(date))
        preds = sm.nonparametric.lowess(true_estimate.to_numpy(), date_num)
        ax.plot(preds[:, 0], preds[:, 1], '-', color='green')

    if legend:
        ax.legend(loc='upper left')
    else:
        ax.legend().remove()
    ax.xaxis.set_tick_params(rotation=45)
    ax.xaxis.set_ticks(np.arange(len(date), step=tick_freq))
    ax.set_xlabel('Date')
    ax.set_ylabel('Polarization')
    ax.grid()
    return ax


def plot_aggregated_bytopic(date: list, true_estimate: list, topics: list,
                            figsize: tuple = (10, 10), tick_freq: int = 20,
                            trend_line: bool = False, lowess: bool = True,
                            dims: tuple = None, error: list = None):
    unique_topics = sorted(np.unique(topics))
    date = np.unique(date)
    if dims is None:
        dim1 = dim2 = int(np.ceil(np.sqrt(len(unique_topics))))
    else:
        dim1, dim2 = dims
    fig, ax = plt.subplots(dim1, dim2, figsize=figsize, sharey=True, sharex=True)
    ax = ax.ravel()
    for i in range(dim1 * dim2):
        if i >= dim1 ** dim2 - dim2:
            ax[i].set_xlabel('Date')
        if i in range(0, dim1 * dim2, dim1):
            ax[i].set_ylabel('Polarization')
        ax[i].xaxis.set_tick_params(rotation=45)
        if i < len(unique_topics):
            ax[i].xaxis.set_ticks(np.arange(len(date), step=tick_freq))
            ax[i].plot(date, true_estimate[topics == unique_topics[i]], lw=3,
                       label='Leave out estimator', color='blue')
            if trend_line:
                date_num = np.arange(len(date))
                b, m = polyfit(date_num, true_estimate[topics == unique_topics[i]].to_numpy(), 1)
                ax[i].plot(date_num, b + m * date_num, '-', color='red')

            if lowess:
                date_num = np.arange(len(date))
                preds = sm.nonparametric.lowess(true_estimate[topics == unique_topics[i]].to_numpy(), date_num)
                ax[i].plot(preds[:, 0], preds[:, 1], '-', color='orange')
                
            if error is not None:
                ax[i].errorbar(date, true_estimate[topics == unique_topics[i]], yerr=error[:, topics == unique_topics[i]])

            ax[i].set_title(f'Topic {unique_topics[i]}')
            ax[i].grid()
    return fig
