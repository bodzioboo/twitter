#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 16:32:53 2020

@author: piotr
"""

import pandas as pd
import networkx as nx
from collections import defaultdict
import pdb
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def add_edge_attributes(df, network, col_label:str, cols_attrs:list):
        """
        
    
        Parameters
        ----------
        df : pd.DataFrame
            data source.
        network : networkx graph
            DESCRIPTION.
        col_label: str
            name of column with edge labels
        cols_attrs : list
            name of columns to be used as edge attributes
    
        Returns
        -------
        network : networkx Graph
            DESCRIPTION.
    
        """
        
        #get edges in df:    
        labels = set(df[col_label].tolist())
        edges = list(filter(lambda x: x[2] in labels, list(network.edges.keys()))) 
        
        #sort edges by col_label:
        #edges.sort(key = lambda x: labels.index(x[2]))
        df = df.sort_values(by = [col_label])
        edges.sort(key = lambda x: x[2])
        
        
        
        #get attribute dictionary
        attrs = dict(zip(df[col_label], df[cols_attrs].to_dict('records')))
        attrs = {k:attrs[k[2]] for k in edges}
        
        
        #set attributes
        
        nx.set_edge_attributes(network, attrs)
        
        
        return network



def build_twitter_network_multi(df: pd.DataFrame, col_node: str, col_label:str, 
                                cols_from, cols_to, network = None, cols_attrs: list = None, 
                                nodes: list = None, filter_fun = None):
    """
    

    Parameters
    ----------
    df : pd.DataFrame
        data source.
    col_node : str
        name of column indicating node.
    col_label: str
        name of column with edge labels
    cols_from : str
        name of the column indicating connections from the node.
    cols_to : str
        name of the column indicating connections to the node.
    network : nx.MultiDiGraph class instance
        existing networkx graph. The default is None.
    cols_attrs : list, optional
        name of columns to be used as edge attributes. The default is None.
    nodes: list
        preset list of nodes to use
    flter_fun: function filtering the data frame

    Returns
    -------
    network : networkx Graph
        DESCRIPTION.

    """
    
    #filter the data frame
    if filter_fun:
        df = filter_fun(df)
    
    #init network if not provided:
    if network is None:
        network = nx.MultiDiGraph()
        
    #add nodes:
    if not nodes:
        nodes = df[col_node].unique().tolist()
        network.add_nodes_from([node for node in nodes if node not in network.nodes()])
    
    #for all n1 -> n2 variables:
    for var in cols_from:
        edges = zip(df[col_node].tolist(), df[var], df[col_label].tolist())
        edges = [(n1, n2, w) for n1, n2, w in edges if pd.notna(n2)]
        network.add_edges_from(edges)
    
    #for all n2 -> n1 variables:
    for var in cols_to:
        edges = zip(df[var].tolist(), df[col_node], df[col_label].tolist())
        edges = [(n1, n2, w) for n1, n2, w in edges if pd.notna(n1)]
        network.add_edges_from(edges)
        
        
    if cols_attrs:
        network = add_edge_attributes(df, network, col_label, cols_attrs)
        

    return network


def multi_to_di(network:nx.MultiDiGraph):
    edges = defaultdict(lambda: 0)
    for e1, e2, _ in network.edges:
        edges[(e1,e2)] += 1
    edges = dict(edges)
    edges = [(k, v, w) for (k, v), w in edges.items()]
    G = nx.DiGraph()
    G.add_weighted_edges_from(edges)
    return G

def build_twitter_network(df, col_node, cols_from, cols_to, network = None, 
                          col_attrs = None):
    """
    

    Parameters
    ----------
    df : pd.DataFrame
        data source.
    col_node : str
        name of column indicating node.
    cols_from : str
        name of the column indicating connections from the node.
    cols_to : str
        name of the column indicating connections to the node.
    network : nx.Graph class
        existing networkx graph. The default is None.

    Returns
    -------
    network : nx.Graph class
        networkx graph 

    """
    
    #init network if not provided:
    if network is None:
        network = nx.DiGraph()
        
    #add nodes:
    nodes = df[col_node].unique().tolist()
    network.add_nodes_from([node for node in nodes if node not in network.nodes()])
    
    #get edges with weights:
    edges = []
    
    #for all n1 -> n2 variables:
    for var in cols_from:
        new_edges = list(zip(df[col_node], df[var]))
        _ = [edges.append((n1,n2,w)) for (n1, n2), w in pd.value_counts(new_edges).items() if pd.notna(n2)]

    #for all n2 -> n1 variables:
    for var in cols_to:
        new_edges = list(zip(df[var], df[col_node]))
        _ = [edges.append((n1,n2,w)) for (n1, n2), w in pd.value_counts(new_edges).items() if pd.notna(n1)]
        
    for edge in edges:
        if edge[:2] in network.edges:
            network[edge[0]][edge[1]]['weight'] += edge[2]
        else:
            network.add_edge(edge[0], edge[1], weight = edge[2])
            
    
    return network





#test:
if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from twitter_tools.utils import check_errors
    data = pd.read_csv('/home/piotr/projects/twitter/data/scraped/gov_tweets_new.csv', index_col = 0, 
                       dtype = str, nrows = 100000)
    data = check_errors(data)
    
    cols_from = ["in_reply_to_user_id_str"]
    cols_to = ["retweeted_status-user-id_str", "quoted_status-user-id_str"]
    
    G = build_twitter_network_multi(data, col_node = 'user-id_str', 
                                    col_label = 'id_str', cols_from = cols_from, 
                                    cols_to = cols_to, cols_attrs = ['created_at','entities-hashtags'])
    Gp = multi_to_di(G)