# -*- coding: utf-8 -*-
"""
Util file includes utility functions
"""
from os import listdir
from os.path import isfile, join
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
import math
import sys
import time

def create_graph(adjacency_path=None):
    AM = np.loadtxt(adjacency_path)
    N = AM.shape[0]  
    g = nx.Graph()        
    g.add_nodes_from(range(N), label = "")
    g.graph['node_order'] = range(N)

    list_edges = []
    for u in range(N-1):
        for v in range(u+1,N):
            w = AM[u,v]
            if w != 0.0:
                list_edges.append((u, v))
    g.add_edges_from(list_edges, label="")
    return g  

def node_labeling(g=None, label_path=None): 
    
    list_labels = load_list_from_file(label_path)
    
    dict_node_label = {}
    for idx in range(len(g.nodes())):
        dict_node_label[idx] = list_labels[idx]
        
    #nx.set_node_attributes(g,'label', dict_node_label)
    nx.set_node_attributes(g,name = 'label', values = dict_node_label)
    
def list_files_in_folder(folder_path):    
    """
    Return: A list of the file names in the folder
    """
          
    list = listdir(folder_path)
    onlyfiles = [ f for f in list  if isfile(join(folder_path,f)) ]
    return onlyfiles 

def load_list_from_file(file_path):
    """
    Return: A list saved in a file
    """
    
    f = open(file_path,'r')
    listlines = [line.rstrip() for line in f.readlines()]
    f.close()
    return listlines

def hops_neighbors(graph=None, source=None, n_hop=None):
    hop_step = []
    temp_list_nodes = [source]
    hop_step.append(temp_list_nodes)
    
    accumulative_set = set([source])
    for hop in range(1, n_hop+1):
        current_nodes = []
        for u in temp_list_nodes:
            neighbors = graph.neighbors(u)
            current_nodes.extend(neighbors)
        hop_step.append(list(set(current_nodes) - accumulative_set))
        accumulative_set = accumulative_set.union(set(current_nodes))
        temp_list_nodes = current_nodes
    return hop_step

def get_dict_hops_neighbors(graph=None, n_hop=None):
    dict_hops_neighbors = {}    
    for n in range(len(graph.nodes())):
        dict_hops_neighbors[n] = hops_neighbors(graph=graph, source=n, n_hop=n_hop)
    return dict_hops_neighbors

def get_dict_hops_neighbors_save(graph=None, n_hop=None):
    dict_hops_neighbors = {}    
    for n in range(len(graph.nodes())):
        list_hop_neighbors = hops_neighbors(graph=graph, source=n, n_hop=n_hop)
        nodes = []
        for hop_id in range(1, n_hop):
            nodes.extend(list_hop_neighbors[hop_id])
        dict_hops_neighbors[n] = nodes
    return dict_hops_neighbors
    
def deepwl(graph=None, feature_matrix=None, n_hop=None):
    
    """ separately compute between hops """
    N = len(graph.nodes())
    G = np.zeros((N, N)) 
    dict_hops_neighbors = get_dict_hops_neighbors(graph=graph, n_hop=n_hop)
   
    # Loop over every node couple
    for u in range(N):
        print u
        sys.stdout.flush()        
        u_vec = feature_matrix[u,:]
        for v in range(u, N):
            v_vec = feature_matrix[v,:]
            dot_value_check = u_vec.multiply(v_vec).sum()
            
            if dot_value_check != 0:
                dot_value_sum = 0
                for hop_id in range(1, n_hop):
                    u_vec_sum = csr_matrix(feature_matrix[dict_hops_neighbors[u][hop_id],:].sum(axis=0))
                    v_vec_sum = csr_matrix(feature_matrix[dict_hops_neighbors[v][hop_id],:].sum(axis=0))
                    dot_value_sum = u_vec_sum.multiply(v_vec_sum).sum()
                    dot_value_sum+= dot_value_sum
                                   
                G[u, v] = G[u,v] = dot_value_check*dot_value_sum                
    # Normalize kernel matrix G   
    print "Starting normalizing"
    sys.stdout.flush()
    for idx1 in range(N):
        for idx2 in range(idx1,N):
            if G[idx1,idx2] !=0 and G[idx2,idx1] !=0:
                G[idx1,idx2] = G[idx2,idx1] = G[idx1,idx2]/math.sqrt(G[idx1,idx1]*G[idx2,idx2])
   
    for idx in range(N):
        G[idx, idx] = 1.0
                
    return G                    

def extract_submatrix(row_indices, col_indices, A):
    """ Extracting a submatrix from  matrix A
    
    Parameter:
    row_indices: row index list that we want to extract
    col_indices: Column index list that we want to extract
    A: Matrix
    
    Return:
    submatrix of A
    """

    len_row = len(row_indices)
    len_col = len(col_indices)
    
    M = np.zeros((len_row,len_col))
    for order1, idx_row in enumerate(row_indices):
        for order2, idx_col in enumerate(col_indices):
            M[order1,order2] = A[idx_row,idx_col]
    
    return M