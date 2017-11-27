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

def deepwl_separate(graph=None, iters_features=None, n_hop=None):
    
    """ Compute deepwl kernel which separately compute between hops """
    N = len(graph.nodes())
    n_iter = len(iters_features)

    G = np.zeros((N, N)) 
    dict_hops_neighbors = get_dict_hops_neighbors(graph=graph, n_hop=n_hop)
   
    # Loop over every node couple
    for u in range(N-1):          
        #print "node: ", u
        #sys.stdout.flush()
        for v in range(u, N):
            list_hop_neighbors_u = dict_hops_neighbors[u]
            list_hop_neighbors_v = dict_hops_neighbors[v]
            
            for iter_id in range(n_iter):
                n_features = iters_features[iter_id][0].shape[1]
                dot_value = 0
                
                for hopid in range(n_hop):
                    u_vec = csr_matrix([[0]*n_features])
                    v_vec = csr_matrix([[0]*n_features])
                    
                    for w in list_hop_neighbors_u[hopid]:
                        u_vec = u_vec + iters_features[iter_id][0][w,:]
                    
                    for w in list_hop_neighbors_v[hopid]:
                        v_vec = v_vec + iters_features[iter_id][0][w,:]
                    
                    dot_value+= u_vec.multiply(v_vec).sum()
                    
                G[u, v] += dot_value
                G[v, u] += dot_value
    # Normalize kernel matrix G   
    for idx1 in range(N-1):
        for idx2 in range(idx1,N):
            G[idx1,idx2] = G[idx2,idx1] = G[idx1,idx2]/math.sqrt(G[idx1,idx1]*G[idx2,idx2])
    
    return G
                    
def deepwl_combine(graph=None, iters_features=None, n_hop=None):
    
    """ Combine features between hops """
    
    N = len(graph.nodes())
    n_iter = len(iters_features)
    
    G = np.zeros((N, N)) 
    dict_hops_neighbors = get_dict_hops_neighbors(graph=graph, n_hop=n_hop)

    # Loop over every node couple
    for u in range(N-1):          
        #print "node: ", u
        #sys.stdout.flush()
        for v in range(u, N):
            list_hop_neighbors_u = dict_hops_neighbors[u]
            list_hop_neighbors_v = dict_hops_neighbors[v]
            
            for iter_id in range(n_iter):
                n_features = iters_features[iter_id][0].shape[1]
                dot_value = 0
                
                u_vec = csr_matrix([[0]*n_features])
                v_vec = csr_matrix([[0]*n_features])                
                
                for hopid in range(n_hop):

                    for w in list_hop_neighbors_u[hopid]:
                        u_vec = u_vec + iters_features[iter_id][0][w,:]
                    
                    for w in list_hop_neighbors_v[hopid]:
                        v_vec = v_vec + iters_features[iter_id][0][w,:]
                    
                dot_value+= u_vec.multiply(v_vec).sum()
                    
                G[u, v] += dot_value
                G[v, u] += dot_value
    # Normalize kernel matrix G   
    for idx1 in range(N-1):
        for idx2 in range(idx1,N):
            G[idx1,idx2] = G[idx2,idx1] = G[idx1,idx2]/math.sqrt(G[idx1,idx1]*G[idx2,idx2])
    
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
