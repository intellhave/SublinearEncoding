"""
Learn a set of weights for clustering

This implementation follows descriptions in the paper 
"A Framework for feature selection in clustering"

"""
from __future__ import division
import numpy as np
from tools import get_n_cluster
from time import time

def apply_weights(X, weights):
    """
     Returned a new dataset with  weights applied 
    """    
    Xp = np.ones(X.shape)
    for i in range(X.shape[0]):    
        Xp[i,:] = np.multiply(X[i,:], weights)
            
    return Xp


def get_wcss(X, labels):
    """
    Compute the within-cluster-sum-of-squares
    """    
    d = X.shape[1]  # Dimensionality of the data
    wcss = np.zeros(d)
    for i in np.unique(labels):
        mask = (labels == i)
        if np.sum(mask)>1:
            wcss += np.sum( np.square( X[mask, :] - np.mean(X[mask, :], axis = 0)), axis = 0)
    #bcss = np.sum( np.square( x - np.mean(x, axis = 0)), axis = 0) - wcss
    return wcss #, bcss

def soft_thresholding(x, d):
    return np.sign(x) * np.maximum(0, np.abs(x) - d)


def _binary_search(argu, sumabs):
    l2n_argu = np.linalg.norm(argu)
    if l2n_argu == 0 or np.sum(np.abs(argu / l2n_argu)) <= sumabs:
        return 0
    lam1 = 0
    lam2 = np.max(np.abs(argu)) - 1e-5
    iter = 1
    while iter <= 15 and (lam2 - lam1) > 1e-4:
        su = soft_thresholding(argu, (lam1 + lam2) / 2.)
        if np.sum(np.abs(su / np.linalg.norm(su))) < sumabs:
            lam2 = (lam1 + lam2) / 2.
        else:
            lam1 = (lam1 + lam2) / 2.
        iter += 1
    return (lam1 + lam2) / 2.


def learn_feature_weights(X, labels, s=10):
    """
    Learn a vector indicating the weights for each feature
    """
    start_time = time()

    N = X.shape[0]
    d = X.shape[1]    
    wcss = get_wcss(X, labels)
    tss = get_wcss(X, np.ones(N))
    lam = _binary_search(-wcss + tss, s)        
    ws_unscaled = soft_thresholding(-wcss+tss, lam)
    print 'lam = ',lam
    print('++++++++++Time for soft thresholding =  %.3f' %(time()-start_time))            
    return ws_unscaled/np.linalg.norm(ws_unscaled)

    


# def learn_feature_multiple_weights(X, labels, lbd = 0.8):

#     N = X.shape[0]
#     d = X.shape[1]    
#     n_clusters = get_n_cluster(labels)
#     W = np.zeros(shape=(n_clusters, d)) 
#     S = np.zeros(shape=(n_clusters, d))
#     for i in range(n_clusters):        

#         # Extract all points belongging to the cluster:
#         cluster_idx = [idx for idx, value in enumerate(labels) if value == i]
#         # Compute the vector S_i:
#         s_i = np.zeros(shape=(d,))
#         for j in cluster_idx:
#             for k in cluster_idx:                
#                 s_i += np.square(X[j,:]-X[k,:])
#         # Assign to matrix 
#         S[i,:] = s_i

#     # Now compute the matrix W
#     for i in range(n_clusters):
#         sl = np.sum(np.exp(-S[i,:]/lbd))        
#         W[i,:] = np.exp(-S[i,:]/lbd)/sl

#     return W
