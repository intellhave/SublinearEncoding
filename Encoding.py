"""
    Interface for the encoders    
"""
from __future__ import division
import numpy as np
from sl_encoding import SLEncoding, SLDecode
from evaluate import precision_evaluate
from random import randint
from time import time
import pdb
   
def test_encoding(train_data, test_data, periods, 
                  method=None, 
                  dimension_split = None,
                  f_select_r = 0.65,
                  s_feature = 10,
                  tollerance_list = [10]):

    N_train = train_data.shape[0]
    N_test = test_data.shape[0]
    d = train_data.shape[1]    
    start_time = time()
    # Encode with train data
    encode_params = SLEncoding(train_data, periods, 
                                dimension_split=dimension_split, 
                                f_select_r=f_select_r,
                                s_feature = s_feature)
    encode_time = (time() - start_time)
    print ('++++++++++Encoding Time = %.3f' % (time() - start_time))
    
    #Decode        
    start_time = time()
    Y = SLDecode(test_data, periods, encode_params, dimension_split=dimension_split)
    decode_time = time() - start_time
    print('++++++++++Query time = %.3f' % (time() - start_time))

    # Compute Compress Ratio
    n_classifier = len(periods)        
    #Computing storage sizes (in byes)
    bits_per_number = 64
    storage_size = 0
    for i in range(n_classifier):
        param = encode_params[i]        
        select_idx = param[0]
        select_idx_storage = d/8        # Number of byes needed to store d selection bits
        clf_storage = periods[i]*len(select_idx)*(bits_per_number/8) # SVM classifiers need to store periods[i] hyperplanes, each of dimension select_idx       
        storage_size += clf_storage + select_idx_storage

    # Converge to Megabytes
    storage_size = storage_size/1024
    compress_ratio = storage_size/(N_train*d)
        
    # Compute accuracy
    precision = precision_evaluate(Y, tollerance_list=tollerance_list)
    return precision, compress_ratio
