# refer from test_oxford.py
# Test with clustering
from __future__ import division
import numpy as np
from sklearn import preprocessing
from Encoding import test_encoding
from evaluate import Evaluate
from pca import PCA_transform
from utils import *
from time import time
import scipy.io as sio
import pdb
import argparse

def run_sl_encoding(train_file, n_periods, 
                    test_file = None,              
                    sl_f_select = 0.65,
                    sl_s_feature = 100):

    # Load training data and testing data
    trainData = load_pickle(train_file)
    if test_file is None:
        testData = trainData
    else:
        testData = load_pickle(test_file)
    
    # Optional - Perform PCA to reduce the dimensionality of the data
    # Note that the more reductions result in lower precision
    #dim_reduction = 1000
    #trainData = PCA_transform(trainData, dim_reduction)
    #testData = PCA_transform(testData, dim_reduction)

    N = 200 # Number of data points to load
    train_input = trainData[0:N, :]
    test_input = testData[0:N, :]

    # Automatically find the pattern periods.
    # This is to support the co-primes requirement of the method proposed by Yu et al.
    # For our method, co-prime is not required, this can also be set manually.
    periods = find_periods(N, n_patterns) 

    precision, compresion_rate = test_encoding( train_input, 
                                  test_input, 
                                  periods,                                               
                                  dimension_split=None,
                                  f_select_r=sl_f_select,
                                  s_feature=sl_s_feature,
                                  tollerance_list=[1,5,10,20,25])


    print('Precision = {}; Compression Ratio = {}'.format(precision, compresion_rate))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
       
    parser.add_argument('--inputFile', '-i', type = str, default='./data/brisbane/train/all/lost_features.pickle' )
    parser.add_argument('--testFile', '-t', type = str, default='./data/brisbane/test/all/lost_features.pickle' )
    parser.add_argument('--nPatterns','-n', type=int, default = 2)

    parser = vars(parser.parse_args())
    n_patterns = parser['nPatterns']
    input_file = parser['inputFile']
    test_file = parser['testFile']    

    run_sl_encoding(input_file, n_patterns, test_file)

    

    



