"""
Use subspace learning method to conduct sub-linear scaling encoding
"""
import numpy as np
import os
import glob
from feature_selection import learn_feature_weights, apply_weights
from utils import  gen_labels
from evaluate import Evaluate
from time import time
import scipy.io as sio
import pickle
from sklearn import preprocessing

# Change this to select different LinearSVM library
from liblinear.liblinearutil import *
#from sklearn.svm import LinearSVC
#from thundersvm.thundersvmScikit import SVC as thunderSVC



#====================================ENCODING==================================================
def SLEncoding(X, periods, dimension_split = None, f_select_r = 2/3, s_feature=10):
    """
        :type X ndarray(N,d)
        :param X: Training data in the form of a np matrix with N data points in d dimension
        :param periods: the periods for the templates
        :param dimension_split: Similar to product quantization, if this matrix is supplied, each classifier is train on a subset of features
        :param f_select_r : Ratio between the number of features selected to train SVM w.r.t original dimension
        :param s_feaure: s parameter such that ||w||_1 
        :return classifiers_params, contains the parameters to encode the dataset

    """
    N = X.shape[0]
    d = X.shape[1]
    n_classifiers = len(periods)    
    all_labels = gen_labels(N, periods)
    # Now, all the labels have been generated, start to conduct quantization
    classifiers_params = []
    X_train = X

    for classifier_idx in range(n_classifiers):
        """
            Iterate through each classifier, learn a clustering based on the fixed labels
        """
        print '-------Training classifier', classifier_idx                
        # If dimension split is provided, extract the features as training data
        # The splitting is similar to the scheme used in product quantization
        if dimension_split is not None:        
            s_col, e_col = get_column_range(d, dimension_split, n_classifiers, classifier_idx)
            X_train = X[:,s_col:e_col]

        #Now, start the training
        start_time = time()
        labels = all_labels[:, classifier_idx]            

        feature_weights = learn_feature_weights(X_train, labels, s=s_feature)                    
        # Sort the features 
        n_select = np.floor(f_select_r*d).astype(np.int32)
        sorted_idx = list(np.argsort(feature_weights)[::-1])
        select_idx = sorted_idx[0:n_select]        
        sio.savemat('selected_idx_{}.mat'.format(classifier_idx), {'selected_idx':select_idx})                
        Xp = X_train[:, select_idx]   
        print ('++++++++++Finish learning the feature weights, learning time = %.3f'%(time() - start_time))     
        
            
        start_time = time()
        svm_prob = problem(labels, Xp)
        clf = train(svm_prob, '-s 0 -c 4 -B 1 -n 8')
        save_model('svm_model_{}.pickle'.format(classifier_idx), clf)

        classifiers_params += [(select_idx, clf)]
        print('liblinear time =  %.5f'%(time()-start_time))
        #classifiers_params += [(centroids, V, m, clf)]
               
    return classifiers_params


#====================================DECODING==================================================
def SLDecode(X, periods, classifiers_params, dimension_split = None):
    """
        From the classifiers_params, decode the labels for the data set X
    """
    N = X.shape[0]
    d = X.shape[1]
    n_classifiers = len(classifiers_params)
    
    #all_labels = gen_labels(N, n_classifiers)    
    all_cls_labels = np.zeros(shape=(N, n_classifiers), dtype = 'int')    

    X_test = X
    # Decode for each classifier
    for j in range(n_classifiers):        
        if dimension_split is not None:        
            s_col, e_col = get_column_range(d, dimension_split, n_classifiers, j)
            X_test = X[:,s_col:e_col]
        
        param = classifiers_params[j]
        centroids = param[0]
        #V = param[1]
        #m = param[2]
        select_idx = param[0]        
        clf = param[1]
        #_ ,Xp , _ = projection_to_subspace(X_test, centroids, V, m)            
        #Xp = apply_weights(X_test*1e3,feature_weights)        
        Xp = X_test[:, select_idx]
        #cls_labels = clf.predict(Xp)        
        print ('=============NOTE TO CONSIDER ADDING -b 1=======')
        cls_labels, _, _ = predict([], Xp, clf)        
        all_cls_labels[:, j] = cls_labels
    

    # Compute the labels based on the decoded labels from individual classifiers:    
    decoded_labels = np.zeros(shape=(N,),dtype=np.int32)
    for i in range(N):
        decoded_labels[i] = label_from_classifiers(all_cls_labels[i,:], periods)
    
    return decoded_labels


def get_column_range(dimension, dimension_split, n_classifiers, classifier_idx):
    s_col = 0  # Starting column to extract
    e_col = dimension-1 # Ending column to extract
    if classifier_idx > 0:
        s_col = dimension_split[classifier_idx-1]            
    if classifier_idx < n_classifiers-1:
        e_col = dimension_split[classifier_idx]-1
    return s_col, e_col
           


def label_from_classifiers(classifiers_labels, periods):
    
    n_classifiers = len(periods)    
    label_multiplier = [1]*n_classifiers    
    j = n_classifiers - 2

    while j>=0:
        label_multiplier[j] = label_multiplier[j+1]*periods[j+1]
        j -= 1 
    
    label = 0
    for i in range(n_classifiers):
        label += classifiers_labels[i]*label_multiplier[i]

    return label


if __name__ == '__main__': 
    periods = [3, 4, 5]
    N = 60
    d = 1024
    X = np.random.rand(N,d)
    params = SLEncoding(X, periods)
    decoded_labels = SLDecode(X, params)


    # Test  label_from_classifier
    # labels = gen_labels(N, periods)
    # for i in range(labels.shape[0]):
    #     print label_from_classifiers(labels[i,:], periods)



    # Test subspace projection
    #subspace_kmeans_single(X, 2)





        




