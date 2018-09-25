# Evaluate the accuracy of the method
from __future__ import division
import numpy as np
import math

def precision_evaluate(decoded_labels, tollerance_list=[1]):    
    
    N = len(decoded_labels)    
    decoded_gt = range(N)

    precision_list = []
    for tollerance in tollerance_list:
        correct = 0
        for i in range(len(decoded_labels)):
            if abs(decoded_labels[i] - decoded_gt[i]) <= tollerance:
                correct +=1
        precision_list += [correct/N]    
    print precision_list
    return precision_list


def Evaluate(Y):
    """
    P : Prediction
    Currently, this is converted from previous matlab code
    """
    n = Y.shape[0]
    recall = 0
    precision = 0
    auc = 0
    y = 1
    u = 1/n
    for i in range(n):
        if Y[i] < n:
            recall += 1
        if Y[i] == i:
            precision += 1
        else:
            y = y - u
        auc = auc + u*y
    
    precision = precision/n
    recall = recall/n
    
    return precision, recall, auc


def dataset_evaluation(sequence_decoded_labels, 
                       sequence_gt, 
                       decoded_labels,                       
                       file_sizes,
                       tollerance_list = [1]):
    
    decoded_gt = []
    for fz in file_sizes:
        decoded_gt += range(fz)
        
    correct_list = []

    for tollerance in tollerance_list:
        correct = 0
        for i in range(len(decoded_labels)):
            if sequence_decoded_labels[i] == sequence_gt[i]:
                if abs(decoded_labels[i] - decoded_gt[i]) <= tollerance:
                    correct += 1          
        correct_list += [correct/len(decoded_labels)]     
    
    return correct_list


def evaluate_with_labels(Y, gt_labels):
    """
    P : Prediction
    Currently, this is converted from previous matlab code
    """
    Y = np.asarray(Y, dtype='int')
    n = Y.shape[0]
    recall = 0
    precision = 0
    auc = 0
    y = 1
    u = 1/n
    for i in range(n):
        if Y[i] < n:
            recall += 1
        if Y[i] == gt_labels[i]:
            precision += 1
        else:
            y = y - u
        auc = auc + u*y
    
    precision = precision/n
    recall = recall/n
    
    return precision, recall, auc


