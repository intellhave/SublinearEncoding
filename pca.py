# Perform PCA for data
from sklearn.decomposition import PCA
#import skcuda.linalg as linalg
#from skcuda.linalg import PCA as cuPCA
from time import time


def PCA_transform(X, nComponents):
    start_time = time()
    print('Running PCA for N = %d '%X.shape[0])

    pca = PCA(n_components = nComponents)
    X = pca.fit_transform(X)
    
    print('++++++++++PCA Completed, PCA time = %.3f'%(time()-start_time))
    return X