import numpy as np
from scipy.io import loadmat
import time

def loaddata(filename):
    """
    Returns xTr,yTr,xTe,yTe
    xTr, xTe are in the form nxd
    yTr, yTe are in the form nx1
    """
    data = loadmat(filename)
    xTr = data["xTr"]; # load in Training data
    yTr = np.round(data["yTr"]); # load in Training labels
    xTe = data["xTe"]; # load in Testing data
    yTe = np.round(data["yTe"]); # load in Testing labels
    return xTr.T,yTr.T,xTe.T,yTe.T


def l2distance(X,Z=None):
    # function D=l2distance(X,Z)
    #
    # Computes the Euclidean distance matrix.
    # Syntax:
    # D=l2distance(X,Z)
    # Input:
    # X: nxd data matrix with n vectors (columns) of dimensionality d
    # Z: mxd data matrix with m vectors (columns) of dimensionality d
    #
    # Output:
    # Matrix D of size nxm
    # D(i,j) is the Euclidean distance of X(:,i) and Z(:,j)
    #
    # call with only one input:
    # l2distance(X)=l2distance(X,X)
    #
    # Remember that the Euclidean distance can be expressed as: x^2-2x*z+z^2

    if Z is None:
        Z = X
    n,d = X.shape
    m = Z.shape[0]
    S = np.dot(np.diag(np.dot(X,X.T)).reshape(n,1),np.ones((1, m),dtype=int))
    R = np.dot(np.ones((n, 1),dtype=int),np.diag(np.dot(Z,Z.T)).reshape(1,m))
    G = np.dot(X,Z.T)
    D2 = S-2*G+R
    D = np.sqrt(D2)
    return D


def l1distance(X,Z=None):
    # function D=l1distance(X,Z)
    #
    # Computes the Manhattan distance matrix.
    # Syntax:
    # D=l1distance(X,Z)
    # Input:
    # X: nxd data matrix with n vectors (columns) of dimensionality d
    # Z: mxd data matrix with m vectors (columns) of dimensionality d
    #
    # Output:
    # Matrix D of size nxm
    # D(i,j) is the Manhattan distance of X(:,i) and Z(:,j)
    #
    # call with only one input:
    # l2distance(X)=l2distance(X,X)
    if Z is None:
        Z = X
    n,d = X.shape
    m = Z.shape[0]
    t1 = np.repeat(X.reshape(1,n,d),m,axis=0)
    t2 = np.repeat(Z,n,axis=0).reshape(m,n,d)
    D = np.sum(abs(t1-t2),axis=2)
    return D

def findknn(xTr,xTe,k,d_type = 'l2'):
    #Find the k nearest training examples and their distances.
    # returns two matrices both of shape KxN_t, where N_t is the number of
    #test examples
    #d_type: distance type can only be l2 or l1 distance (Manhattan and Euclidean)
    # xTr: training data of shape NxD
    # K: K nearest
    
    if d_type =='l2': 
        DD = l2distance(xTr,xTe).T
    elif d_type == 'l1': 
        DD = l1distance(xTr,xTe)
    else: raise ValueError('Only supports Euclidean and Manhattan distances')
    indices = np.argsort(DD,axis=1)[:,0:k].T
    dists = np.sort(DD,axis=1)[:,0:k].T
    return indices, dists 

def knnclassifier(xTr,yTr,xTe,k,d_type='l2'):
    from scipy import stats
    I,D = findknn(xTr,xTe,k,d_type)
    preds = np.array(stats.mode(yTr[I]).mode).flatten()
    return preds

#def knnclassifier(xTr,yTr,xTe,k,d_type='l2'):
#    #Function that memorizes the training examples and predict the labels  
#    #in the test set using the k-nearst examples in the training data
#    #returns the predicted labels
#    # xTr: training data of shape NxD
#    # yTr: training labels
#    # K: K nearest
#    # xTe: test data of shape N_txD
#    #d_type: distance type (l1 or l2)
#    
#    from scipy import stats
#    I,D = findknn(xTr,xTe,k,d_type = d_type)
#    preds1 = np.array(stats.mode(yTr[I]).mode).flatten()
#    preds2 = np.array(stats.mode(yTr[I]*-1).mode*-1).flatten()
#    i = np.array(preds2==preds1)
#    if ((np.sum(preds2==preds1)==preds1.shape) or (k-1==0)):
#        return preds1
#    else:
#        k=k-1
#        I,D = findknn(xTr,xTe,k,d_type = d_type)
#        predsn = np.array(stats.mode(yTr[I]).mode).flatten()
#        preds1[~i] = predsn[~i]
#    return preds1

if __name__ == '__main__':
    print("Face Recognition: (1-nn)")
    xTr,yTr,xTe,yTe=loaddata("../Dataset/faces.mat")
    t0 = time.time()
    preds = knnclassifier(xTr,yTr,xTe,1,'l1') #l1 distance works better
    result= np.mean(yTe.flatten() == preds)
    t1 = time.time()
    print("Execution time:",t1-t0)
    print("Test accuracy: %.2f%%" %(result*100))
    
    print("Handwritten digits Recognition: (3-nn)")
    xTr,yTr,xTe,yTe=loaddata("../Dataset/digits.mat"); # load the data
    t0 = time.time()
    preds = knnclassifier(xTr,yTr,xTe,1,'l2') #l2 distance works better
    result = np.mean(yTe.flatten() == preds)
    t1 = time.time()
    print("Execution time:",t1-t0)
    print("Test accuracy: %.2f%%" %(result*100.0))

