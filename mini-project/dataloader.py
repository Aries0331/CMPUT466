from __future__ import division  # floating point division
import math
import numpy as np

####### Main load functions

def load_dataset(trainsize=200, testsize=317):
    """ A physics classification dataset """
    filename = 'datasets/firedataset.csv'
    dataset = loadcsv(filename)
    trainset, testset = splitdataset(dataset,trainsize, testsize)    
    return trainset,testset

###### feature selection setup
def load_STFWI(trainsize=200, testsize=317):
    """ A physics classification dataset """
    filename = 'datasets/firedataset_STFWI.csv'
    dataset = loadcsv(filename)
    trainset, testset = splitdataset(dataset,trainsize, testsize)    
    return trainset,testset

def load_STM(trainsize=200, testsize=317):
    """ A physics classification dataset """
    filename = 'datasets/firedataset_STM.csv'
    dataset = loadcsv(filename)
    trainset, testset = splitdataset(dataset,trainsize, testsize)    
    return trainset,testset

def load_FWI(trainsize=200, testsize=317):
    """ A physics classification dataset """
    filename = 'datasets/firedataset_FWI.csv'
    dataset = loadcsv(filename)
    trainset, testset = splitdataset(dataset,trainsize, testsize)    
    return trainset,testset

def load_M(trainsize=200, testsize=317):
    """ A physics classification dataset """
    filename = 'datasets/firedataset_M.csv'
    dataset = loadcsv(filename)
    trainset, testset = splitdataset(dataset,trainsize, testsize)    
    return trainset,testset 

####### Helper functions

def loadcsv(filename):
    dataset = np.genfromtxt(filename, delimiter=',')
    return dataset

def splitdataset(dataset, trainsize, testsize, testdataset=None, featureoffset=None, outputfirst=None):
    """
    Splits the dataset into a train and test split
    If there is a separate testfile, it can be specified in testfile
    If a subset of features is desired, this can be specifed with featureinds; defaults to all
    Assumes output variable is the last variable
    """
    # Generate random indices without replacement, to make train and test sets disjoint
    randindices = np.random.choice(dataset.shape[0],trainsize+testsize, replace=False)
    featureend = dataset.shape[1]-1
    outputlocation = featureend    
    if featureoffset is None:
        featureoffset = 0
    if outputfirst is not None:
        featureoffset = featureoffset + 1
        featureend = featureend + 1
        outputlocation = 0
    
    Xtrain = dataset[randindices[0:trainsize],featureoffset:featureend]
    ytrain = dataset[randindices[0:trainsize],outputlocation]
    Xtest = dataset[randindices[trainsize:trainsize+testsize],featureoffset:featureend]
    ytest = dataset[randindices[trainsize:trainsize+testsize],outputlocation]

    if testdataset is not None:
        Xtest = dataset[:,featureoffset:featureend]
        ytest = dataset[:,outputlocation]        

    # Normalize features, with maximum value in training set
    # as realistically, this would be the only possibility    
    for ii in range(Xtrain.shape[1]):
        maxval = np.max(np.abs(Xtrain[:,ii]))
        if maxval > 0:
            Xtrain[:,ii] = np.divide(Xtrain[:,ii], maxval)
            Xtest[:,ii] = np.divide(Xtest[:,ii], maxval)
                        
    # Add a column of ones; done after to avoid modifying entire dataset
    Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0],1))))
    Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0],1))))
                              
    return ((Xtrain,ytrain), (Xtest,ytest))


def create_susy_dataset(filenamein,filenameout,maxsamples=100000):
    dataset = np.genfromtxt(filenamein, delimiter=',')
    y = dataset[0:maxsamples,0]
    X = dataset[0:maxsamples,1:9]
    data = np.column_stack((X,y))
    
    np.savetxt(filenameout, data, delimiter=",")
