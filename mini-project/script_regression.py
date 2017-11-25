from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np
import matplotlib.pyplot as plt

import dataloader as dtl
import regressionalgorithms as algs

def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction,ytest))

def l1err(prediction,ytest):
    """ l1 error """
    return np.linalg.norm(np.subtract(prediction,ytest),ord=1)

def l2err_squared(prediction,ytest):
    """ l2 error squared """
    return np.square(np.linalg.norm(np.subtract(prediction,ytest)))

def geterror(predictions, ytest):
    """ mean squared error """
    # Can change this to other error values
    # return (l2err_squared(predictions,ytest)/ytest.shape[0])/2
    return 0.5*l2err_squared(predictions,ytest)/ytest.shape[0]


if __name__ == '__main__':
    trainsize = 200
    testsize = 317
    numruns = 5

    regressionalgs = {'Random': algs.Regressor(),
                'Mean': algs.MeanPredictor(),
                # 'Neural Network': algs.NeuralNetwork({'epochs': 100}),
                # 'Support Vector Machines': algs.SVM(),
             }
    numalgs = len(regressionalgs)

    # Enable the best parameter to be selected, to enable comparison
    # between algorithms with their best parameter settings
    parameters = (
        {'regwgt': 0.0, 'nh': 32},
                      )
    numparams = len(parameters)

    errors = {}
    # initialize for x and y axis
    x = {}
    y = {}

    for learnername in regressionalgs:
        errors[learnername] = np.zeros((numparams,numruns))

    for r in range(numruns):
        trainset, testset = dtl.load_dataset(trainsize,testsize)
        print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],r))

        for p in range(numparams):
            params = parameters[p]
            for learnername, learner in regressionalgs.items():
                # Reset learner for new parameters
                learner.reset(params)
                print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
                learner.learn(trainset[0], trainset[1])
                # Test model
                predictions = learner.predict(trainset[0])  
                # get return value of errors from each regression function 
                y[learnername] = learner.data()
                error = geterror(trainset[1], predictions) # change to training error
                # stderr = np.std(predictions,ddof=1)
                print ('Training error for ' + learnername + ': ' + str(error))
                predictions_test = learner.predict(testset[0]) 
                error_test = geterror(testset[1], predictions_test)
                print ('Test error for ' + learnername + ': ' + str(error_test))
                errors[learnername][p,r] = error
                # errors[learnername][p,r] = stderr


    for learnername in regressionalgs:
        besterror = np.mean(errors[learnername][0,:])
        bestparams = 0
        for p in range(numparams):
            aveerror = np.mean(errors[learnername][p,:])
            if aveerror < besterror:
                besterror = aveerror
                bestparams = p

        # Extract best parameters
        learner.reset(parameters[bestparams])
        #print ('Best parameters for ' + learnername + ': ' + str(learner.getparams()))
        print ('Average error for ' + learnername + ': ' + str(besterror))
        print ('Standard error for ' + learnername + ': ' + str(np.std(errors[learnername][bestparams,:])/math.sqrt(numruns)))


