from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np

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
    trainsize = 1000
    testsize = 5000
    numruns = 1

    regressionalgs = {'Random': algs.Regressor(),
                'Mean': algs.MeanPredictor(),
                'FSLinearRegression5': algs.FSLinearRegression({'features': [1,2,3,4,5]}),
                # 'LassoRegression5': algs.LassoRegression({'features': [1,2,3,4,5]}),
                # 'SGD5': algs.SGD({'features': [1,2,3,4,5]}),
                # 'FSLinearRegression50': algs.FSLinearRegression({'features': range(50)}),
                # Increase the number of selected features (up to all the features)
                # 'FSLinearRegression100': algs.FSLinearRegression({'features': range(100)}),
                # 'FSLinearRegression200': algs.FSLinearRegression({'features': range(200)}),
                'FSLinearRegression385': algs.FSLinearRegression({'features': range(385)}),
                'RidgeLinearRegression385': algs.RidgeLinearRegression({'features': range(385)}),
                # 'LassoRegression385': algs.LassoRegression({'features': range(385)}),
                # 'SGD385': algs.SGD({'features': range(385)}),
             }
    numalgs = len(regressionalgs)

    # Enable the best parameter to be selected, to enable comparison
    # between algorithms with their best parameter settings
    parameters = (
        {'regwgt': 0.0},
        {'regwgt': 0.01},
        {'regwgt': 1.0},
                      )
    numparams = len(parameters)
    
    errors = {}
    for learnername in regressionalgs:
        errors[learnername] = np.zeros((numparams,numruns))

    for r in range(numruns):
        trainset, testset = dtl.load_ctscan(trainsize,testsize)
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
                # print ("predictions:")
                # print (predictions)
                # print ("testset")
                # print (testset[1])
                error = geterror(trainset[1], predictions)
                stderr = np.std(predictions,ddof=1)
                print ('Error for ' + learnername + ': ' + str(error))
                print ('Standard error for ' + learnername + ': ' + str(stderr))
                errors[learnername][p,r] = error
                errors[learnername][p,r] = stderr


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

    # report the standard error, 
    # i.e. the sample standard deviation divided by the square root of the number of runs.
