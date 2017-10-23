from __future__ import division  # floating point division
import numpy as np
import math

import utilities as utils

class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects w from a Gaussian distribution
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        """ Reset learner """
        self.weights = None
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        self.weights = None
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.weights = np.random.rand(Xtrain.shape[1])

    def predict(self, Xtest):
        """ Most regressors return a dot product for the prediction """
        ytest = np.dot(Xtest, self.weights)
        return ytest

class RangePredictor(Regressor):
    """
    Random predictor randomly selects value between max and min in training set.
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.min = 0
        self.max = 1
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        return ytest

class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """
    def __init__( self, parameters={} ):
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.mean = None
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.mean = np.mean(ytrain)

    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean


class FSLinearRegression(Regressor):
    """
    Linear Regression with feature selection(FS), and ridge regularization
    Main linear Regression class
    """
    def __init__( self, parameters={} ):
        self.params = {'features': [1,2,3,4,5]} # subselected features
        self.reset(parameters)

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
        # print (Xless)

        # Sigular Value Decomposition for Xless, s is the singular values
        U, s, V = np.linalg.svd(Xless, full_matrices=True)
        print(s)
        # If the sigular value is 0 at the diagonal, force of weight of corresponding feature to be 0
        # for i in range(len(s)):
        #     if s[i] == 0:
        #         self.weights[i] = 0
        #     else:
        #         self.weights[i] = np.dot(np.dot(np.linalg.inv(np.dot(Xless[i].T,Xless[i])/numsamples), Xless[i].T),ytrain)/numsamples

        # self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)/numsamples), Xless.T),ytrain)/numsamples # simple explicitly slove for w
        print (self.weights)
        # presudoinverse pinv
        self.weights = np.dot(np.dot(np.linalg.pinv(np.dot(Xless.T,Xless)/numsamples), Xless.T),ytrain)/numsamples # simple explicitly slove for w

    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

class RidgeLinearRegression(Regressor):
    """
    Linear Regression with ridge regularization (l2 regularization)
    TODO: currently not implemented, you must implement this method
    Stub is here to make this more clear
    Below you will also need to implement other classes for the other algorithms
    """
    def __init__( self, parameters={} ):
        self.params = {'features': [1,2,3,4,5]}

        # Default parameters, any of which can be overwritten by values passed to params
        # self.params = {'regwgt': 0.5}
        self.reset(parameters) 

    def learn(self, Xtrain, ytrain):
        '''Learns using the traindate'''
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0] # shape[0] calculates the number of rows, shape[1] calculates the number of columns
        Xless = Xtrain[:,self.params['features']]

        # For Ridge Regression, add a ridger regularizer
        # lambda = 0.01
        self.weights = np.dot(np.dot(np.linalg.pinv(np.dot(Xless.T,Xless)/numsamples), Xless.T),ytrain)/numsamples # simple explicitly slove for w

    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xless, self.weights) + 0.01*np.dot(self.weights,self.weights)
        return ytest
    
