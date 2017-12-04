from __future__ import division  # floating point division
import numpy as np
import math

import utilities as utils
import script_regression as script
import matplotlib.pyplot as plt

# Neural Network regression
from sklearn.neural_network import MLPRegressor
# Support Vector Machines
from sklearn.svm import SVR
# Random Forest regression
from sklearn.ensemble import RandomForestRegressor

class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects w from a Gaussian distribution
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        self.reset(parameters)
        self.yaxis = np.zeros(0)

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

    def data(self):
        self.yaxis = np.zeros(0)

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

class NeuralNetwork(Regressor):

    def __init__( self, parameters={} ):
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        """ Reset learner """
        self.mlp = None
        self.weights = None
        self.resetparams(parameters)
    
    def learn(self, Xtrain, ytrain):

        # print(Xtrain.shape, ytrain.shape)
        self.mlp = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='sgd', early_stopping=True)

        self.weights = self.mlp.fit(Xtrain, ytrain)

    def predict(self, Xtest):

        ytest = self.mlp.predict(Xtest)
        # print("NN")
        # print (ytest)

        return ytest

class SVM(Regressor):

    def __init__( self, parameters={} ):
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        """ Reset learner """
        self.clf = None
        self.weights = None
        self.resetparams(parameters)
    
    def learn(self, Xtrain, ytrain):

        # print(Xtrain.shape, ytrain.shape)
        self.clf = SVR(kernel='rbf')

        self.weights = self.clf.fit(Xtrain, ytrain)

    def predict(self, Xtest):

        ytest = self.clf.predict(Xtest)
        # print("SVR")
        # print (ytest)

        return ytest

class RandomForest(Regressor):

    def __init__( self, parameters={} ):
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        """ Reset learner """
        self.regr = None
        self.weights = None
        self.resetparams(parameters)
    
    def learn(self, Xtrain, ytrain):

        # print(Xtrain.shape, ytrain.shape)
        self.regr = RandomForestRegressor(max_depth=2, random_state=0)

        self.weights = self.regr.fit(Xtrain, ytrain)

    def predict(self, Xtest):

        ytest = self.regr.predict(Xtest)
        # print("SVR")
        # print (ytest)

        return ytest

