from __future__ import division  # floating point division
import numpy as np
import math

import utilities as utils
import script_regression as script

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

        # Sigular Value Decomposition for Xless.T * Xless, s is the singular values
        # U, s, V = np.linalg.svd(np.dot(Xless.T, Xless)/numsamples, full_matrices=False)
        # print(np.diag(s))
        # d = len(s)
        # S = np.diag(s)
        # print (np.linalg.inv(S))
        # print(np.linalg.inv(S).shape[0],np.linalg.inv(S).shape[1],U.T.shape[0],U.T.shape[1],V.shape[0],V.shape[1])

        # If the sigular value is 0 at the diagonal, force of weight of corresponding feature to be 0
        # self.weights = np.dot(np.dot(V, np.dot(np.linalg.inv(S), U.T)), ytrain)/numsamples # simple explicitly slove for w
        # self.weights = np.dot(np.dot(np.dot(V.T, np.dot(np.linalg.inv(S), U.T)), Xless.T), ytrain)/numsamples
        self.weights = np.dot(np.dot(np.linalg.pinv(np.dot(Xless.T,Xless)/numsamples), Xless.T),ytrain)/numsamples # simple explicitly slove for w
        # self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)/numsamples), Xless.T),ytrain)/numsamples # simple explicitly slove for w
        # print (self.weights)
        # presudoinverse pinv
        # self.weights = np.dot(np.dot(np.linalg.pinv(np.dot(Xless.T,Xless)/numsamples), Xless.T),ytrain)/numsamples # simple explicitly slove for w

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
        # self.params = {'features': [1,2,3,4,5]}

        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'regwgt': 0.01}
        self.reset(parameters) 

    def learn(self, Xtrain, ytrain):
        '''Learns using the traindate'''
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0] # shape[0] calculates the number of rows, shape[1] calculates the number of columns
        # Xless = Xtrain[:,self.params['features']]
        d = Xtrain.shape[1]
        I = np.identity(d)
        # print (I.shape, Xless.shape)
        # For Ridge Regression, add a ridger regularizer
        # lambda = 0.01
        # no need to worry about singular value since we are adding lambda*I term which is always greater than 0
        self.weights = np.dot(np.dot(np.linalg.inv(np.add(np.dot(Xtrain.T, Xtrain)/numsamples, 0.01*I)), Xtrain.T), ytrain)/numsamples # simple explicitly slove for w

    def predict(self, Xtest):
        # Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xtest, self.weights)
        return ytest
    
""" Question2 d) """
class LassoRegression(Regressor):
    """
    Linear Regression with feature selection(FS), and ridge regularization
    Main linear Regression class
    """
    def __init__( self, parameters={} ):
        # self.params = {'features': [1,2,3,4,5]} # subselected features
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def prox(self, weight, stepsize):
        # print(weight)
        # print(weight[1])
        for i in range (weight.shape[0]):
            if self.weights[i] > 0.01*stepsize:
                self.weights[i] = self.weights[i] - 0.01*stepsize
                # print (self.weights[i])
            elif np.absolute(self.weights[i]) <= 0.01*stepsize:
                self.weights[i] = 0
                # print (self.weights[i])
            elif self.weights[i] < -0.01*stepsize:
                self.weights[i] = self.weights[i] + 0.01*stepsize
                # print (self.weights[i])

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        # Xless = Xtrain[:,self.params['features']]

        # U, s, V = np.linalg.svd(np.dot(Xless.T, Xless)/numsamples, full_matrices=False)
        # S = np.diag(s)
        # self.weights = np.dot(np.dot(V, np.dot(np.linalg.inv(S), U.T)), ytrain)/numsamples # simple explicitly slove for w
        self.weights = np.dot(np.dot(np.linalg.pinv(np.dot(Xtrain.T, Xtrain)/numsamples), Xtrain.T),ytrain)/numsamples
        # print (self.weights)
        count = 0
        d = Xtrain.shape[1]
        w = np.zeros([d,1])
        # print (w.shape)
        err = float('inf')
        tolerance = 10e-5
        XX = np.dot(Xtrain.T, Xtrain)/numsamples
        Xy = np.dot(Xtrain.T, ytrain)/numsamples
        # print (XX.shape, Xy.shape)
        sum = 0
        for i in range (Xtrain.shape[0]):
            for j in range (Xtrain.shape[1]):
                sum = np.square(Xtrain[i][j]) + sum
        stepsize = 1/(2*np.sqrt(sum))
        # print(stepsize, stepsize.shape)
        # size = [numsamples, numsamples]
        # temp = np.zeros(size)
        # temp2 = np.diag(self.weights)
        # temp[:temp2.shape[0], :temp2.shape[1]] = temp2
        # objective
        # obj = np.add(np.dot((np.subtract(np.dot(Xless, self.weights), ytrain)).T, (np.subtract(np.dot(Xless, self.weights), ytrain)))/numsamples, temp)
        # gradient = np.dot(Xless.T, np.subtract(np.dot(Xless, self.weights), ytrain))/numsamples
        # print (np.dot((np.subtract(np.dot(Xless, self.weights), ytrain)).T, (np.subtract(np.dot(Xless, self.weights), ytrain)))/numsamples)
        # print(Xtrain.shape, self.weights.shape)
        c_w = script.geterror(np.dot(Xtrain, self.weights), ytrain)
        while np.absolute(c_w-err) > tolerance:
            err = c_w
            var = np.add(np.subtract(self.weights, stepsize*np.dot(XX, self.weights)), stepsize*Xy)
            # print (var)
            self.prox(var, stepsize)
            c_w = script.geterror(np.dot(Xtrain, self.weights), ytrain)
            # obj = np.add(np.square(np.subtract(np.dot(Xless, self.weights), ytrain))/numsamples, temp)
            # gradient = np.dot(Xless.T, np.subtract(np.dot(Xless, self.weights), ytrain))/numsamples
            # count = count + 1
        # print (self.weights)

    def predict(self, Xtest):
        # Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xtest, self.weights)
        return ytest


""" Question2 e) """
class SGD(Regressor):

    def __init__( self, parameters={} ):
        self.params = {'features': [1,2,3,4,5]} # subselected features
        self.reset(parameters)

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]

        U, s, V = np.linalg.svd(Xless, full_matrices=False)
        S = np.diag(s)
        self.weights = np.dot(np.dot(np.dot(V.T, np.dot(np.linalg.inv(S), U.T)), Xless.T), ytrain)/numsamples

        t = 1
        epochs = 1000
        stepsize = 0.01

        for i in range (epochs):
            # shuffle data points from 1, ..., numbersamples
            arr = np.arange(numsamples)
            np.random.shuffle(arr)
            for j in arr:
                gradient = np.dot(np.subtract(np.dot(Xless[j].T, self.weights), ytrain[j]), Xless[j])
                self.weights = np.subtract(self.weights, 0.01*gradient)

    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

""" Question2 f) """
class bachGD(Regressor):

    def __init__( self, parameters={} ):
        self.params = {'features': [1,2,3,4,5]} # subselected features
        self.reset(parameters)

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]

        U, s, V = np.linalg.svd(np.dot(Xless.T, Xless)/numsamples, full_matrices=False)
        S = np.diag(s)
        self.weights = np.dot(np.dot(np.dot(V.T, np.dot(np.linalg.inv(S), U.T)), Xless.T), ytrain)/numsamples

        count = 0
        epochs = 1000
        d = Xless.shape[1]
        w = np.zeros([d,1])
        # print (w.shape)
        err = float('inf')
        tolerance = 10e-5
        stepsize = 0.01
        size = [numsamples, numsamples]
        temp = np.zeros(size)
        temp2 = np.diag(self.weights)
        temp[:temp2.shape[0], :temp2.shape[1]] = temp2
        # objective
        obj = np.add(np.dot((np.subtract(np.dot(Xless, self.weights), ytrain)).T, (np.subtract(np.dot(Xless, self.weights), ytrain)))/numsamples, temp)
        gradient = np.dot(Xless.T, np.subtract(np.dot(Xless, self.weights), ytrain))/numsamples
        # print (np.dot((np.subtract(np.dot(Xless, self.weights), ytrain)).T, (np.subtract(np.dot(Xless, self.weights), ytrain)))/numsamples)
        while np.absolute(obj-err).all() > tolerance:
            stepsize = 1/(count + 1)
            err = obj
            self.weights = np.subtract(self.weights, stepsize*gradient)
            obj = np.add(np.square(np.subtract(np.dot(Xless, self.weights), ytrain))/numsamples, temp)
            gradient = np.dot(Xless.T, np.subtract(np.dot(Xless, self.weights), ytrain))/numsamples
            count = count + 1
        print (count)

    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest    