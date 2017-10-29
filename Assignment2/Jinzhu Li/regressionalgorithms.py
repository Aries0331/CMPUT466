from __future__ import division  # floating point division
import numpy as np
import math

import utilities as utils
import script_regression as script
import matplotlib.pyplot as plt

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

        # use presudoinverse pinv instead to handle singular matrix
        self.weights = np.dot(np.dot(np.linalg.pinv(np.dot(Xless.T,Xless)/numsamples), Xless.T),ytrain)/numsamples # simple explicitly slove for w
        
        # self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)/numsamples), Xless.T),ytrain)/numsamples # simple explicitly slove for w
        # print (self.weights)
        # self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)/numsamples), Xless.T),ytrain)/numsamples # simple explicitly slove for w

    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

""" Question2 c) """
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
        self.params = {'regwgt': 0.0, 'regwgt': 0.01, 'regwgt': 1.0}
        self.reset(parameters) 

    def learn(self, Xtrain, ytrain):
        '''Learns using the traindate'''
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0] # shape[0] calculates the number of rows, shape[1] calculates the number of columns
        # Xless = Xtrain[:,self.params['features']]
        d = Xtrain.shape[1]
        # identity matrix
        I = np.identity(d)
        # print (I.shape, Xless.shape)
        # lambda = 0.01

        # For Ridge Regression, add a ridger regularizer
        self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Xtrain.T, Xtrain)/numsamples, self.params['regwgt']*I)), Xtrain.T), ytrain)/numsamples # simple explicitly slove for w

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
        self.params = {'regwgt': 0.0, 'regwgt': 0.01, 'regwgt': 1.0}
        self.reset(parameters)

    # proximal methods, from 6.4 in notes
    def prox(self, weight, stepsize, regwgt):
        # print(weight)
        # print(weight[1])
        # print (regwgt)
        for i in range (weight.shape[0]):
            # print (weight[i])
            if weight[i] > regwgt*stepsize:
                self.weights[i] = weight[i] - regwgt*stepsize
                # print (self.weights[i])
            elif np.absolute(weight[i]) <= regwgt*stepsize:
                self.weights[i] = 0
                # print (self.weights[i])
            elif weight[i] < -regwgt*stepsize:
                self.weights[i] = weight[i] + regwgt*stepsize
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
        # self.weights = np.dot(np.dot(np.linalg.pinv(np.dot(Xtrain.T, Xtrain)/numsamples), Xtrain.T),ytrain)/numsamples
        # print (self.weights.shape)
        count = 0
        # d = Xtrain.shape[1]
        self.weights = np.zeros([385,]) # intialize the weights of vectors of zeros
        # print (self.weights.shape)
        err = float('inf') # set error to be infinite at the beginning
        tolerance = 10e-5
        XX = np.dot(Xtrain.T, Xtrain)/numsamples
        Xy = np.dot(Xtrain.T, ytrain)/numsamples
        # print (XX.shape, Xy.shape)
        # l2 norm calculation
        sum_ = 0
        for i in range (Xtrain.shape[0]):
            for j in range (Xtrain.shape[1]):
                sum_ = np.square(Xtrain[i][j]) + sum_
        # norm = np.linalg.norm(self.weights)
        # print (norm, sum_)
        stepsize = 1/(2*np.sqrt(sum_)) # fixed stepsize
        # c(w)
        c_w = script.geterror(np.dot(Xtrain, self.weights), ytrain) 

        """
        Batch gradient descent for l1 regularized linear regression, from Algorithm 4 in notes
        """
        while np.absolute(c_w-err) > tolerance:
            err = c_w
            # print (var, self.weights)
            # The proximal operator projects back into the space of sparse solutions given by l1
            var = np.add(np.subtract(self.weights, stepsize*np.dot(XX, self.weights)), stepsize*Xy)
            self.prox(var, stepsize, self.params['regwgt'])
            # print (self.weights)
            # norm = 0
            # for i in range (self.weights.shape[0]):
                # norm = norm + np.absolute(self.weights[i])
            # calculate l1 norm
            norm = np.linalg.norm(self.weights, ord=1)
            # print (norm, norm_)
            # update C(w)
            c_w = script.geterror(np.dot(Xtrain, self.weights), ytrain) + self.params['regwgt'] * norm
            # count = count + 1
        # print (self.weights)

    def predict(self, Xtest):
        # Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xtest, self.weights)
        return ytest


""" Question2 e) """
class SGD(Regressor):

    def __init__( self, parameters={} ):
        self.params = {} # subselected features
        self.reset(parameters)
        self.numruns = 5
        self.yaxis = np.zeros(1000)

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        # Xless = Xtrain[:,self.params['features']]

        # U, s, V = np.linalg.svd(Xless, full_matrices=False)
        # S = np.diag(s)
        # self.weights = np.dot(np.dot(np.linalg.pinv(np.dot(Xtrain.T, Xtrain)/numsamples), Xtrain.T),ytrain)/numsamples
        self.weights = np.zeros([385,]) # intialize the weights of vectors of zeros

        # t = 1
        epochs = 1000
        stepsize = 0.01

        # xaxis = np.zeros(1000) 

        """
        Stochastic gradient descent, from Algorithm 3 in notes
        """
        for i in range (epochs):
            # shuffle data points from 1, ..., numbsamples
            arr = np.arange(numsamples)
            np.random.shuffle(arr)
            for j in arr:
                gradient = np.dot(np.subtract(np.dot(Xtrain[arr[j]].T, self.weights), ytrain[arr[j]]), Xtrain[arr[j]])
                # print (gradient)
                stepsize = 0.01/(1+i) # decrease stepsize to converge
                self.weights = np.subtract(self.weights, stepsize*gradient)
                # print(self.weights)

            # store the MSE for plotting
            self.yaxis[i] += script.geterror(np.dot(Xtrain, self.weights), ytrain)

        # x = np.arange(1000)
        # plt.plot(x, self.yaxis)
        # plt.show()
    def data(self):
        """ return the averaged error in y axis """
        return self.yaxis/self.numruns

    def predict(self, Xtest):
        # Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xtest, self.weights)
        return ytest

""" Question2 f) """
class batchGD(Regressor):

    def __init__( self, parameters={} ):
        self.params = {}
        self.reset(parameters)
        self.numruns = 5
        self.yaxis = np.zeros(5000)
        self.xaxis = np.arange(5000)

    """
    Line search algorithm, from Algorithm 1 in notes
    """
    def lineSearch(slef, Xtrain, ytrain, weight_t, gradient, cost):

        numsamples = Xtrain.shape[0]
        # Optimization parameters
        stepsize_max = 1.0
        t = 0.5 # stepsize reduces more quickly
        tolerance = 10e-5
        stepsize = stepsize_max
        weight = weight_t.copy() # weight_t is self.weights 
        obj = cost
        max_interation = 100
        i = 0

        # while number of backtracking iterations is less than maximum iterations 
        while i < max_interation:
            weight = weight_t - stepsize * gradient
            # Ensure improvement is at least as much as tolerance
            if (cost < obj-tolerance):
                break
            # Else, the objective is worse and so we decrease stepsize
            else:
                # print ("else")
                stepsize = t * stepsize
            cost = script.geterror(np.dot(Xtrain, weight), ytrain)
            i = i + 1
        # If maximum number of iterations reached, which means we could not improve solution
        if i == max_interation:
            # print ("i")
            stepsize = 0
            return stepsize 
        return stepsize

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        # Xless = Xtrain[:,self.params['features']]

        # U, s, V = np.linalg.svd(np.dot(Xless.T, Xless)/numsamples, full_matrices=False)
        # S = np.diag(s)
        # self.weights = np.dot(np.dot(np.linalg.pinv(np.dot(Xtrain.T, Xtrain)/numsamples), Xtrain.T),ytrain)/numsamples
        self.weights = np.zeros([385,]) # intialize the weights of vectors of zeros

        count = 0 # count the number of iterations before converge
        # print (w.shape)
        err = float('inf')
        tolerance = 10e-6
        stepsize = 0.01
   
        c_w = script.geterror(np.dot(Xtrain, self.weights), ytrain)
        # gradient = np.dot(Xless.T, np.subtract(np.dot(Xless, self.weights), ytrain))/numsamples
        # print (np.dot((np.subtract(np.dot(Xless, self.weights), ytrain)).T, (np.subtract(np.dot(Xless, self.weights), ytrain)))/numsamples)
        
        """
        Batch gradient descent, from Algorithm 2 in notes
        """
        while np.absolute(c_w-err) > tolerance:
            err = c_w
            gradient = np.dot(Xtrain.T, np.subtract(np.dot(Xtrain, self.weights), ytrain))/numsamples
            # The step-size is chosen by line-search
            stepsize = self.lineSearch(Xtrain, ytrain, self.weights, gradient, c_w)
            # print(self.weights)
            self.weights = self.weights - stepsize*gradient
            c_w = script.geterror(np.dot(Xtrain, self.weights), ytrain)
            # print(self.weights)

            # give a limit to iterations used to plot, here I limit it to 5000 iterations
            if (count < 5000):
                # self.xaxis[count] = count
                # store the MSE for plotting
                self.yaxis[count] += err
            count = count + 1
        # print (count)

        # plt.plot(self.xaxis, self.yaxis/self.numruns)
        # plt.show()

    def data(self):
        """ return the averaged error in y axis """
        return self.yaxis/self.numruns

    def predict(self, Xtest):
        # Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xtest, self.weights)
        return ytest    