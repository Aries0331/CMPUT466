from __future__ import division  # floating point division
import numpy as np
import utilities as utils

class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
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

    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, parameters={} ):
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1

        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.params['regwgt']*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1
        ytest[ytest < 0] = 0
        return ytest

class NaiveBayes(Classifier):
    """ Gaussian naive Bayes;  """

    def __init__(self, parameters={}):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it should ignore this last feature
        self.params = {'usecolumnones': True}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.means = []
        self.stds = []
        self.numfeatures = 0
        self.numclasses = 0

    def learn(self, Xtrain, ytrain):
        """
        In the first code block, you should set self.numclasses and
        self.numfeatures correctly based on the inputs and the given parameters
        (use the column of ones or not).

        In the second code block, you should compute the parameters for each
        feature. In this case, they're mean and std for Gaussian distribution.
        """

        ### YOUR CODE HERE
        self.numclasses = 2
        self.numfeatures = 9
        ### END YOUR CODE

        origin_shape = (self.numclasses, self.numfeatures)
        self.means = np.zeros(origin_shape)
        self.stds = np.zeros(origin_shape)

        ### YOUR CODE HERE
        self.mean = np.mean(Xtrain, axis=0)
        self.std = np.std(Xtrain, axis=0)

        self.class_mean = np.zeros(origin_shape)
        self.class_std = np.zeros(origin_shape)

        for i in set(ytrain):
            index = np.where(ytrain==i)
            # print (index)
            self.class_mean[i] = np.mean(Xtrain[index], axis=0)
            self.class_std[i] = np.std(Xtrain[index], axis=0)
        ### END YOUR CODE

        assert self.means.shape == origin_shape
        assert self.stds.shape == origin_shape

    def predict(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """
        ytest = np.zeros(Xtest.shape[0], dtype=int)
        
        ### YOUR CODE HERE
        h =[]
        for x in Xtest:
            xless = np.repeat([x], self.numclasses, axis=0)
            likelihood = (1.0/np.sqrt(2*np.pi*np.square(self.class_std))) * np.exp((-1.0/(2*np.square(self.class_std)))*np.square(xless-self.class_mean))
            likelihood = np.prod(likelihood, axis=1)
            ytest.tolist().append(np.argmax(likelihood))
        for i in range (Xtest.shape[0]):
            if ytest[i] >= 0.5:
                ytest[i] = 1
            else:
                ytest[i] = 0        
        # print (ytest.shape)
        ### END YOUR CODE

        assert len(ytest) == Xtest.shape[0]
        return ytest

class LogitReg(Classifier):

    def __init__(self, parameters={}):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None'}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        if self.params['regularizer'] is 'l1':
            self.regularizer = (utils.l1, utils.dl1)
        elif self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))
    
    def sigmoid(self, x):
        ''' sigmoid function '''
        y = 1.0/(1+np.exp(-1.0*x))

        return y

    def logit_cost(self, theta, X, y):
        """
        Compute cost for logistic regression using theta as the parameters.
        """

        cost = 0.0

        ### YOUR CODE HERE
        # print("--1")
        p_1 = utils.sigmoid(np.dot(theta, X))
        # print (p_1)
        cost = y*np.log(p_1) + (1-y)*np.log(1-p_1) 
        cost = cost[0] 
        ### END YOUR CODE

        return cost

    def logit_cost_grad(self, theta, X, y):
        """
        Compute gradients of the cost with respect to theta.
        """

        grad = np.zeros(len(theta))

        ### YOUR CODE HERE
        # print (X.shape, y.shape)
        p_1 = utils.sigmoid(np.dot(X,theta))
        # print (p_1.shape, y.shape, X.shape)
        grad = p_1 - y
        ### END YOUR CODE

        return grad

    def learn(self, Xtrain, ytrain):
        """
        Learn the weights using the training data
        """

        self.weights = np.zeros(Xtrain.shape[1],)

        ### YOUR CODE HERE
        epochs = 100
        stepsize = 0.01
        numsamples = Xtrain.shape[0]
        for i in range (epochs):
            # shuffle data points from 1, ..., numbsamples
            arr = np.arange(numsamples)
            np.random.shuffle(arr)
            for j in arr:
                gradient = np.dot(self.logit_cost_grad(self.weights, Xtrain[arr[j]], ytrain[arr[j]]), Xtrain[arr[j]])
                # print (gradient)
                stepsize = 0.01/(1+i) # decrease stepsize to converge
                self.weights = self.weights-stepsize*gradient
        ### END YOUR CODE

    def predict(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """
        ytest = np.zeros(Xtest.shape[0], dtype=int)

        ### YOUR CODE HERE
        # print("22")
        # print (Xtest.shape, self.weights.shape)
        h = utils.sigmoid(np.dot(Xtest, self.weights))
        for i in range (Xtest.shape[0]):
            if h[i] >= 0.5:
                ytest[i] = 1
            else:
                ytest[i] = 0

        ### END YOUR CODE

        assert len(ytest) == Xtest.shape[0]
        return ytest

class NeuralNet(Classifier):
    """ Implement a neural network with a single hidden layer. Cross entropy is
    used as the cost function.

    Parameters:
    nh -- number of hidden units
    transfer -- transfer function, in this case, sigmoid
    stepsize -- stepsize for gradient descent
    epochs -- learning epochs

    Note:
    1) feedforword will be useful! Make sure it can run properly.
    2) Implement the back-propagation algorithm with one layer in ``backprop`` without
    any other technique or trick or regularization. However, you can implement
    whatever you want outside ``backprob``.
    3) Set the best params you find as the default params. The performance with
    the default params will affect the points you get.
    """
    def __init__(self, parameters={}):
        self.params = {'nh': 16,
                    'transfer': 'sigmoid',
                    'stepsize': 0.01,
                    'epochs': 10}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')
        # self.w_input = None
        # self.w_output = None

    def init(self, X, Y):
        std = 1.0/np.sqrt(X.shape[1])
        self.numfeatures = X.shape[1]
        self.w_input = np.random.normal(0, std, (self.params['nh'], self.numfeatures))
        self.w_output = np.random.normal(0, std, (1, self.params['nh']))
        # print(self.w_input.shape, self.w_output.shape)

    def feedforward(self, inputs):
        """
        Returns the output of the current neural network for the given input
        """
        # hidden activations
        # print(self.w_input.shape, inputs.shape)
        a_hidden = self.transfer(np.dot(self.w_input, inputs)) # f2

        # output activations
        # print(self.w_output.shape, a_hidden.shape)
        a_output = self.transfer(np.dot(self.w_output, a_hidden)) # f1

        return (a_hidden, a_output)

    def backprop(self, x, y):
        """
        Return a tuple ``(nabla_input, nabla_output)`` representing the gradients
        for the cost function with respect to self.w_input and self.w_output.
        """

        ### YOUR CODE HERE
        # h = np.zeros(self.params['nh'])
        # for i in range ():
        # print(x.shape)
        h, y_hat = self.feedforward(x)
        # print(h.shape, y_hat.shape)
        # print("-----")
        # print (self.feedforward(x))
        # print("-----")
        d_1 = y_hat - y
        d_2 = np.zeros(self.params['nh'])
        nabla_output = np.zeros((1,self.params['nh']))
        for i in range (self.params['nh']):
            nabla_output[0][i] = d_1 * h[i]
        nabla_input = np.zeros((self.params['nh'], self.numfeatures))
        for i in range (self.params['nh']):
            # print (h.shape, self.w_output.shape)
            d_2[i] = (self.w_output[0][i] * d_1) * h[i] * (1-h[i])
            nabla_input[i] = np.dot(d_2[i], x)    
        # print(self.w_output.shape, nabla_output.shape)
        # print(nabla_input, nabla_output)
        ### END YOUR CODE

        assert nabla_input.shape == self.w_input.shape
        assert nabla_output.shape == self.w_output.shape
        return (nabla_input, nabla_output)

    # TODO: implement learn and predict functions
    def learn(self, Xtrain, ytrain):
        """
        Learn the weights using the training data
        """
        self.init(Xtrain, ytrain)
        stepsize = self.params['stepsize']
        epochs = self.params['epochs']
        # nabla_input, nabla_output = self.backprop(Xtrain, ytrain)
        for i in range (epochs):
            arr = np.arange(Xtrain.shape[0])
            np.random.shuffle(arr)
            for j in arr:
                # gradient_1 = np.dot(np.subtract(np.dot(Xtrain[arr[j]].T, self.weights), ytrain[arr[j]]), Xtrain[arr[j]])
                # print ("----")
                # print (Xtrain[j].shape)
                gradient_1, gradient_2 = self.backprop(Xtrain[j], ytrain[j])
                # print (gradient)
                self.w_output = self.w_output - stepsize*gradient_2
                self.w_input = self.w_input - stepsize*gradient_1


    def predit(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """
        ytest = np.zeros(Xtest.shape[0], dtype=int)

        for i in range (Xtest.shape[0]):
            h, y = self.feedforward(Xtest[i])
            if y >= 0.5:
                ytest[i] = 1
            else:
                ytest[i] = 0

        assert len(ytest) == Xtest.shape[0]
        return ytest        

class KernelLogitReg(LogitReg):
    """ Implement kernel logistic regression.

    This class should be quite similar to class LogitReg except one more parameter
    'kernel'. You should use this parameter to decide which kernel to use (None,
    linear or hamming).

    Note:
    1) Please use 'linear' and 'hamming' as the input of the paramteter
    'kernel'. For example, you can create a logistic regression classifier with
    linear kerenl with "KernelLogitReg({'kernel': 'linear'})".
    2) Please don't introduce any randomness when computing the kernel representation.
    """
    def __init__(self, parameters={}):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None', 'kernel': 'None'}
        self.reset(parameters)

    def resrt(self, parameters):
        self.resetparams(parameters)

    def init(self, Xtrain, ytrain):
        self.numcenters = 10
        self.centers = Xtrain[:self.numcenters]

    def linear(self, x, c):
        '''
        linear kernel
        '''
        k = 0
        for i in range (x.shape[0]):
            k = k + x[i]*c[i]
        return k

    def hamming(self, x, c):
        '''
        Hamming distance kernel
        '''
        k = 0 
        for i in range (len(x)):
            if x[i] != c[i]:
                k = k + 1
        return k

    def logit_cost(self, theta, X, y):
        """
        Compute cost for logistic regression using theta as the parameters.
        """

        cost = 0.0

        for i in range (X.shape[0]):
            cost = cost + (y[i]-1)*np.dot(X[i], theta) + np.log(utils.sigmoid(np.dot(X[i], theta)))

        cost = cost/n*(-1.0)

        return cost

    def logit_cost_grad(self, theta, X, y):
        """
        Compute gradients of the cost with respect to theta.
        """

        grad = np.zeros(len(theta))
        stepsize = 0.01
        grad = np.dot(utils.sigmoid(np.dot(X, theta))-y, X)
        grad = grad * stepsize       

        return grad

    def transform(self, Xtrain):
        '''
        transform the data to new representation
        '''
        Ktrain = np.zeros((Xtrain.shape[0], self.numcenters))

        for i in range (Xtrain.shape[0]):
            for j in range (self.numcenters):
                if self.params['kernel'] == 'linear':
                    Ktrain[i][j] = self.linear(Xtrain[i], self.centers[j])
                elif self.params['kernel'] == 'hamming':
                    Ktrain[i][j] == self.hamming(Xtrain[i], self.centers[j])
        return Ktrain

    def learn(self, Xtrain, ytrain):
        """
        Learn the weights using the training data.

        Ktrain the is the kernel representation of the Xtrain.
        """
        Ktrain = None

        ### YOUR CODE HERE
        self.init(Xtrain, ytrain)
        Ktrain = self.transform(Xtrain)
        ### END YOUR CODE

        self.weights = np.zeros(Ktrain.shape[1],)

        ### YOUR CODE HERE
        epochs = 100
        stepsize = 0.01
        numsamples = Xtrain.shape[0]
        for i in range (epochs):
            # shuffle data points from 1, ..., numbsamples
            arr = np.arange(numsamples)
            np.random.shuffle(arr)
            for j in arr:
                gradient = np.dot(self.logit_cost_grad(self.weights, Ktrain[j], ytrain[j]), Ktrain[j])
                # print (gradient)
                self.weights = self.weights-stepsize*gradient
        ### END YOUR CODE

        self.transformed = Ktrain # Don't delete this line. It's for evaluation.

    # TODO: implement necessary functions
    def predict(self, Xtest):
        """
        Use the parameters computed in self.learn to give predictions on new
        observations.
        """
        ytest = np.zeros(Xtest.shape[0], dtype=int)

        ktest = self.transform(Xtest)
        ytest = utils.sigmoid(np.dot(ktest, self.weights))
        for i in range (len(ytest)):
            if ytest[i] >= 0.5:
                ytest[i] = 1
            else:
                ytest[i] = 0

        return ytest       

# ======================================================================

def test_lr():
    print("Basic test for logistic regression...")
    clf = LogitReg()
    theta = np.array([0.])
    X = np.array([[1.]])
    y = np.array([0])

    try:
        cost = clf.logit_cost(theta, X, y)
    except:
        raise AssertionError("Incorrect input format for logit_cost!")
    assert isinstance(cost, float), "logit_cost should return a float!"

    try:
        grad = clf.logit_cost_grad(theta, X, y)
    except:
        raise AssertionError("Incorrect input format for logit_cost_grad!")
    assert isinstance(grad, np.ndarray), "logit_cost_grad should return a numpy array!"

    print("Test passed!")
    print("-" * 50)

def test_nn():
    print("Basic test for neural network...")
    clf = NeuralNet()
    X = np.array([[1., 2.], [2., 1.]])
    y = np.array([0, 1])
    clf.learn(X, y)

    assert isinstance(clf.w_input, np.ndarray), "w_input should be a numpy array!"
    assert isinstance(clf.w_output, np.ndarray), "w_output should be a numpy array!"

    try:
        res = clf.feedforward(X[0, :])
    except:
        raise AssertionError("feedforward doesn't work!")

    try:
        res = clf.backprop(X[0, :], y[0])
    except:
        raise AssertionError("backprob doesn't work!")

    print("Test passed!")
    print("-" * 50)

def main():
    test_lr()
    test_nn()

if __name__ == "__main__":
    main()
