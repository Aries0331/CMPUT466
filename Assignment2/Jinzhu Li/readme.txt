Name: Jinzhu Li
Student ID: 1461696

Question 2:

Python version: python 2.7

(a) When number of features more than 50, we will get a singular matrix error, and it is caused by the inverse of Xtranspose*X cannot be calculated since zeros in the matrix makes it not full-rank. This can occur because data sets include large numbers of features, which are identical, similar, or nearly linearly dependent.
To fix the issue, we can eithrt change the inverse function "np.linalg.inv" to "np.linalg.pinv", which calculate the pseudo-inverse instead, or we can use singular matrix decomposition.

(c) For training error, which implemented in this question, will be larger than (a) since we only foucs on the training set, and there is a chance of for (a) to overfitting with feature selection, which results in a quite small error for training data. 
But for test error, Ridge Regression in (c) will be better since it includes a l2 regularizer which balance the error and prior, or training data and test data, hence works better on new data, i.e. less error on test.

(f) For stochastic gradient descent, it is much faster to approach the minima(time to converge) although it is not always heading to the beat direction of minima.
For batch gradient descent, it takes longer to converge, and the error is larger at the beginning. But it always decreases in  the direction of minima and guarantees to converge at some point.
