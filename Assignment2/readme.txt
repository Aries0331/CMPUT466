Name: Jinzhu Li
Student ID: 1461696

Question 2:

Python version:

(a) When number of features more than 50, we will get a singular matrix error, and it is caused by the inverse of Xtranspose*X cannot be calculated since zeros in the matrix makes it not full-rank. To fix the issue, we can eithrt change the inverse function "np.linalg.inv" to "np.linalg.pinv", which calculate the pseudo-inverse instead, or we can use singular matrix decomposition.

(c) For training error, which implemented in this question, will be larger than (a) since we only foucs on the training set, and there is a chance of overfitting. But for test error, Ridge Regression in (c) will be better since it includes a regularizer which balance the error and prior, hence works better on new data, i.e. less error.

(f) 
