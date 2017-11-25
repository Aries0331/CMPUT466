CMPUT 466 Assignment 3
Jinzhu Li
1461696

Python version: 2.7

Question 1:
 
 (a) We will get the error: RuntimeWarning: divide by zero encountered in true_divide
 This is because the standard deviation is zero.
 After fix the problem, include column of ones and not include column of ones has the same result.

 (d) Average error for Random: 49.34 +- 2.24693341989e-15
Best parameters for Naive Bayes: {'usecolumnones': False}
Average error for Naive Bayes: 25.78 +- 0.0
Best parameters for Naive Bayes Ones: {'usecolumnones': True}
Average error for Naive Bayes Ones: 25.78 +- 0.0
Best parameters for Linear Regression: {'regwgt': 0.0}
Average error for Linear Regression: 25.02 +- 1.12346670994e-15
Best parameters for Neural Network: {'nh': 32, 'transfer': 'sigmoid', 'epochs': 100, 'stepsize': 0.01}
Average error for Neural Network: 24.08 +- 1.12346670994e-15
Best parameters for Logistic Regression: {'regwgt': 0.0, 'regularizer': 'None'}
Average error for Logistic Regression: 26.64 +- 2.24693341989e-15

Generally, the neural network with 32 hidden units performers better than Linear Regression, than Naive Bayes, than logistic regression.

 Question 2:

 (a) Average error for Kernel Logistic Regress linear: 24.68 +- 0.0
Kernel logistic regression with a linear kernel has lower error than Naive Bayes, and Linear regression, but does not do better than Neural network.

 (b) Average error for Random: 49.698 +- 0.189154962927
Average error for Kernel Logistic Regress hamming: 24.188 +- 0.0872330212706 
Hamming kernal works better than the random predictor since it has a lower error