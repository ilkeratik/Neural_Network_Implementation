import numpy as np

def cross_entropy_cost(Y_hat, Y):
    m = Y.shape[1]
    logprobs = np.dot(np.log(Y_hat), Y.T)+ np.dot(np.log(1-Y_hat),(1-Y).T)
    cost = np.divide(logprobs, -m)
    return cost

def logistic_cost(Y_hat, Y):
    m = Y.shape[1]
    loss = np.dot(Y, np.log(Y_hat).T) + np.dot((1-Y),np.log(1-Y_hat).T)
    cost = np.divide(loss, -m)
    return cost