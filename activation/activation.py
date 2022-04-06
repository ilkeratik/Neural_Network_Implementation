import numpy as np

class Activation:

    @staticmethod
    def sigmoid(x):
        s = 1/(1+np.exp(-x))
        return s

    @staticmethod
    def sigmoid_derivative(x):
        s = Activation.sigmoid(x)
        ds = s*(1-s)
        return ds