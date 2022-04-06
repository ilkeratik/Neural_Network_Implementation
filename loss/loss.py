import numpy as np
def absolute_error(y_hat, y):
    """
    Computes absolute error for given prediction and real value matrices.

    Arguments:
    y_hat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- absolute error
    """
    loss = np.sum(np.abs(y-y_hat))
    return loss

def squared_error(y_hat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- squared error
    """
    
    diff = y-y_hat
    loss = np.dot(diff, diff)
    return loss