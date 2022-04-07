import numpy as np
def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    v = image.reshape(image.shape[0]*image.shape[1]*image.shape[2],1)
    
    return v

def normalize_rows(x):
    """
    Normalizes each row of the matrix x (to have unit length).
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = np.divide(x,x_norm)

    return x

def softmax(x):
    """
    Calculates the softmax for each row of the input x. for matrices of shape (m,n).
    Argument:
    x -- A numpy matrix of shape (m,n)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (m,n)
    """
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = np.divide(x_exp, x_sum)
    
    return s

