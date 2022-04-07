import numpy as np
import copy
from activation.activation import Activation
class LogisticRegression():

    def __init__(self):
        self.W = None
        self.b = None

    def initialize_with_zeros(self, dim):
        """
        This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
        
        Argument:
        dim -- size of the w vector we want (or number of parameters in this case)
        
        Returns:
        w -- initialized vector of shape (dim, 1)
        b -- initialized scalar (corresponds to the bias) of type float
        """
        
        w = np.zeros((dim,1))
        b = 0.0

        return w, b
    
    def propagate(self, w, b, X, Y):

        m = X.shape[1]
        A = Activation.sigmoid(np.dot(w.T,X)+b)
        cost = (-1/m)*np.sum((Y*np.log(A)+(1-Y)*np.log(1-A)))
        
        dw = (1/m)*np.dot(X,(A-Y).T)
        db = (1/m)*np.sum(A-Y)
        cost = np.squeeze(np.array(cost))

        grads = {"dw": dw,
                "db": db}
        
        return grads, cost

    def optimize(self, w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
        """
        This function optimizes w and b by running a gradient descent algorithm
        
        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- True to print the loss every 100 steps
        
        Returns:
        params -- dictionary containing the weights w and bias b
        grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
        
        Tips:
        You basically need to write down two steps and iterate through them:
            1) Calculate the cost and the gradient for the current parameters. Use propagate().
            2) Update the parameters using gradient descent rule for w and b.
        """
        
        w = copy.deepcopy(w)
        b = copy.deepcopy(b)
        
        costs = []
        
        for i in range(num_iterations):
            grads,cost = self.propagate(w, b, X, Y)
            
            dw = grads["dw"]
            db = grads["db"]
            
            w = w - learning_rate*dw
            b = b - learning_rate*db
            
            if i % 100 == 0:
                costs.append(cost)
            
                # Print the cost every 100 training iterations
                if print_cost:
                    print ("Cost after iteration %i: %f" %(i, cost))
        params = {"W": w,
                "b": b}
        
        grads = {"dw": dw,
                "db": db}
        
        return params, grads, costs
    
    def train(self, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
        self.W, self.b = self.initialize_with_zeros(X.shape[0])

        params, grads, costs = self.optimize(self.W, self.b, X, Y, num_iterations, learning_rate, print_cost=print_cost)
        
        self.W, self.b  = params['W'], params['b']

        return costs

    def predict(self, X):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
        
        Arguments:
        X -- data of size (num_px * num_px * 3, number of examples)
        
        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        '''
        w,b = self.W, self.b
        print(w,b)
        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        w = w.reshape(X.shape[0], 1)

        A = Activation.sigmoid(np.dot(w.T,X)+b)
        
        Y_prediction = np.array([1 if i> 0.5 else 0 for i in A[0]]).reshape(1,-1)

        return Y_prediction
    
    def print_metrics(self, y_pred, y_actual):
        print("model accuracy: {} %".format(100 - np.mean(np.abs(y_pred - y_actual)) * 100))
    
    def visualize_loss(self, costs):
        import matplotlib.pyplot as plt
        x = [x*100 for x in range(1,len(costs)+1)]
        plt.plot(x,costs)
        plt.xlabel('Iteration') 
        plt.ylabel('Loss') 
        plt.title("Loss graph")
        plt.show()
    
    @staticmethod
    def load_dataset():
        import h5py
        train_dataset = h5py.File('datasets/cat_not-cat/train_catvnoncat.h5', "r")
        train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
        train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

        test_dataset = h5py.File('datasets/cat_not-cat/test_catvnoncat.h5', "r")
        test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
        test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

        classes = np.array(test_dataset["list_classes"][:]) # the list of classes
        
        train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
        test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
        
        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

    @staticmethod
    def flatten_and_standardize_image_dataset(train_set_x_orig, test_set_x_orig):
        train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
        test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

        train_set_x = train_set_x_flatten / 255.
        test_set_x = test_set_x_flatten / 255.

        return train_set_x , test_set_x
if __name__ == '__main__':
    lr = LogisticRegression()
    
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = LogisticRegression.load_dataset()
    train_set_x , test_set_x = LogisticRegression.flatten_and_standardize_image_dataset(train_set_x_orig, test_set_x_orig)
    print(test_set_x.shape)
    costs = lr.train(train_set_x, train_set_y, num_iterations=900, learning_rate=0.01, print_cost=True)
    
    y_pred = lr.predict(test_set_x)
    lr.print_metrics(y_pred, test_set_y)
    lr.visualize_loss(costs)



        