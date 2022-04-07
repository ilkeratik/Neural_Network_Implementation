import numpy as np
from activation.activation import sigmoid
from loss.cost import cross_entropy_cost
import matplotlib.pyplot as plt

class OneHiddenLayerClassification:

    def __init_(self):
        self.parameters = {}
        self.costs = []

    def initialize_parameters(self, n_x, n_h, n_y):
        """
        Argument:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer
        
        Returns:
        params -- python dictionary containing your parameters:
                        W1 -- weight matrix of shape (n_h, n_x)
                        b1 -- bias vector of shape (n_h, 1)
                        W2 -- weight matrix of shape (n_y, n_h)
                        b2 -- bias vector of shape (n_y, 1)
        """ 
        #np.random.seed(2)   
        W1 = np.random.randn(n_h,n_x) * 0.01
        b1 = np.zeros((n_h,1))
        W2 = np.random.randn(n_y,n_h) * 0.01
        b2 = np.zeros((n_y,1))

        parameters = {"W1": W1,
                    "b1": b1,
                    "W2": W2,
                    "b2": b2}
        return parameters

    def forward_propagation(self, X):
        """
        Argument:
        X -- input data of size (n_x, m)
        parameters -- python dictionary containing your parameters (output of initialization function)
        
        Returns:
        A2 -- The sigmoid output of the second activation
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
        """
        parameters = self.parameters
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']

        # define layers and activations
        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1) # tanh used in activation of hidden layer
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2) # Y_pred
        
        assert A2.shape == (1, X.shape[1]), "shape of matrices doesn't match"
        
        cache = {"Z1": Z1,
                "A1": A1,
                "Z2": Z2,
                "A2": A2}
        
        return A2, cache
    
    def backward_propagation(self, cache, X, Y):
        """
        Implement the backward propagation using the instructions above.
        
        Arguments:
        parameters -- python dictionary containing our parameters 
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
        X -- input data of shape (2, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)
        
        Returns:
        grads -- python dictionary containing your gradients with respect to different parameters
        """
        m = X.shape[1]
        parameters = self.parameters
        W1 = parameters['W1']
        W2 = parameters['W2']
        A1 = cache['A1']
        A2 = cache['A2']

        dZ2 = A2-Y
        dW2 = np.divide(np.dot(dZ2, A1.T), m)
        db2 = np.divide(np.sum(dZ2, axis=1, keepdims=True), m)
        dZ1 = np.multiply(np.dot(W2.T,dZ2), (1-np.power(A1,2))) ## not dW2.T
        dW1 = np.divide(np.dot(dZ1, X.T), m)
        db1 = np.divide(np.sum(dZ1, axis=1, keepdims=True), m)
        
        grads = {"dW1": dW1,
                "db1": db1,
                "dW2": dW2,
                "db2": db2}
        
        return grads
    
    def gradient_descent_step(self, grads, learning_rate = 0.01):
        """
        Updates parameters using the gradient descent update rule given above
        
        Arguments:
        parameters -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients 
        
        Returns:
        parameters -- python dictionary containing your updated parameters 
        """
        # Retrieve weights and biases
        parameters = self.parameters
        W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        
        # Retrieve gradients
        dW1 = grads['dW1']
        db1 = grads['db1']
        dW2 = grads['dW2']
        db2 = grads['db2']

        # Update rule for each parameter
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2

        # update dictionary-parameters
        self.parameters = {"W1": W1,
                    "b1": b1,
                    "W2": W2,
                    "b2": b2}
    
    def train_nn_model(self, X, Y, n_hidden, num_iterations=5000, learning_rate=0.01, print_cost=False):
        """
        Arguments:
        X -- Input vector
        Y -- labels
        n_hidden: hidden layer size
        """
        n_x = X.shape[0]
        n_y = Y.shape[0]

        self.parameters = self.initialize_parameters(n_x, n_hidden, n_y)
        costs = []
        for i in range(num_iterations):
            A2, cache = self.forward_propagation(X)
            
            grads = self.backward_propagation(cache, X, Y)

            # update parameters
            self.gradient_descent_step(grads,learning_rate=learning_rate)

            if print_cost and i % 1000 == 0:
                cost = cross_entropy_cost(A2, Y)
                costs.append(cost)
                print("Cost after iteration %i: %f" %(i, cost))
        self.costs = costs
        return self.parameters
    
    def predict(self, X):
        A2, cache = self.forward_propagation(X)
        predictions = np.array([True if i> 0.5 else False for i in A2[0]]).reshape(1,-1)
    
        return predictions

    def load_planar_dataset():
        np.random.seed(1)
        m = 400 # number of examples
        N = int(m/2) # number of points per class
        D = 2 # dimensionality
        X = np.zeros((m,D)) # data matrix where each row is a single example
        Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
        a = 4 # maximum ray of the flower

        for j in range(2):
            ix = range(N*j,N*(j+1))
            t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
            r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
            X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
            Y[ix] = j
            
        X = X.T
        Y = Y.T

        return X, Y
    @staticmethod
    def load_extra_datasets():
        import sklearn.datasets
        N = 200
        noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
        noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
        blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
        gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)
        no_structure = np.random.rand(N, 2), np.random.rand(N, 2)
        
        return noisy_circles#, noisy_moons, blobs, gaussian_quantiles, no_structure
    @staticmethod
    def plot_decision_boundary(model, X, y):
        # Set min and max values and give it some padding
        x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
        y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
        h = 0.01
        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        # Predict the function value for the whole grid
        Z = model(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.ylabel('x2')
        plt.xlabel('x1')
        plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    
    def visualize_planar_model(self, X, Y):
        self.plot_decision_boundary(lambda x: self.predict(x.T), X, Y)
        plt.title("Decision Boundary for the model")
        plt.show()

    def visualize_costs(self):
        x = [x*1000 for x in range(1,len(self.costs)+1)]
        plt.plot(x,self.costs)
        plt.xlabel('Iteration') 
        plt.ylabel('Loss') 
        plt.title("Loss graph")
        plt.show()

if __name__ == '__main__':
    #X, Y = OneHiddenLayerClassification.load_planar_dataset()
    X, Y = OneHiddenLayerClassification.load_extra_datasets()
    X, Y = X.T, Y.reshape(1,-1)
    OHLC = OneHiddenLayerClassification()
    OHLC.train_nn_model(X, Y, 6, num_iterations=20000, learning_rate=0.4, print_cost=True)
    OHLC.visualize_planar_model(X,Y)
    OHLC.visualize_costs()