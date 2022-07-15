import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
print(current)
parent = os.path.dirname(current)
sys.path.append(parent)


import numpy as np
from activation.activation import sigmoid
from loss.cost import cross_entropy_cost
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import seaborn as sns
class OneHiddenLayerCls:

    def __init__(self, parameters={}):
        self.parameters = parameters
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
    
    def train_nn_model(self, X, Y, n_hidden, num_iterations=5000, learning_rate=0.01, batch_size=False, print_cost=False):
        """
        Arguments:
        X -- Input vector
        Y -- labels
        n_hidden: hidden layer size
        """
        n_x = X.shape[0]
        n_y = Y.shape[0]

        m_x = X.shape[1]
        self.parameters = self.initialize_parameters(n_x, n_hidden, n_y)
        costs = []
        for i in range(num_iterations):
            Y_batch = None
            if batch_size: # if batch_size is set mini batches will be used to optimize
                batch = np.random.choice(range(m_x), batch_size)
                X_batch = X.iloc[:,batch]
                Y_batch = Y[0,batch].reshape(1,-1)
                A2, cache = self.forward_propagation(X_batch)
                grads = self.backward_propagation(cache, X_batch, Y_batch)
            
            else:
                A2, cache = self.forward_propagation(X)
                grads = self.backward_propagation(cache, X, Y)

            # update parameters
            self.gradient_descent_step(grads,learning_rate=learning_rate)

            if print_cost and i % 10 == 0:
                if batch_size:
                    cost = cross_entropy_cost(A2, Y_batch)
                else:
                    cost = cross_entropy_cost(A2, Y)

                costs.append(cost)
                print("Cost after iteration %i: %f" %(i, cost))
        self.costs = costs
        return self.parameters
    
    def predict(self, X):
        A2, cache = self.forward_propagation(X)
        predictions = np.array([True if i> 0.5 else False for i in A2[0]]) #.reshape(1,-1)
    
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

    def predict_weight_dimension_reduced(self, X):
        temp = self.parameters
        self.parameters["W1"] = self.parameters["W1"][:, 0:2]
        # self.parameters["W2"] = self.parameters["W2"][:, 0:2]
        A2, cache = self.forward_propagation(X)
        predictions = np.array([True if i> 0.5 else False for i in A2[0]]) #.reshape(1,-1)

        self.weights = temp
        return predictions

    @staticmethod
    def plot_decision_boundary(model, X, y):
        # Set min and max values and give it some padding
        if X.shape[0] != 2:
            X = X[-6:-8:-1]
        print('s')
        print(X.shape, y.shape)
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
        self.plot_decision_boundary(lambda x: self.predict_weight_dimension_reduced(x.T), X.T, Y.T)
        plt.title("Decision Boundary for the model")
        plt.show()

    def plot_costs(self, ax):
        x = [x*10 for x in range(1,len(self.costs)+1)]
        ax.plot(x,self.costs)
        ax.set_xlabel('Iteration') 
        ax.set_ylabel('Loss') 
        ax.set_title("Loss graph")

    def calculate_metrics(self, y_actual, y_pred):
        f1_sc = f1_score(y_actual, y_pred)
        acc_sc = accuracy_score(y_actual, y_pred)
        rec_sc = recall_score(y_actual, y_pred)
        prec_sc = precision_score(y_actual, y_pred)
        conf = confusion_matrix(y_actual, y_pred)
        res = " accuracy score: {}\n recall_score: {}\n precision_score: {}\n f1_score: {}"
        print(res.format(acc_sc, rec_sc, prec_sc, prec_sc, f1_sc))

        return conf

    def print_metrics(self, X_train, X_test, y_train, y_test):
        y_pred_train = OHLC.predict(X_train.T)
        y_pred_test = OHLC.predict(X_test.T)
        print(y_pred_train.shape, y_pred_test.shape, y_train.shape, y_test.shape)
        print('----------------------------\nThe training accuracy\n----------------------------')
        train_conf_mat = OHLC.calculate_metrics(y_train, y_pred_train)
        print('----------------------------\nThe test accuracy\n--------------------------------')
        test_conf_mat = OHLC.calculate_metrics(y_test, y_pred_test)

        return (train_conf_mat, test_conf_mat)

    def visualize_conf_mat(self, cf_matrix, ax=None):
        group_names = ['True Neg','False Pos','False Neg','True Pos']
        group_counts = ['{0:0.0f}'.format(value) for value in
                        cf_matrix.flatten()]
        group_percentages = ['{0:.2%}'.format(value) for value in
                            cf_matrix.flatten()/np.sum(cf_matrix)]
        labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
                zip(group_names,group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)

        categories = ['Non-fraud', 'Fraud']
        if ax:
            sns.heatmap(cf_matrix, annot=labels, ax=ax, fmt='', xticklabels=categories, yticklabels=categories, cmap='Blues')
        else:
            sns.heatmap(cf_matrix, annot=labels, fmt='', xticklabels=categories, yticklabels=categories, cmap='Blues')

    def plot_conf_matrices(self,train_conf_mat, test_conf_mat , ax1, ax2):

        ax1.set_title('Confusion Matrix for training accuracy')
        self.visualize_conf_mat(train_conf_mat, ax1)

        ax2.set_title('Confusion Matrix for test accuracy')
        self.visualize_conf_mat(test_conf_mat, ax2)
    
    @staticmethod
    def preprocess_balance_dataset_and_save():
        df = pd.read_csv(parent + '/datasets/fraud_detection/creditcard.csv')
    
        # Scaling features
        #---------------------------------
        rob_scaler = RobustScaler()
        df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
        df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))
        df.drop(['Time','Amount'], axis=1, inplace=True)
        scaled_amount = df['scaled_amount']
        scaled_time = df['scaled_time']
        df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
        df.insert(0, 'scaled_amount', scaled_amount)
        df.insert(1, 'scaled_time', scaled_time)

        # Undersampling
        #---------------------------------
        df = df.sample(frac=1)
        # amount of fraud classes 492 rows.
        fraud_df = df.loc[df['Class'] == 1]
        non_fraud_df = df.loc[df['Class'] == 0][:492]
        normal_distributed_df = pd.concat([fraud_df, non_fraud_df])
        # Shuffle dataframe rows
        undersampled_df = normal_distributed_df.sample(frac=1, random_state=42)

         # Removing outliers
        #---------------------------------
        # # -----> V14 Removing Outliers (Highest Negative Correlated with Labels)
        v14_fraud = undersampled_df['V14'].loc[undersampled_df['Class'] == 1].values
        q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
        v14_iqr = q75 - q25

        v14_cut_off = v14_iqr * 1.5
        v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off
        undersampled_df = undersampled_df.drop(undersampled_df[(undersampled_df['V14'] > v14_upper) | (undersampled_df['V14'] < v14_lower)].index)

        # -----> V12 removing outliers from fraud transactions
        v12_fraud = undersampled_df['V12'].loc[undersampled_df['Class'] == 1].values
        q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)
        v12_iqr = q75 - q25

        v12_cut_off = v12_iqr * 1.5
        v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off
        undersampled_df = undersampled_df.drop(undersampled_df[(undersampled_df['V12'] > v12_upper) | (undersampled_df['V12'] < v12_lower)].index)

        # -----> V10 removing outliers from fraud transactions
        v10_fraud = undersampled_df['V10'].loc[undersampled_df['Class'] == 1].values
        q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)
        v10_iqr = q75 - q25

        v10_cut_off = v10_iqr * 1.5
        v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off
        undersampled_df = undersampled_df.drop(undersampled_df[(undersampled_df['V10'] > v10_upper) | (undersampled_df['V10'] < v10_lower)].index)

        undersampled_df.to_csv('mini_ds_creditcard.csv')


if __name__ == '__main__':
    OHLC = OneHiddenLayerCls()

    # Planar dataset
    # X, Y = OneHiddenLayerCls.load_planar_dataset()
    # OHLC.train_nn_model(X, Y, 10, num_iterations=1000, learning_rate=0.3, print_cost=True)
    # OHLC.visualize_planar_model(X.T,Y.T)

    #X, Y = OneHiddenLayerCls.load_extra_datasets()
    
    # Fraud_detection dataset
    #OneHiddenLayerCls.preprocess_balance_dataset_and_save() # RUN ONLY ONCE
    df = pd.read_csv(parent + '/datasets/fraud_detection/mini_ds_creditcard.csv')
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,1:-1], df.iloc[:,-1], test_size=0.1, random_state=42)
    X , Y = (X_train, y_train)
    print(X.columns)
    print(X.shape, Y.shape)
    OHLC.train_nn_model(X.T, Y.values.reshape((-1,1)).T, 20, num_iterations=5000, learning_rate=0.3, batch_size=100, print_cost=True)
    X_np = X.to_numpy()
    # OHLC.visualize_planar_model(X_np, Y) # Not sure if it works true for non planar 2D data.
    
    f, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(15, 5))
    train_conf_mat, test_conf_mat = OHLC.print_metrics(X_train, X_test, y_train, y_test)
    OHLC.plot_conf_matrices(train_conf_mat, test_conf_mat, ax1, ax2)
    OHLC.plot_costs(ax3)
    plt.show()
   
