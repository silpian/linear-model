import numpy as np
from utils import soft_thresholding

class LinearModel:
    def predict(self, X):
        """
        X - M x p input matrix

        Assuming self.beta_hat has been fit on training data, 
        predict a new y vector given a new input matrix X using y = X * self.beta_hat. 

        Returns the predicted y vector.
        """

        M, p = X.shape
        assert (self.p == p), f"Input matrix has incorrect number of parameters. Expected {self.p}, got {p}"

        if self.standardize:
            X = (X - self.sample_X_mean)/self.sample_X_std

        return np.matmul(X, self.beta_hat)

    def get_params(self):
        """
        return coefficient vector self.beta_hat
        """

        return self.beta_hat

    def mse(self, y, X):
        """
        y - M x 1 output vector
        X - M x p input matrix
        
        Assuming self.beta_hat has been fit on training data, 
        predict a new y vector given a new input matrix X using y = X * self.beta_hat.
        Then, compute the mean squared error (1/M)*||y - X*self.beta_hat||^2
        
        Return the mean squared error.
        """
        M, p = X.shape

        if self.standardize:
            y = (y - np.mean(y, axis=0))/np.std(y, axis=0)
        
        return (1/M)*np.sum((y - self.predict(X))**2)

    def standardize_variables(self, y, X):
        """
        y - N x 1 output vector
        X - N x p input matrix
        
        returns:
        y - where each value has been standardized wrt to the vector's sample variance sample mean
        X - where each column of X has been standardized with respect to its sample variance and sample mean
        """

        self.sample_y_mean = np.mean(y, axis=0)
        self.sample_y_std = np.mean(y, axis = 0)
        self.sample_X_mean = np.mean(X, axis=0)
        self.sample_X_std = np.std(X, axis=0)

        return (y - self.sample_y_mean)/self.sample_y_std, (X - self.sample_X_mean)/self.sample_X_std

class OLS(LinearModel):
    def __init__(self, standardize=False):
        self.standardize = standardize

    def fit_by_closed_form(self, y, X):
        """
        y - N x 1 output vector
        X - N x p input matrix

        Fits coefficients to y, X and stores in self.beta_hat using the normal equation
        """

        self.N, self.p = X.shape

        if self.standardize:
            y, X = super().standardize_variables(y, X)

        self.beta_hat = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, y))

    def fit_by_gradient_descent(self, y, X, learning_rate=0.00001, iter=100):
        """
        y - N x 1 output vector
        X - N x p input matrix
        learning_rate - eta in the equation: beta_hat = beta_hat - eta * gradient
        iter - number of iterations

        Fits coefficients to y, X and stores in self.beta_hat using gradient descent
        """
        self.N, self.p = X.shape

        if self.standardize:
            y, X = super().standardize_variables(y, X)

        beta_hat = np.random.rand(self.p, 1)
        
        for _ in range(iter):
            beta_hat = beta_hat + 2*learning_rate*np.matmul(X.T, y - np.matmul(X, beta_hat))

        self.beta_hat = beta_hat

    def fit_by_coordinate_descent(self, y, X, iter=100):
        """
        y - N x 1 output vector
        X - N x p input matrix
        iter - number of iterations

        Fits coefficients to y, X and stores in self.beta_hat using cyclical coordinate descent
        """
        self.N, self.p = X.shape

        if self.standardize:
            y, X = super().standardize_variables(y, X)

        beta_hat = np.random.rand(self.p, 1)
        
        for _ in range(iter):
            residual = y - np.matmul(X, beta_hat)
            for j in range(self.p):
                residual = residual + (X[:, j]*beta_hat[j]).reshape(-1,1)
                beta_hat[j] = (1/np.dot(X[:, j], X[:, j])) * np.matmul(residual.T, X[:, j])
                residual = residual - (X[:, j]*beta_hat[j]).reshape(-1,1) 
                # Note that beta_hat[j] that we add back to the residual
                # is the updated beta_hat[j]. Thus, the the residual changes with each coordinate update
        
        self.beta_hat = beta_hat

class Ridge(LinearModel):
    def __init__(self, standardize=False):
        self.standardize = standardize

    def fit_by_closed_form(self, y, X, reg_param):
        """
        y - N x 1 output vector
        X - N x p input matrix
        reg_param - L2 regularization parameter, equivalent to lambda in the equation, but lambda is a Python keyword
        Fits coefficients to y, X and stores in self.beta_hat using the closed form equation:
            beta_hat = (X^T X + lambda I_p)^{-1} X^T y
        """

        self.N, self.p = X.shape

        if self.standardize:
            y, X = super().standardize_variables(y, X)

        self.beta_hat = np.matmul(np.linalg.inv(np.matmul(X.T, X) + reg_param*np.identity(self.p)), np.matmul(X.T, y))

    def fit_by_gradient_descent(self, y, X, reg_param, learning_rate=0.001, iter=1000):
        """
        y - N x 1 output vector
        X - N x p input matrix
        reg_param - L2 regularization parameter, equivalent to lambda in the equation, but lambda is a Python keyword
        learning_rate - eta in the equation: beta_hat = beta_hat - eta * gradient
        iter - number of iterations

        Fits coefficients to y, X and stores in self.beta_hat using gradient descent
        """
        self.N, self.p = X.shape

        if self.standardize:
            y, X = super().standardize_variables(y, X)

        beta_hat = np.random.rand(self.p, 1)
        
        for _ in range(iter):
            beta_hat = beta_hat + 2*learning_rate*(np.matmul(X.T, y - np.matmul(X, beta_hat)) - reg_param*beta_hat)

        self.beta_hat = beta_hat

class Lasso(LinearModel):
    def __init__(self, standardize=False):
        self.standardize = standardize

    def fit_by_coordinate_descent(self, y, X, reg_param=None, iter=100):
        """
        y - N x 1 output vector
        X - N x p input matrix
        reg_param - L1 regularization parameter, equivalent to lambda in the equation, but lambda is a Python keyword
        iter - number of iterations

        Fits coefficients to y, X and stores in self.beta_hat using cyclical coordinate descent
        """
        self.N, self.p = X.shape

        if self.standardize:
            y, X = super().standardize_variables(y, X)

        beta_hat = np.random.rand(self.p, 1)

        if reg_param is None:
            reg_param = choose_reg_param_for_lasso(y, X)
        for _ in range(iter):
            residual = y - np.matmul(X, beta_hat)
            for j in range(self.p):
                residual = residual + (X[:, j]*beta_hat[j]).reshape(-1,1)
                
                beta_hat[j] = ((1/np.dot(X[:, j], X[:, j])) * 
                               soft_thresholding(np.matmul(residual.T, X[:, j]), reg_param))
                
                residual = residual - (X[:, j]*beta_hat[j]).reshape(-1,1) 
                # Note that beta_hat[j] that we add back to the residual
                # is the updated beta_hat[j]. Thus, the the residual changes with each coordinate update.

        self.beta_hat = beta_hat

    def choose_reg_param_for_lasso(self, y, X, grid_size=30):
        """
        y - N x 1 output vector
        X - N x p input matrix
        
        Returns the best L1 regularization parameter (lambda) for Lasso determined by leave-one-out cross validation
        """
        self.N, self.p = X.shape

        if self.standardize:
            y, X = super().standardize_variables(y, X)
        
        # choose a grid of possible values for lambda
        min_grid = 0
        max_grid = (1/self.N)*np.max(np.matmul(y.T, X))
        print(f"Regularization parameter chosen among {grid_size} evenly spaced values from 0 to {max_grid}")
        
        grid = np.linspace(min_grid, max_grid, grid_size)
        mse_vector = np.zeros((grid_size, 1))
        
        for row in range(self.N):
            X_leave_one_out = np.delete(X, row, axis=0)
            y_leave_one_out = np.delete(y, row, axis=0)
            
            beta_matrix = np.zeros((grid_size, self.p))
            for i in range(grid_size):
                reg_param = grid[i]
                self.fit_by_coordinate_descent(y_leave_one_out, X_leave_one_out, reg_param=reg_param, iter=10)
                beta_matrix[i] = self.get_params().T
            mse_vector = mse_vector + (np.matmul(beta_matrix, X[row].reshape(-1,1)) - y[row])
        index_of_min_mse = np.argmin(mse_vector**2)

        return grid[index_of_min_mse]