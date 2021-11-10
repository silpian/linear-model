import numpy as np
from residual_utils import PseudoResidual, Residual
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
            return self.sample_y_mean + self.sample_y_std*np.matmul(X, self.beta_hat)
        
        else:
            return np.matmul(X, self.beta_hat)
        
    def get_params(self):
        """
        return coefficient vector self.beta_hat
        """

        return self.beta_hat
    
    def error(self, y, X, error_function='mse'):
        """
        y - M x 1 output vector
        X - M x p input matrix
        
        Assuming self.beta_hat has been fit on training data, 
        predict a new y vector given a new input matrix X using y = X * self.beta_hat.
        Then return the error
        """
        
        return Residual.keyword_to_function[error_function](y, self.predict(X))
    
    def r_squared(self, y, X):
        """
        y - M x 1 output vector
        X - M x p input matrix
        
        Assuming self.beta_hat has been fit on training data, 
        predict a new y vector given a new input matrix X using y = X * self.beta_hat.
        Then compute and return the R squared of the linear regression
        """
        return (1 - np.var(y - self.predict(X))/np.var(y))
    
    def adjusted_r_squared(self, y, X):
        """
        y - M x 1 output vector
        X - M x p input matrix
        
        Assuming self.beta_hat has been fit on training data, 
        predict a new y vector given a new input matrix X using y = X * self.beta_hat.
        Then compute and return the adjusted R squared of the linear regression
        """
        M, p = X.shape
        return (1 - (M-1)/(M-p) * np.var(y - self.predict(X))/np.var(y))
    
    def standardize_variables(self, y, X, set_params=True):
        """
        y - N x 1 output vector
        X - N x p input matrix
        
        returns:
        y - where each value has been standardized wrt to the vector's sample variance sample mean
        X - where each column of X has been standardized with respect to its sample variance and sample mean
        """
        if set_params:
            self.sample_y_mean = np.mean(y, axis=0)
            self.sample_X_mean = np.mean(X, axis=0)

            self.sample_y_std = np.std(y, axis=0)
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

class GradientBoostedOLS(LinearModel):
    def __init__(self, standardize=True):
        self.standardize = standardize
    
    def fit(self, 
            y, 
            X, 
            validation_fraction=0.2, 
            learning_rate=0.1, 
            tol=0.01,
            num_iter=10000, 
            max_num_iter_no_improvement=20, 
            error_function='mse'):
        """
        y - N x 1 output vector
        X - N x p input matrix
        validation_fraction - fraction of training set used for early stopping
        learning_rate - eta in the equation: beta_hat = beta_hat - eta * gradient
        tol - required improvement in mse
        num_iter - number of iterations
        num_iter_no_improvement - number of iterations without improvement > tol before early stopping

        Fits coefficients to y, X and stores in self.beta_hat using boosted OLS
        """
        pseudo_residual_function = PseudoResidual.keyword_to_function[error_function]
        residual_function = Residual.keyword_to_function[error_function]
        
        self.N, self.p = X.shape
        
        y_X = np.hstack((y, X))
        np.random.shuffle(y_X)
        y, X = y_X[:, [0]], y_X[:, 1:]
        
        split_index = int((1-validation_fraction)*self.N)
        
        X_train = X[:split_index]
        X_validation = X[split_index:]
        
        y_train = y[:split_index]
        y_validation = y[split_index:]
        
        if self.standardize:
            y_train, X_train = super().standardize_variables(y_train, X_train)
            y_validation, X_validation = super().standardize_variables(y_validation, X_validation, set_params=False)
        
        beta_hat = np.zeros((self.p, 1))
        
        y_train_prediction = np.zeros(y_train.shape)
        y_validation_prediction = np.zeros(y_validation.shape)
        
        pseudo_residual = pseudo_residual_function(y_train, y_train_prediction)
        min_validation_residual = residual_function(y_validation, y_validation_prediction)
        
        num_iter_no_improvement = 0
        for _ in range(num_iter):
            if num_iter_no_improvement >= max_num_iter_no_improvement:
                print(f"Early stopping at iteration: {_}")
                break
            # correlation coefficient = beta hat because y and X are standardized
            correlation = np.mean(X_train*pseudo_residual, axis=0)/np.std(pseudo_residual)
            max_correlation_idx = np.argmax(np.abs(correlation))
            parameter_update = learning_rate*correlation[max_correlation_idx]
            
            # add new univariate regressor on residual against chosen X variable
            beta_hat[max_correlation_idx] += parameter_update
            
            # update predictions and residual
            y_train_prediction += parameter_update * X_train[:, [max_correlation_idx]]
            y_validation_prediction += parameter_update * X_validation[:, [max_correlation_idx]]
            
            pseudo_residual = pseudo_residual_function(y_train, y_train_prediction)
            validation_residual = residual_function(y_validation, y_validation_prediction)
            
            if validation_residual < min_validation_residual - tol:
                num_iter_no_improvement = 0
                min_validation_residual = validation_residual
            else:
                num_iter_no_improvement += 1

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

    def choose_reg_param_for_lasso(self, y, X, max_is_OLS_size=True, grid_size=30):
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
        if max_is_OLS_size:
            ols = OLS()
            ols.fit_by_closed_form(y, X)
            max_grid = np.sum(np.abs(ols.get_params()))
        else:
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