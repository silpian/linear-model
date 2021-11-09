import numpy as np

class PseudoResidual:
    @staticmethod
    def mse_pseudo_residual(y, y_hat):
        return 2*(y - y_hat)
    
    @staticmethod
    def mae_pseudo_residual(y, y_hat):
        return np.sign(y - y_hat)
    
    @staticmethod
    def log_loss_pseudo_residual(y, y_hat):
        def sigmoid(x):
            return 1/(1 + np.expz(-x))
        
        return y - sigmoid(y_hat)
    
    keyword_to_function = {
        'mse': mse_pseudo_residual.__func__,
        'mae': mae_pseudo_residual.__func__,
        'log_loss': log_loss_pseudo_residual.__func__
    }


class Residual:
    @staticmethod
    def mse(y, y_hat):
        return np.mean((y - y_hat)**2)
    
    @staticmethod
    def mae(y, y_hat):
        return np.mean(np.abs(y - y_hat))
    
    keyword_to_function = {
        'mse': mse.__func__,
        'mae': mae.__func__
    }