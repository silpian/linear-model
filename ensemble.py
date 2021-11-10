import numpy as np

class Ensemble:
    def __init__(self, base_learner_class, num_of_base_learners=10):
        """
        
        Bagging Ensemble Learner
        Assumes randomization of data in fit() function of each base learner
        such as in the gradient boosted OLS class
        
        """
        if num_of_base_learners <= 0:
            raise NotImplementedError
        self.num_of_base_learners = num_of_base_learners
        self.base_learners = [base_learner_class() for _ in range(num_of_base_learners)]
        
    def fit(self, y, X):
        try:
            for base_learner in self.base_learners:
                base_learner.fit(y, X)
        except AttributeError:
            raise
    
    def predict(self, X):
        mean_prediction = np.zeros((X.shape[0], 1))
        for base_learner in base_learners:
            mean_prediction += base_learner.predict(X)/self.num_of_base_learners
        
        return mean_prediction

class LinearEnsemble(Ensemble):
    def __init__(self, base_learner_class, num_of_base_learners=10):
        super().__init__(base_learner_class, num_of_base_learners)
        
        self.sample_X_mean = 0
        self.sample_X_std = 0
        self.sample_y_mean = 0
        self.sample_y_std = 0
        
        self.standardize = self.base_learners[0].standardize
    
    def fit(self, y, X):
        self.beta_hat = np.zeros((X.shape[1], 1))
        
        try:
            for base_learner in self.base_learners:
                base_learner.fit(y, X)
                self.beta_hat += base_learner.beta_hat/self.num_of_base_learners
                if self.standardize:
                    self.sample_X_mean += base_learner.sample_X_mean/self.num_of_base_learners
                    self.sample_y_mean += base_learner.sample_y_mean/self.num_of_base_learners
                    self.sample_X_std += base_learner.sample_X_std**2
                    self.sample_y_std += base_learner.sample_y_std**2
            if self.standardize:
                self.sample_X_std = np.sqrt(self.sample_X_std)
                self.sample_y_std = np.sqrt(self.sample_y_std)
                
        except AttributeError:
            raise
    
    def predict(self, X):
        if self.standardize:
            X = (X - self.sample_X_mean)/self.sample_X_std
            return self.sample_y_mean + self.sample_y_std*np.matmul(X, self.beta_hat)
        
        else:
            return np.matmul(X, self.beta_hat)