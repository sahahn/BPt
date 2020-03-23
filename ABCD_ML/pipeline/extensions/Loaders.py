from sklearn.base import BaseEstimator

class Identity(BaseEstimator):
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        pass
    
    def fit_transform(self, X, y=None):
        
        return self.transform(X)
    
    def transform(self, X):
        
        return X.flatten()