import numpy as np


class Ensemble_Model():

    def __init__(self, models):
        self.models = models

    def predict(self, X):
        '''Calls predict on each model and
        returns the averaged prediction.

        Parameters
        ----------
        X : array_like
            The input data to be passed to each model
            for prediction. Should be the right shape.
        '''

        preds = [model.predict(X) for model in self.models]
        return np.mean(preds, axis=0)

    def predict_proba(self, X):
        '''Calls predict_proba on each model and
        returns the averaged prediction.

        Parameters
        ----------
        X : array_like
            The input data to be passed to each model
            for prediction. Should be the right shape.
        '''

        preds = [model.predict_proba(X) for model in self.models]
        return np.mean(preds, axis=0)