'''
ABCD_ML Project

Simple implementation for ensembling regression and classification models
'''

from ABCD_ML.Train_Models import train_regression_model, train_binary_model
import numpy as np


class Ensemble_Model():
    '''Basic ensemble wrapper'''

    def __init__(self,
                 data,
                 score_key,
                 model_names,
                 int_cv=3,
                 problem_type='regression',
                 class_weight='balanced',
                 extra_params={}
                 ):

        self.models = []
        
        if problem_type == 'regresson':
            for name in model_names:
                model = train_regression_model(data, score_key, model_type, int_cv, extra_params)
                self.models.append(model)

        elif problem_type == 'binary':
            for name in model_name:
                model = train_binary_model(data, score_key, model_type, int_cv, class_weight, extra_params)
                self.models.append(model)

    def predict(self, X):
        '''Averages predictions from all models'''

        preds = [model.predict(X) for model in self.models]
        return np.mean(preds, axis=0)

    def predict_proba(self, X):
        '''Averages predicted probabilities from all models'''

        preds = [model.predict_proba(X) for model in self.models]
        return np.mean(preds, axis=0)