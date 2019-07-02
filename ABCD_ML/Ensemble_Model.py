'''
ABCD_ML Project

Simple implementation for ensembling regression and classification models
'''
import numpy as np

from ABCD_ML.Train_Models import train_model

class Ensemble_Model():
    '''Basic ensemble wrapper'''

    def __init__(self,
                 problem_type,
                 data,
                 score_key,
                 CV,
                 model_names,
                 int_cv=3,
                 metric='r2',
                 class_weight='balanced',
                 random_state=None,
                 n_jobs=1,
                 extra_params={}
                 ):

        self.problem_type = problem_type
        self.models = []
        
        for model_type in model_names:
            
            model = train_model(problem_type,
                                data,
                                score_key,
                                CV,
                                model_type,
                                int_cv,
                                metric,
                                class_weight,
                                random_state,
                                n_jobs,
                                extra_params)
            
            self.models.append(model)

    def predict(self, X):
        '''Averages predictions from all models'''

        preds = [model.predict(X) for model in self.models]
        return np.mean(preds, axis=0)

    def predict_proba(self, X):
        '''Averages predicted probabilities from all models'''

        preds = [model.predict_proba(X) for model in self.models]
        return np.mean(preds, axis=0)