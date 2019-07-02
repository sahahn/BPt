'''
ABCD_ML Project

Main class extension file for the Machine Learning functionality
'''
import numpy as np

from ABCD_ML.Ensemble_Model import Ensemble_Model
from ABCD_ML.Train_Models import train_model
from ABCD_ML.ML_Helpers import scale_data, compute_macro_micro
from ABCD_ML.Scoring import get_score

def evaluate_model(self,
                   problem_type,
                   model_type = 'linear',
                   data_scaler = 'standard',
                   n_splits = 3,
                   n_repeats = 2,
                   int_cv = 3,
                   metric = 'r2', 
                   random_state = None,
                   class_weight = 'balanced',
                   extra_params = {}
                   ):

    #Perform pre-modeling data check
    self.premodel_check()
    
    #Setup the desired splits, using the gloablly defined train subjects
    subject_splits = self.CV.repeated_k_fold(subjects = self.train_subjects,
                                             n_repeats = n_repeats,
                                             n_splits = n_splits,
                                             random_state = random_state)

    scores = []

    #For each split, test the model
    for train_subjects, test_subjects in subject_splits:

        score = self.test_model(problem_type = problem_type,
                                train_subjects = train_subjects,
                                test_subjects = test_subjects,
                                model_type = model_type,
                                data_scaler = data_scaler,
                                int_cv = int_cv,
                                metric = metric,
                                random_state = random_state,
                                class_weight = class_weight,
                                return_model = False,
                                extra_params = extra_params)

        scores.append(score)

    #Return the computed macro and micro mean and stds
    return compute_macro_micro(scores, n_repeats, n_splits)

def test_model(self,
               problem_type,
               train_subjects = None,
               test_subjects = None,
               model_type = 'linear',
               data_scaler = 'standard',
               int_cv = 3,
               metric = 'r2',
               random_state = None,
               class_weight = 'balanced',
               return_model = False,
               extra_params = {}
               ):

    #Split the data to train/test
    train_data, test_data = self.split_data(train_subjects, test_subjects)
    
    #If passed a data scaler, scale and transform the data
    train_data, test_data = scale_data(train_data, test_data, data_scaler, self.data_keys, extra_params)

    #Create a trained model based on the provided parameters
    model = self.get_trained_model(problem_type = problem_type,
                                   data = train_data,
                                   model_type = model_type,
                                   int_cv = int_cv,
                                   metric = metric,
                                   class_weight = class_weight,
                                   random_state = random_state,
                                   extra_params = extra_params)

    #Compute the score of the trained model on the testing data
    score = get_score(problem_type, model, test_data, self.score_key, metric)

    if return_model:
        return score, model
    
    return score

def premodel_check(self):

    if self.all_data is None:
        self.prepare_data()
    
    if self.train_subjects is None:
        print('No train-test set defined! Performing one automatically with default split =.25')
        print('If no test set is intentional, just called train_test_split(test_size=0)')
        self.train_test_split(test_size=.25)

def split_data(self,
               train_subjects,
               test_subjects
               ):

    '''Function to split train and test data / check for setting to global train/test subjects'''

    if train_subjects is None:
        train_subjects = self.train_subjects
    if test_subjects is None:
        test_subjects = self.test_subjects

    train_data = self.all_data.loc[train_subjects]
    test_data = self.all_data.loc[test_subjects]

    return train_data, test_data

def get_trained_model(self,
                      problem_type,
                      data,
                      model_type,
                      int_cv,
                      metric,
                      class_weight = 'balanced',
                      random_state = None,
                      extra_params = {}
                      ):

    '''Helpfer function for training either an ensemble or one single model'''

    model_params = {'problem_type': problem_type,
                    'data' : data,
                    'score_key' : self.score_key,
                    'CV' : self.CV,
                    'model_type' : model_type,
                    'int_cv' : int_cv,
                    'metric' : metric,
                    'class_weight' : class_weight,
                    'random_state' : random_state,
                    'n_jobs' : self.n_jobs,
                    'extra_params' : extra_params}
            
    if type(model_type) == list:
        model = Ensemble_Model(**model_params)
    else:
        model = train_model(**model_params)






























