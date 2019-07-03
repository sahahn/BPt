''' 
Main class file for ABCD_ML class, contains most of the data loading functionality
'''
import pandas as pd
from ABCD_ML.Data_Helpers import get_unique_combo
from ABCD_ML.CV import CV

class ABCD_ML():

    def __init__(self,
                 eventname = 'baseline_year_1_arm_1',
                 use_default_subject_ids = True,
                 default_na_values = ['777', '999'],
                 n_jobs = 8,
                 verbose = True
                 ):

        self.eventname = eventname
        self.use_default_subject_ids = use_default_subject_ids
        self.default_na_values = default_na_values
        self.verbose = verbose

        self.data, self.covars = [], []
        self.scores, self.strat = [], []
        self.name_map, self.exclusions = {}, set()
        self.covar_encoders, self.score_encoder, self.strat_encoders = {}, None, {}
        self.original_score_key = 'score'

        self.n_jobs = n_jobs

        self.all_data, self.train_subjects, self.test_subjects = None, None, None
        self.CV = CV()

    def print(self, *args):
        '''Overriding the print function to allow from verbosity'''

        if self.verbose:
            print(*args)

    ### Data loader functionality ###
    from ABCD_ML._Data import (load_name_mapping,
                               load_data,
                               load_custom_data,
                               load_covars,
                               load_scores,
                               load_custom_scores,
                               load_strat_values,
                               load_exclusions,
                               clear_exclusions,
                               common_load,
                               merge_existing,
                               proc_df,
                               load_set_of_subjects,
                               process_subject_name,
                               drop_na,
                               filter_by_eventname,
                               process_new,
                               prepare_data)

    ### Validation / CV functionality ###
    def define_validation_strategy(self,
                                   groups = None,
                                   stratify = None
                                   ):
        
        '''
        Define a validation stratagy to be used during different train/test splits,
        in addition to model selection and model hyperparameter CV. 
           
        In general, these options are:
        
            Random: Just make splits randomly
            Group Preserving: Make splits that ensure subjects that are part of specific group
                are all within the same fold e.g., split by family, so that people with the same family id
                are always a part of the same fold.
            Stratifying: Make splits such that the distribution of a given group is as equally split between
                two folds as possible, so simmilar to matched halves or e.g., in a binary or categorical predictive
                context, splits could be done to ensure roughly equal distribution of the dependent class.

        For now, it is possible to define only one overarching stratagy (One could imagine combining group preserving splits
            while also trying to stratify for class, but the logistics become more complicated).
            Though, within one strategy it is certainly possible to provide multiple values e.g.,
            for stratification you can stratify by score (the dependent variable to be predicted) as well as say sex,
            though with addition of unique value, the size of the smallest unique group decreases.

        By default (if this function is not called) just random.

        groups -- string or list (if merging multiple groups) of loaded strat column names, to preserve by.
        stratify -- string or list (if merging multiple) of loaded strat column names (AND/OR 'score' for binary/categorical),
            to preserve distribution between folds.
        '''

        if groups is not None:

            if type(groups) is str:
                self.CV = CV(groups = self.strat[groups])
            elif type(groups) is list:
                self.CV = CV(groups = get_unique_combo(self.strat, groups))

        elif stratify is not None:
            
            if type(stratify) is str:

                if stratify == 'score':
                    self.strat[self.score_key] = self.scores[self.score_key]
                    stratify = self.score_key

                self.CV = CV(stratify = self.strat[stratify])

            elif type(stratify) is list:

                if 'score' in list:
                    self.strat[self.score_key] = self.scores[self.score_key]
                    stratify = [self.score_key if s == 'score' else s for s in stratify]

                self.CV = CV(stratify = get_unique_combo(self.strat, stratify))
    
    def train_test_split(self,
                         test_size=None,
                         test_loc=None,
                         test_subjects=None,
                         random_state=None):

        '''
        Define the overarching train / test split.
           
        test_size -- If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split.
            If int, represents the absolute number of test groups.
            Set to None if using test_loc or test_subjects.
        test_loc -- Location of file to load in test subjects from.
            File should be formatted as one subject per line.
        test_subjects -- List or set of test subjects to pass in directly.
        random_state -- If using test_size, then can optionally provide a random state
        '''
        
        if self.all_data is None:
            self.prepare_data()

        if test_size is not None:
            self.train_subjects, self.test_subjects = self.CV.train_test_split(self.all_data.index, test_size, random_state)

        else:
            test_subjects = self.load_set_of_subjects(loc=test_loc, subjects=test_subjects)
            train_subjects = [subject for subject in self.all_data.index if subject not in test_subjects]
            self.train_subjects, self.test_subjects = pd.Index(train_subjects, name='src_subject_id'), pd.Index(test_subjects, name='src_subject_id')
        
        self.print('Performed train/test split, train size:', len(self.train_subjects), 'test size: ', len(self.test_subjects))

    ### Machine Learning functionality ###
    from ABCD_ML._ML import (evaluate_model,
                             test_model,
                             premodel_check,
                             split_data,
                             get_trained_model)
                              
    def show_model_types(self, problem_type):
        pass

    def show_metrics(self, problem_type):
        pass

    
    
        
        
    