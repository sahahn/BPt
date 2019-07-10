"""
ABCD_ML.py
====================================
The main file for the ABCD_ML project.
"""
import pandas as pd
from ABCD_ML.Data_Helpers import get_unique_combo
from ABCD_ML.CV import CV


class ABCD_ML():
    '''The main class used in ABCD_ML project'''

    def __init__(self, eventname='baseline_year_1_arm_1',
                 use_default_subject_ids=True,
                 default_na_values=['777', '999'], n_jobs=1,
                 original_score_key='score', verbose=True):
        '''Main class init

        Parameters
        ----------
        eventname : str or None, optional (default = baseline_year_1_arm_1)
            Optional value to provide, specifying to keep certain rows
            when reading data based on the eventname flag.
            As ABCD is a longitudinal study, this flag lets you select only
            one specific time point, or if set to None, will load everything.

        use_default_subject_ids : bool, optional (default = True)
            Flag to determine the usage of 'default' subject id behavior.
            If set to True, this will convert input NDAR subject ids
            into upper case, with prepended NDAR_ - type format.
            If set to False, then all input subject names must be entered
            explicitly the same, no preprocessing will be done on them.

        default_na_values : list, optional (default = ['777', '999'])
            Additional values to treat as NaN, by default ABCD specific
            values of '777' and '999' are treated as NaN,
            and those set to default by Pandas 'read_csv' function.
            Note: if new values are passed here,
            it will override these default '777' and '999' NaN values.

        n_jobs : int, optional (default = 1)
            Number of additionally processors to use during training
            of machine learning models. This default parameter can
            still be overriden if n_jobs is passed in
            extra params in a specific training instance.

        original_score_key : str, optional (default = 'score')
            This parameter refers to the column name / key, that the
            target variable of interest will be stored under. There are not a
            lot of reasons to change this setting, except in the case of
            a naming conflict - or just for further customization.

        verbose: bool, optional (default = True)
            If set to true will display diagnostic and other output during
            dataloading and model training ect... if set to False this output
            will be muted.
        '''

        # Set class parameters
        self.eventname = eventname
        self.use_default_subject_ids = use_default_subject_ids
        self.default_na_values = default_na_values
        self.n_jobs = n_jobs
        self.original_score_key = original_score_key
        self.verbose = verbose

        # Initialze various variables
        self.data, self.covars = [], []
        self.scores, self.strat = [], []
        self.name_map, self.exclusions = {}, set()
        self.covar_encoders, self.score_encoder = {}, None
        self.strat_encoders = {}
        self.all_data, self.train_subjects = None, None
        self.test_subjects = None
        self.CV = CV()

    def _print(self, *args):
        '''Overriding the print function to allow for
        customizable verbosity within class methods

        Parameters
        ----------
        args
            Anything that would be passed to default python print
        '''

        if self.verbose:
            print(*args)

    # Data loader functionality
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

    # Validation / CV functionality
    def define_validation_strategy(self, groups=None,
                                   stratify=None):
        '''Define a validation stratagy to be used during different train/test splits,
        in addition to model selection and model hyperparameter CV.

        Parameters
        ----------
        groups : str, list or None, optional (default = None)
            In the case of str input, will assume the str to refer
            to a column key within the loaded strat data,
            and will assign it as a value to preserve groups by
            during any train/test or K-fold splits.
            If a list is passed, then each element should be a str,
            and they will be combined into all unique
            combinations of the elements of the list.

        stratify : str, list or None, optional (default = None)
            In the case of str input, will assume the str to refer
            to a column key within the loaded strat data,
            and will assign it as a value to preserve
            distribution of groups by during any train/test or K-fold splits.
            'score' or whatever the value of self.original_score_key,
            can also be passed in the case of binary/categorical problems.
            If a list is passed, then each element should be a str,
            and they will be combined into all unique combinations of
            the elements of the list.

        Notes
        ----------
        Validation stratagy choices are explained in more detail:

            Random: Just make splits randomly

            Group Preserving: Make splits that ensure subjects that are
                part of specific group are all within the same fold
                e.g., split by family, so that people with the same family id
                are always a part of the same fold.

            Stratifying: Make splits such that the distribution of a given
                group is as equally split between two folds as possible,
                so simmilar to matched halves or
                e.g., in a binary or categorical predictive context,
                splits could be done to ensure roughly equal distribution
                of the dependent class.

        For now, it is possible to define only one overarching stratagy
        (One could imagine combining group preserving splits
        while also trying to stratify for class,
        but the logistics become more complicated).
        Though, within one strategy it is certainly possible to
        provide multiple values
        e.g., for stratification you can stratify by score
        (the dependent variable to be predicted)
        as well as say sex, though with addition of unique value,
        the size of the smallest unique group decreases.
        '''

        if groups is not None:

            if type(groups) is str:
                self.CV = CV(groups=self.strat[groups])
            elif type(groups) is list:
                self.CV = CV(groups=get_unique_combo(self.strat, groups))

        elif stratify is not None:

            if type(stratify) is str:

                if stratify == self.original_score_key:
                    self.strat[self.score_key] = self.scores[self.score_key]
                    stratify = self.score_key

                self.CV = CV(stratify=self.strat[stratify])

            elif type(stratify) is list:

                if self.original_score_key in list:
                    self.strat[self.score_key] = self.scores[self.score_key]
                    stratify = [self.score_key if s == self.original_score_key
                                else s for s in stratify]

                self.CV = CV(stratify=get_unique_combo(self.strat, stratify))

    def train_test_split(self, test_size=None, test_loc=None,
                         test_subjects=None, random_state=None):
        '''Define the overarching train / test split, highly reccomended.

        test_size: float, int or None, optional (default = None)
            If float, should be between 0.0 and 1.0 and represent
            the proportion of the dataset to be included in the test split.
            If int, represents the absolute number (or target number) to
            include in the testing group.
            Set to None if using test_loc or test_subjects.

        test_loc : str, Path or None, optional (default = None)
            Location of a file to load in test subjects from.
            The file should be formatted as one subject per line.

        test_subjects : list, set, array-like or None, optional (default=None)
            An explicit list of subjects to constitute the testing set

        random_state : int or None, optional (default=None)
            If using test_size, then can optionally provide a random state, in
            order to be able to recreate an exact test set.
        '''

        if self.all_data is None:
            self.prepare_data()

        if test_size is not None:
            self.train_subjects, self.test_subjects = self.CV.train_test_split(
                                 self.all_data.index, test_size, random_state)

        else:
            test_subjects = self.load_set_of_subjects(loc=test_loc,
                                                      subjects=test_subjects)
            train_subjects = [subject for subject in self.all_data.index
                              if subject not in test_subjects]

            self.train_subjects = pd.Index(train_subjects,
                                           name='src_subject_id')
            self.test_subjects = pd.Index(test_subjects,
                                          name='src_subject_id')

        self._print('Performed train/test split, train size:',
                    len(self.train_subjects), 'test size: ',
                    len(self.test_subjects))

    # Machine Learning functionality
    from ABCD_ML._ML import (evaluate_model,
                             test_model,
                             premodel_check,
                             split_data,
                             get_trained_model)

    def show_model_types(self, problem_type=None):
        '''Print out the avaliable machine learning models,
        optionally restricted by problem type.

        Parameters
        ----------
        problem_type : {binary, categorical, regression, None}, optional
        (default = None), where `problem_type` is the underlying ML problem
        '''
        pass

    def show_metrics(self, problem_type=None):
        '''Print out the avaliable metrics/scorers,
        optionally resitricted by problem type

        Parameters
        ----------
        problem_type : {binary, categorical, regression, None}, optional
        (default = None), where `problem_type` is the underlying ML problem
        '''
        pass
