"""
ABCD_ML.py
====================================
The main file for the ABCD_ML project.
"""
import pandas as pd
import numpy as np
from ABCD_ML.Data_Helpers import get_unique_combo
from ABCD_ML.CV import CV


class ABCD_ML():
    '''The main class used in ABCD_ML project'''

    def __init__(self, eventname='baseline_year_1_arm_1',
                 use_default_subject_ids=True,
                 default_dataset_type='basic',
                 default_na_values=['777', '999'],
                 original_targets_key='targets', low_memory_mode=False,
                 random_state=None, verbose=True):
        '''Main class init

        Parameters
        ----------
        eventname : str or None, optional
            Optional value to provide, specifying to keep certain rows
            when reading data based on the eventname flag.
            As ABCD is a longitudinal study, this flag lets you select only
            one specific time point, or if set to None, will load everything.
            (default = baseline_year_1_arm_1)

        use_default_subject_ids : bool, optional
            Flag to determine the usage of 'default' subject id behavior.
            If set to True, this will convert input NDAR subject ids
            into upper case, with prepended NDAR_ - type format.
            If set to False, then all input subject names must be entered
            explicitly the same, no preprocessing will be done on them.
            (default = True)

        default_dataset_type : {'basic', 'explorer', 'custom'}, optional
            The default dataset_type / file-type to load from.
            Dataset types are,

            - 'basic' : ABCD2p0NDA style, (.txt and tab seperated)

            - 'explorer' : 2.0_ABCD_Data_Explorer style
                 (.csv and comma seperated)

            - 'custom' : A user-defined custom dataset. Right now this is only
                supported as a comma seperated file, with the subject names in
                a column called 'src_subject_id'.

            (default = 'basic')

        default_na_values : list, optional
            Additional values to treat as NaN, by default ABCD specific
            values of '777' and '999' are treated as NaN,
            and those set to default by pandas 'read_csv' function.
            Note: if new values are passed here,
            it will override these default '777' and '999' NaN values.
            (default = ['777', '999'])

        original_targets_key : str, optional
            This parameter refers to the column name / key, that the
            target variable of interest will be stored under. There are not a
            lot of reasons to change this setting, except in the case of
            a naming conflict - or just for further customization.
            (default = 'targets')

        low_memory_mode : bool, optional
            This parameter dictates behavior around loading in data,
            specifically, if `low_memory_mode` is set to True,
            then when loading data from multiple sources, only common
            subjects will be saved as each data source is loaded.
            For comparison, when low memory mode if off, the dropping
            of non-common subjects occurs later. Though regardless of if low
            memory mode is on or off, subjects will be dropped right away
            when exclusions or strat is loaded. Non low memory mode
            behavior is useful when the user wants to try loading different
            data, and doesn't want automatic drops to occur.
            If set to True, individual dataframes self.data, self.covars ect...
            will also be deleted from memory as soon as modeling begins.
            (default = False)

        random_state : int, RandomState instance or None, optional
            The default random state, either as int for a specific seed,
            or if None then the random seed is set by np.random.
            (default = None)

        verbose: bool, optional
            If set to true will display diagnostic and other output during
            dataloading and model training ect... if set to False this output
            will be muted.
            (default = True)
        '''

        # Set class parameters
        self.eventname = eventname
        self.use_default_subject_ids = use_default_subject_ids
        self.default_dataset_type = default_dataset_type
        self.default_na_values = default_na_values
        self.original_targets_key = original_targets_key
        self.low_memory_mode = low_memory_mode
        self.random_state = random_state
        self.verbose = verbose

        # Initialze various variables
        self.data, self.covars = pd.DataFrame(), pd.DataFrame()
        self.targets, self.strat = pd.DataFrame(), pd.DataFrame()
        self.name_map, self.exclusions = {}, set()
        self.covars_encoders, self.targets_encoder = {}, None
        self.strat_encoders = {}
        self.all_data, self.train_subjects = None, None
        self.test_subjects = None
        self.CV = CV()
        self.default_ML_params = {}

        self._print('ABCD_ML object initialized')

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
    from ABCD_ML._Data import (load_name_map,
                               load_data,
                               load_covars,
                               load_targets,
                               load_strat,
                               load_exclusions,
                               clear_name_map,
                               clear_data,
                               clear_covars,
                               clear_targets,
                               clear_strat,
                               clear_exclusions,
                               _load_datasets,
                               _load_dataset,
                               _common_load,
                               _load,
                               _merge_existing,
                               _proc_df,
                               _load_set_of_subjects,
                               _process_subject_name,
                               _drop_na,
                               _filter_by_eventname,
                               _process_new,
                               _prepare_data)

    # Validation / CV functionality
    def define_validation_strategy(self, groups=None,
                                   stratify=None):
        '''Define a validation stratagy to be used during different train/test splits,
        in addition to model selection and model hyperparameter CV.
        See Notes for more info.

        Parameters
        ----------
        groups : str, list or None, optional
            In the case of str input, will assume the str to refer
            to a column key within the loaded strat data,
            and will assign it as a value to preserve groups by
            during any train/test or K-fold splits.
            If a list is passed, then each element should be a str,
            and they will be combined into all unique
            combinations of the elements of the list.
            (default = None)

        stratify : str, list or None, optional
            In the case of str input, will assume the str to refer
            to a column key within the loaded strat data,
            and will assign it as a value to preserve
            distribution of groups by during any train/test or K-fold splits.
            'targets' or whatever the value of self.original_targets_key,
            (self.original_targets_key can just be passed).
            Warning: Passing self.targets_key can lead to error
            specifically when self.targets_key is a list.
            can also be passed in the case of binary/categorical problems.
            If a list is passed, then each element should be a str,
            and they will be combined into all unique combinations of
            the elements of the list.
            (default = None)

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
        e.g., for stratification you can stratify by target
        (the dependent variable to be predicted)
        as well as say sex, though with addition of unique value,
        the size of the smallest unique group decreases.
        '''

        if groups is not None:

            if isinstance(groups, str):
                self.CV = CV(groups=self.strat[groups])
            elif isinstance(groups, list):
                self.CV = CV(groups=get_unique_combo(self.strat, groups))

            self._print('CV defined with group preserving behavior, over',
                        len(np.unique(self.CV.groups)), 'unique values.')

        elif stratify is not None:

            # Proc either one input or a list of multiple to merge
            if isinstance(stratify, str):

                if stratify == self.original_targets_key:
                    self.strat[self.original_targets_key] =\
                        self._get_one_col_targets()

                self.CV = CV(stratify=self.strat[stratify])

            elif isinstance(stratify, list):

                if self.original_targets_key in stratify:
                    self.strat[self.original_targets_key] =\
                        self._get_one_col_targets()

                self.CV = CV(stratify=get_unique_combo(self.strat, stratify))

            self._print('CV defined with stratifying behavior, over',
                        len(np.unique(self.CV.stratify)), 'unique values.')

    def _get_one_col_targets(self):
        '''Helper method that returns targets as one column,
        if orginally multicolumn, then converts back to one column.'''

        try:
            self.targets_key
        except AttributeError:
            print('Targets must be loaded before a validation strategy can',
                  'be defined with targets included...')

        if isinstance(self.targets_key, list):

            encoded = self.targets_encoder[1].inverse_transform(self.targets)
            encoded = np.squeeze(encoded)

            # To preserve subject index, set to col in self.targets
            self.targets[self.original_targets_key] = encoded
            targets = self.targets[self.original_targets_key]

            # Then remove.
            self.targets = self.targets.drop(self.original_targets_key, axis=1)

        else:
            targets = self.targets[self.original_targets_key]

        assert targets.dtype != float, \
            "Stratify by targets can only be used by binary or categorical \
             target types."

        return targets

    def train_test_split(self, test_size=None, test_loc=None,
                         test_subjects=None, random_state=None):
        '''Define the overarching train / test split, highly reccomended.

        test_size: float, int or None, optional
            If float, should be between 0.0 and 1.0 and represent
            the proportion of the dataset to be included in the test split.
            If int, represents the absolute number (or target number) to
            include in the testing group.
            Set to None if using test_loc or test_subjects.
            (default = None)

        test_loc : str, Path or None, optional
            Location of a file to load in test subjects from.
            The file should be formatted as one subject per line.
            (default = None)

        test_subjects : list, set, array-like or None, optional
            An explicit list of subjects to constitute the testing set
            (default = None)

        random_state : int None or 'default', optional
            If using test_size, then can optionally provide a random state, in
            order to be able to recreate an exact test set.
            If set to default, will use the value saved in self.random_state
            (default = 'default')
        '''

        if self.all_data is None:
            self._prepare_data()

        if random_state == 'default':
            random_state = self.random_state

        if test_size is not None:
            self.train_subjects, self.test_subjects = self.CV.train_test_split(
                                 self.all_data.index, test_size, random_state)

        else:
            test_subjects = self._load_set_of_subjects(loc=test_loc,
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
    from ABCD_ML._ML import (set_default_ML_params,
                             Evaluate,
                             Test,
                             _premodel_check,
                             _make_ML_params,
                             _print_model_params,
                             _init_model)

    from ABCD_ML.Models import show_model_types, show_models
    from ABCD_ML.Scorers import show_metrics, show_scorers
    from ABCD_ML.Scalers import show_data_scalers, show_scalers
