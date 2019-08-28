"""
_Validation.py
====================================
Main class extension file for defining validation and train test splits.
"""
import pandas as pd
import numpy as np
from ABCD_ML.Data_Helpers import get_unique_combo
from ABCD_ML.CV import CV


def Define_Validation_Strategy(self, groups=None, stratify=None):
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

    - Random : Just make validation splits randomly.

    - Group Preserving : Make splits that ensure subjects that are\
            part of specific group are all within the same fold\
            e.g., split by family, so that people with the same family id\
            are always a part of the same fold.

    - Stratifying : Make splits such that the distribution of a given \
            group is as equally split between two folds as possible, \
            so simmilar to matched halves or \
            e.g., in a binary or categorical predictive context, \
            splits could be done to ensure roughly equal distribution \
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
            l_e = None
            self.CV = CV(groups=self.strat[groups])

        elif isinstance(groups, list):

            combo, l_e = get_unique_combo(self.strat, groups)
            self.CV = CV(groups=combo)

        self._get_info_on(self.CV.groups, groups, 'groups', l_e)

    elif stratify is not None:

        # Proc either one input or a list of multiple to merge
        if isinstance(stratify, str):

            if stratify == self.original_targets_key:
                self.strat[self.original_targets_key] =\
                    self._get_one_col_targets()

            l_e = None
            self.CV = CV(stratify=self.strat[stratify])

        elif isinstance(stratify, list):

            if self.original_targets_key in stratify:
                self.strat[self.original_targets_key] =\
                    self._get_one_col_targets()

            combo, l_e = get_unique_combo(self.strat, stratify)
            self.CV = CV(stratify=combo)

        self._get_info_on(self.CV.stratify, stratify, 'stratify', l_e)


def Train_Test_Split(self, test_size=None, test_loc=None,
                     test_subjects=None, random_state=None):
    '''Define the overarching train / test split, *highly reccomended*.

    Parameters
    ----------
    test_size : float, int or None, optional
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
                                       name=self.subject_id)
        self.test_subjects = pd.Index(test_subjects,
                                      name=self.subject_id)

    self._print('Performed train/test split, train size:',
                len(self.train_subjects), 'test size: ',
                len(self.test_subjects))


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


def _get_info_on(self, all_vals, col_names, v_type, l_e):

    if v_type == 'group':
        chunk = 'group preserving'
    elif v_type == 'stratify':
        chunk = 'stratifying behavior'

    unique_vals, counts = np.unique(all_vals, return_counts=True)

    self._print('CV defined with ', chunk, ' over',
                len(unique_vals), 'unique values.')

    if v_type == 'stratify':

        if l_e is not None:
            raw = l_e.inverse_transform(unique_vals)
            col_split = np.array([r.split('***')[:-1]
                                    for r in raw]).astype(int)
        else:
            col_names = [col_names]
            col_split = np.expand_dims(unique_vals, axis=-1)

        # Make a display df for each col name
        display_df = pd.DataFrame()
        for n in range(len(col_names)):
            name = col_names[n]

            if name in self.strat_encoders:
                encoder = self.strat_encoders[name]
            else:
                encoder = self.targets_encoder
                if isinstance(encoder, tuple):
                    encoder = encoder[0]

            display_df[name] = encoder.inverse_transform(col_split[:, n])

        display_df['Counts'] = counts
        self._display_df(display_df)