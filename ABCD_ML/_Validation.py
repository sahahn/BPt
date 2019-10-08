"""
_Validation.py
====================================
Main class extension file for defining validation and train test splits.
"""
import pandas as pd
import numpy as np
import os
from ABCD_ML.Data_Helpers import get_unique_combo_df, reverse_unique_combo_df
from ABCD_ML.CV import CV


def Define_Validation_Strategy(self, groups=None, stratify=None,
                               train_only_loc=None, train_only_subjects=None,
                               show_original=True):
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
        to a column key within the loaded strat data, or a loaded target col.,
        and will assign it as a value to preserve
        distribution of groups by during any train/test or K-fold splits.
        If a list is passed, then each element should be a str,
        and they will be combined into all unique combinations of
        the elements of the list.

        If any targets cols are passed, they should be categorical or binary
        compatible!

        (default = None)

    train_only_loc : str, Path or None, optional
        Location of a file to load in train_only subjects,
        where any subject loaded as train_only will be assigned to
        every training fold, and never to a testing fold.
        This file should be formatted as one subject per line.

        This parameter is compatible with groups / stratify.

        (default = None)

    train_only_subjects : list, set, array-like or None, optional
        An explicit list or array-like of train_only subjects, where
        any subject loaded as train_only will be assigned to every training
        fold, and never to a testing fold.

        This parameter is compatible with groups / stratify.

        (default = None)

    show_original : bool, optional
        By default when you define stratifying behavior, a dataframe will
        be displayed. This param controls if that dataframe shows original
        names, or if False, then it shows the internally used names.

        (default = True)

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

    train_only =\
        self._load_set_of_subjects(loc=train_only_loc,
                                   subjects=train_only_subjects)
    train_only = np.array(list(train_only))

    if groups is not None and stratify is not None:
        print('Warning: ABCD_ML does not currently support groups and',
              'stratify together!')

    if groups is not None:
        groups = self._add_strat_u_name(groups)

        if isinstance(groups, str):
            groups = [groups]

        grp, l_e = get_unique_combo_df(self.strat, groups)

        self.CV = CV(groups=grp, train_only=train_only)
        self._get_info_on(self.CV.groups, groups, 'groups', l_e, train_only)

    elif stratify is not None:

        if isinstance(stratify, str):
            stratify = [stratify]

        # Check if any target keys passed
        targets = self._get_base_targets_names()
        for target in targets:
            if target in stratify:
                self.strat[target + self.strat_u_name] =\
                    self._get_one_col_targets(target)

        stratify = self._add_strat_u_name(stratify)
        strat, l_e = get_unique_combo_df(self.strat, stratify)

        self.CV = CV(stratify=strat, train_only=train_only)
        self._get_info_on(self.CV.stratify, stratify, 'stratify', l_e,
                          train_only, show_original)

        # Now drop any loaded targets from strat
        for target in targets:

            strat_target_name = target + self.strat_u_name
            if strat_target_name in self.strat:
                self.strat = self.strat.drop(strat_target_name, axis=1)

    # If only train only
    elif len(train_only) > 0:
        self.CV = CV(train_only=train_only)
        self._print(len(train_only), 'Train only subjects defined.')

    else:
        self._print('No params passed, nothing done.')


def Train_Test_Split(self, test_size=None, test_loc=None,
                     test_subjects=None, random_state='default'):
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

    if self.log_dr is not None:

        train_loc = os.path.join(self.exp_log_dr, 'train_subjects.txt')
        test_loc = os.path.join(self.exp_log_dr, 'test_subjects.txt')

        for loc, subjects in zip([train_loc, test_loc], [self.train_subjects,
                                                         self.test_subjects]):
            with open(loc, 'w') as f:
                for subject in subjects:
                    f.write(subject + '\n')


def _add_strat_u_name(self, in_vals):

    if in_vals is None:
        return None

    if isinstance(in_vals, str):
        if self.strat_u_name not in in_vals:
            new_vals = in_vals + self.strat_u_name
        else:
            new_vals = in_vals
    else:
        new_vals = []

        for val in in_vals:
            new_vals.append(self._add_strat_u_name(val))

    return new_vals


def _get_one_col_targets(self, key):
    '''Helper method that returns a target col as one col,
    if orginally multicolumn, then converts back to one column.
    key is the base target name!
    '''

    try:
        len(self.targets_keys) > 0
    except AttributeError:
        raise AttributeError('Targets must be loaded before a validation',
                             'strategy can',
                             'be defined with targets included!')

    targets_key = self._get_targets_key(key)

    if isinstance(targets_key, list):

        # Reverse encoding
        encoder = self.targets_encoders[key][1]
        encoded = encoder.inverse_transform(self.targets[targets_key])
        encoded = np.squeeze(encoded)

        # To preserve subject index, set to a col in self.targets, then remove
        self.targets[key] = encoded
        targets_one_col = self.targets[key]
        self.targets = self.targets.drop(key, axis=1)

    else:
        targets_one_col = self.targets[key]

    if targets_one_col.dtype == float:
        raise TypeError('Stratify by targets can only be used by binary',
                        'or categorical target types!')

    return targets_one_col


def _get_info_on(self, all_vals, col_names, v_type, l_e, train_only,
                 show_original=True):

    if v_type == 'groups':
        chunk = 'group preserving'
    elif v_type == 'stratify':
        chunk = 'stratifying behavior'

    if len(train_only) > 0:
        self._print(len(train_only), 'Train only subjects defined.')
        self._print('Those subjects are excluded from the below stats!')
        self._print()

        non_train_only = np.setdiff1d(all_vals.index, train_only,
                                      assume_unique=True)
        all_vals = all_vals.loc[non_train_only]

    unique_vals, counts = np.unique(all_vals, return_counts=True)

    self._print('CV defined with', chunk, 'over',
                len(unique_vals), 'unique values.')

    if self.verbose and v_type == 'stratify':

        if l_e is not None:
            col_split = reverse_unique_combo_df(unique_vals, l_e)

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
                name = name.replace(self.strat_u_name, '')
                encoder = self.targets_encoders[name]

            if show_original:

                if isinstance(encoder, tuple):
                    encoder = encoder[0]

                if isinstance(encoder, dict):
                    display_df[name] = [encoder[v] for v in col_split[:, n]]

                else:

                    try:
                        display_df[name] =\
                            encoder.inverse_transform(col_split[:, n])
                    except ValueError:
                        display_df[name] =\
                            np.squeeze(encoder.inverse_transform(
                                    col_split[:, n].reshape(-1, 1)))

            else:
                display_df[name] = col_split[:, n]

        display_df['Counts'] = counts
        self._display_df(display_df)
