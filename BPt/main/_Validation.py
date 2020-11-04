"""
_Validation.py
====================================
Main class extension file for defining validation and train test splits.
"""
from deslib.dcs.lca import LCA
import pandas as pd
import numpy as np
import os

from ..helpers.Data_Helpers import get_unique_combo_df, reverse_unique_combo_df


def _get_cv(self, cv_params, show=False, show_original=True, return_df=False):

    from ..helpers.CV import CV

    if isinstance(cv_params, CV):
        return cv_params

    train_only_subjects = cv_params.train_only_subjects
    train_only_loc = cv_params.train_only_loc

    if self.all_data is None:
        self._print('Calling Prepare_All_Data()',
                    'to change the default merge behavior',
                    'call it again!')
        self.Prepare_All_Data()

    if isinstance(train_only_subjects, str):
        if train_only_subjects == 'nan':
            train_only_subjects = self.Get_Nan_Subjects()

    subjects_to_use = None
    if train_only_loc is not None:
        subjects_to_use = train_only_loc

    elif train_only_subjects is not None:
        subjects_to_use = train_only_subjects

    train_only = self._get_subjects_to_use(subjects_to_use)
    train_only = np.array(list(train_only))

    groups = cv_params.groups
    stratify = cv_params.stratify

    df = None

    if groups is not None and stratify is not None:
        raise RuntimeError('Warning: BPt does not currently ',
                           'support groups and',
                           'stratify together!')

    if groups is not None:

        if isinstance(groups, str):
            groups = [groups]

        groups = [self.name_map[g] if g in self.name_map else g
                  for g in groups]
        groups = self._add_strat_u_name(groups)

        grp, l_e = get_unique_combo_df(self.all_data, groups)

        cv = CV(groups=grp, train_only=train_only)

        if show:
            # Note, groups has no return df, will set to None
            df = self._get_info_on(cv.groups, groups, 'groups', l_e,
                                   train_only)

    elif stratify is not None:

        if isinstance(stratify, str):
            stratify = [stratify]

        stratify = [self.name_map[s] if s in self.name_map else s
                    for s in stratify]

        targets = self._get_base_targets_names()

        to_stratify = []
        for s in stratify:

            # Check if target - basically want to only add strat name
            #  if not a target.
            if s in targets:

                # Add as is if target
                to_stratify.append(s)

            # Then it is a strat, so make sure to add name if not already added
            else:
                to_stratify.append(self._add_strat_u_name(s))

        # Get the unique combo of passed to_stratify if multiple
        strat, l_e = get_unique_combo_df(self.all_data, to_stratify)

        # Generate the CV object
        cv = CV(stratify=strat, train_only=train_only)

        # Optional if show
        if show:
            df = self._get_info_on(cv.stratify, to_stratify, 'stratify', l_e,
                                   train_only, show_original,
                                   return_df=return_df)

    # If only train only
    elif len(train_only) > 0:
        cv = CV(train_only=train_only)
        self._print(len(train_only), 'Train only subjects defined.')

    else:
        cv = CV()

    if return_df:
        return cv, df

    return cv


def Define_Validation_Strategy(self, cv=None, groups=None, stratify=None,
                               train_only_loc=None, train_only_subjects=None,
                               show=True, show_original=True, return_df=False):
    '''Define a validation strategy to be used during different train/test
    splits, in addition to model selection and model hyperparameter
    cross validation.
    See Notes for more info.

    Note, can also pass a cv params objects here.

    Parameters
    ----------
    cv :  :class:`CV` or None, optional
        If None, then skip, otherwise can pass a :class:`CV`
        object here, and the rest of the parameters will be skipped.

        ::

            default = None

    groups : str, list or None, optional
        In the case of str input, will assume the str to refer
        to a column key within the loaded strat data,
        and will assign it as a value to preserve groups by
        during any train/test or K-fold splits.
        If a list is passed, then each element should be a str,
        and they will be combined into all unique
        combinations of the elements of the list.

        ::

            default = None

    stratify : str, list or None, optional
        In the case of str input, will assume the str to refer
        to a column key within the loaded strat data, or a loaded target col.,
        and will assign it as a value to preserve
        distribution of groups by during any train/test or K-fold splits.
        If a list is passed, then each element should be a str,
        and they will be combined into all unique combinations of
        the elements of the list.

        Any target_cols passed must be categorical or binary, and cannot be
        float. Though you can consider loading in a float target as a strat,
        which will apply a specific k_bins, and then be valid here.

        In the case that you have a loaded strat val with the same name
        as your target, you can distinguish between the two by passing
        either the raw name, e.g., if they are both loaded as 'Sex',
        passing just 'Sex', will try to use the loaded target. If instead
        you want to use your loaded strat val with the same name - you have
        to pass 'Sex' + self.self.strat_u_name (by default this is '_Strat').

        ::

            default = None

    train_only_loc : str, Path or None, optional
        Location of a file to load in train_only subjects,
        where any subject loaded as train_only will be assigned to
        every training fold, and never to a testing fold.
        This file should be formatted as one subject per line.

        You can load from a loc and pass subjects, the subjects
        from each source will be merged.

        This parameter is compatible with groups / stratify.

        ::

            default = None

    train_only_subjects : set, array-like, 'nan', or None, optional
        An explicit list or array-like of train_only subjects, where
        any subject loaded as train_only will be assigned to every training
        fold, and never to a testing fold.

        You can also optionally specify 'nan' as input, which
        will add all subjects with any NaN data to train only.

        If you want to add both all the NaN subjects and custom
        subjects, call :func:`Get_Nan_Subjects` to get all NaN subjects,
        and then merge them yourself with any you want to pass.

        You can load from a loc and pass subjects, the subjects
        from each source will be merged.

        This parameter is compatible with groups / stratify.

        ::

            default = None

    show : bool, optional
        By default, if True, information about the defined validation
        strategy will be shown, including a dataframe if stratify is defined.

        ::

            default = True

    show_original : bool, optional
        By default when you define stratifying behavior, a dataframe will
        be displayed. This param controls if that dataframe shows original
        names, or if False, then it shows the internally used names.

        ::

            default = True

    return_df : bool, optional

        If set to true, then will return as dataframe version of
        the defined validation strategy. Note: this will return
        None in all cases execpt for when stratifying by a variable
        is requested!

        ::

            default = False

    Notes
    ----------
    Validation strategy choices are explained in more detail:

    - Random
        Just make validation splits randomly.

    - Group Preserving
        Make splits that ensure subjects that are
        part of specific group are all within the same fold
        e.g., split by family, so that people with the same family id
        are always a part of the same fold.

    - Stratifying
        Make splits such that the distribution of a given
        group is as equally split between two folds as possible,
        so simmilar to matched halves or
        e.g., in a binary or categorical predictive context,
        splits could be done to ensure roughly equal distribution
        of the dependent class.

    For now, it is possible to define only one overarching strategy
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

    if cv is not None:
        passed_cv = cv

    else:
        from .Params_Classes import CV
        passed_cv = CV(groups=groups, stratify=stratify,
                       train_only_loc=train_only_loc,
                       train_only_subjects=train_only_subjects)

    returned = self._get_cv(passed_cv, show=show,
                            show_original=show_original,
                            return_df=return_df)

    if return_df:
        self.cv, df = returned
    else:
        self.cv = returned
        df = None

    if return_df:
        return df


def Train_Test_Split(self, test_size=None,
                     test_subjects=None, cv='default',
                     random_state='default', test_loc='depreciated',
                     CV='depreciated'):
    '''Define the overarching train / test split, *highly reccomended*.

    Parameters
    ----------
    test_size : float, int or None, optional
        If float, should be between 0.0 and 1.0 and represent
        the proportion of the dataset to be included in the test split.
        If int, represents the absolute number (or target number) to
        include in the testing group.
        Keep as None if using test_subjects.

        ::

            default = None

    test_subjects : :ref:`Subjects`, optional
        Pass in a :ref:`Subjects` (see for more info) formatted input.
        This will define an explicit set of subjects to use as a test set.
        If anything but None is passed here, nothing should be passed to the
        test_size parameter.

        ::

            default = None

    cv : 'default' or :class:`CV`, optional
        If left as default 'default', use the class defined CV
        for the train test split, otherwise can pass custom behavior

        ::

            default = 'default'

    random_state : int None or 'default', optional
        If using test_size, then can optionally provide a random state, in
        order to be able to recreate an exact test set.

        If set to default, will use the value saved in self.random_state,
        (as set in :class:`BPt.BPt_ML` upon class init).

        ::

            default = 'default'

     test_loc : depreciated
        Pass a single str with the test loc to test_subjects instead.

        ::

            default = 'depreciated'

     CV : 'depreciated'
        Switching to passing cv parameter as cv instead of CV.
        For now if CV is passed it will still work as if it were
        passed as cv.

        ::

            default = 'depreciated'

    '''

    if test_loc != 'depreciated':
        print('Test loc is depreciated, please pass it as a single ',
              'string to test_subjects.')
        test_subjects = test_loc

    if CV != 'depreciated':
        print('Warning: Passing CV is depreciated. Please change to',
              'passing as cv instead.')

        # For now, let it still work
        cv = CV

    if cv == 'default':
        cv_obj = self.cv
    else:
        cv_obj = self._get_cv(cv)

    if test_size is None and test_subjects is None:
        test_size = .2

    if self.all_data is None:
        self._print('Calling Prepare_All_Data()',
                    'to change the default merge behavior',
                    'call it again!')
        self.Prepare_All_Data()

    if random_state == 'default':
        random_state = self.random_state

    if test_size is not None:

        _, subjects, train_only =\
            cv_obj.get_train_only(self.all_data.index)

        self._print('Performing split on', len(subjects), 'subjects', end='')

        if test_size == 0:
            self.train_subjects = subjects
            self.test_subjects = []
            self._print('Warning: Test size of 0 passed, all subjects set to',
                        'train.')

        else:

            if len(train_only) > 0:
                self._print(' with', len(train_only), 'considered train only!')
            else:
                self._print('!')

            self._print('random_state:', random_state)
            self._print('Test split size:', test_size)

            self.train_subjects, self.test_subjects = cv_obj.train_test_split(
                                    self.all_data.index, test_size,
                                    random_state)

        self._print()

    else:

        # Load passed subjects
        test_subjects = self._get_subjects_to_use(test_subjects)

        # Take only the overlap of the passed subjects with what is loaded
        test_subjects = [subject for subject in test_subjects
                         if subject in self.all_data.index]

        # Set remaining subjects to train subjects
        train_subjects = [subject for subject in self.all_data.index
                          if subject not in test_subjects]

        # Set the tr test subjects saved within class obj
        self.train_subjects = pd.Index(train_subjects,
                                       name=self.subject_id)
        self.test_subjects = pd.Index(test_subjects,
                                      name=self.subject_id)

    self._print('Performed train test split')
    self._print('Train size:', len(self.train_subjects))
    self._print('Test size: ', len(self.test_subjects))

    # Save train / test subjects if log_dr
    if self.log_dr is not None:

        train_loc = os.path.join(self.exp_log_dr, 'train_subjects.txt')
        test_loc = os.path.join(self.exp_log_dr, 'test_subjects.txt')

        for loc, subjects in zip([train_loc, test_loc], [self.train_subjects,
                                                         self.test_subjects]):
            with open(loc, 'w') as f:
                for subject in subjects:
                    f.write(str(subject) + '\n')


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


def _get_info_on(self, all_vals, col_names, v_type, l_e, train_only,
                 show_original=True, return_df=False):

    if v_type == 'groups':
        chunk = 'group preserving'
    elif v_type == 'stratify':
        chunk = 'stratifying behavior'
    else:
        chunk = 'error'

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

    if v_type == 'stratify':

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
                name = name.replace(self.strat_u_name, '')
            else:
                name = name.replace(self.strat_u_name, '')
                encoder = self.targets_encoders[name]

            if show_original:

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

        # If return_df and strat, return display_df
        if return_df:
            return display_df

    # All other cases, return None
    return None
