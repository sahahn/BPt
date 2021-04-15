"""
CV.py
====================================
Class for performing train test splits and other cross validation for BPt
"""
import sklearn.model_selection as MS
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


def get_bpt_cv(cv, dataset):
    '''Helper function used to return BPtCV object from passed
    cv params object and dataset.'''

    # Process the passed cv strategy according to dataset
    cv_strategy = dataset._proc_cv_strategy(cv.cv_strategy)

    # Check for is split corresponds to a leave-out-group
    if isinstance(cv.splits, str):

        if not dataset._is_category(cv.splits, check_scopes=False):
            raise RuntimeError('Passed splits col must have scope category.')

        dataset._check_roles()
        if dataset.roles[cv.splits] != 'non input':
            raise RuntimeError('Passed splits col must be role == non input.')

        # @TODO input validation on choice of leave out variable
        splits_vals = dataset[cv.splits]

    else:
        splits_vals = None

    return BPtCV(splits=cv.splits, n_repeats=cv.n_repeats,
                 cv_strategy=cv_strategy, splits_vals=splits_vals,
                 random_state=cv.random_state, only_fold=cv.only_fold)


class BPtCV(BaseEstimator):

    def __init__(self, splits, n_repeats,
                 cv_strategy, splits_vals,
                 random_state='context', only_fold=None):

        self.splits = splits
        self.n_repeats = n_repeats
        self.cv_strategy = cv_strategy
        self.splits_vals = splits_vals
        self.random_state = random_state
        self.only_fold = only_fold

    def get_cv(self, fit_index, random_state, return_index):

        if self.random_state != 'context':
            random_state = self.random_state

        return self.cv_strategy.get_cv(fit_index,
                                       splits=self.splits,
                                       n_repeats=self.n_repeats,
                                       splits_vals=self.splits_vals,
                                       random_state=random_state,
                                       return_index=return_index,
                                       only_fold=self.only_fold)

    def get_split(self, fit_index, random_state):
        '''Used when there is only one split, returns as index'''

        # Make sure splits is a float
        if not isinstance(self.splits, float):
            raise RuntimeError('splits must be a float to use get_split!')

        if self.random_state != 'context':
            random_state = self.random_state

        return self.cv_strategy.train_test_split(fit_index,
                                                 test_size=self.splits,
                                                 random_state=random_state,
                                                 return_index=True)

    def get_n_repeats(self):

        if self.only_fold is None:
            return self.n_repeats
        return 1


def inds_from_names(original_subjects, subject_splits):

    subject_inds = [[[original_subjects.get_loc(name) for name in s]
                     for s in split] for split in subject_splits]
    return subject_inds


class CVStrategy(BaseEstimator):
    '''Class that holds a CV_Behavior'''

    def __init__(self, groups=None, stratify=None, train_only=None):
        '''
        CV class init, for defining split behavior. If groups is None
        and stratify is None, random splits are used.

        Parameters
        ----------
        groups : pandas Series or None, optional
            `groups` should be passed a pandas Series,
            by default passed from ML.strat,
            where the Index are subject id's and the values are
            unique ids - where like ids are to be preserved within
            the same fold.
            E.g., GroupKFold

            (default = None)

        stratify : pandas Series or None, optional
            `groups` should be passed a pandas Series,
            by default passed from ML.strat,
            where the Index are subject id's and the values are
            unique ids - where like ids are to be as evenly distributed
            as possible between folds.
            E.g., StratifiedKFold

            (default = None)

        train_only : array-like, optional
            An array of subjects that should be forced into the
            training set at every fold, or train test split.

            (default = None)

        '''

        self.groups = groups
        self.stratify = stratify
        self.train_only = train_only

        if self.train_only is not None:
            if len(self.train_only) == 0:
                self.train_only = None

    def __repr__(self):
        return 'CV()'

    def __str__(self):
        return self.__repr__()

    def repeated_train_test_split(self, subjects, n_repeats, test_size=.2,
                                  random_state=None, return_index=False):

        subject_splits = []
        for n in range(n_repeats):

            # If a specific random state is passed
            # Make sure each repeat is provided a different random_state
            if random_state is not None:
                random_state += 1

            subject_splits.append(self.train_test_split(
                subjects,
                test_size=test_size,
                random_state=random_state,
                return_index=return_index))

        return subject_splits

    def train_test_split(self, subjects, test_size=.2,
                         random_state=None, return_index=False):
        '''Define a train test split on input subjects, with a given target
        test size.

        Parameters
        ----------
        subjects : array-like
            `subjects` should be a pandas index or numpy array of subjects.
            They should correspond to any subject indexed groups or stratify.

        test_size : float, int or None, optional
            If float, should be between 0.0 and 1.0 and represent
            the proportion of the dataset to be included in the test split.
            If int, represents the absolute number (or target number) to
            include in the testing group.
            (default = .2)

        random_state : int or None, optional
            Optionally can provide a random state, in
            order to be able to recreate exact splits.
            (default=None)

        Returns
        ----------
        array-like
            The training subjects as computed by the split

        array-like
            The testing subjects as computed by the split
        '''

        original_subjects, subjects, train_only = self.get_train_only(subjects)

        if self.groups is not None:
            splitter = MS.GroupShuffleSplit(n_splits=1, test_size=test_size,
                                            random_state=random_state)
            [*inds] = splitter.split(subjects,
                                     groups=self.groups.loc[subjects])

        elif self.stratify is not None:
            splitter = MS.StratifiedShuffleSplit(n_splits=1,
                                                 test_size=test_size,
                                                 random_state=random_state)
            [*inds] = splitter.split(subjects,
                                     y=self.stratify.loc[subjects])

        else:
            splitter = MS.ShuffleSplit(n_splits=1, test_size=test_size,
                                       random_state=random_state)
            [*inds] = splitter.split(subjects)

        inds = inds[0]

        train_subjects, test_subjects = subjects[inds[0]], subjects[inds[1]]
        train_subjects = np.concatenate([train_subjects, train_only])

        if return_index:
            return ([original_subjects.get_loc(name)
                     for name in train_subjects],
                    [original_subjects.get_loc(name)
                     for name in test_subjects])

        return train_subjects, test_subjects

    def repeated_k_fold(self, subjects, n_repeats,
                        n_splits, random_state=None,
                        return_index=False):
        '''Perform a repeated k-fold with class defined split behavior.
        This function simply calls a repeated version of self.k_fold.

        Parameters
        ----------
        subjects : array-like
            `subjects` should be a pandas index or numpy array of subjects.
            They should correspond to any subject indexed groups or stratify.

        n_repeats : int
            The number of times to repeat the k-fold split.

        n_splits : int
            The number of splits to compute within each repeated split.
            Or rather the k in k-fold.

        random_state : int or None, optional
            Optionally can provide a random state, in
            order to be able to recreate exact splits.
            (default = None)

        return_index : bool, optional
            If true, then instead of returning subject ID's, this function
            will return the numerical index of the subject.
            (default = False)

        Returns
        ----------
        list of tuples of array-like
            This function returns a list,
            where each element in this outer list is another fold.
            E.g., if n_repeats = 2, and n_splits = 2, then this list will
            have a length of 4. Within each list,
            there is a tuple with two elements.
            The first index of the tuple contains the array-like
            list of either subjects or numerical index
            (This depends on the value of return index)
            of the training subjects in that fold.
            The second contains the testing subjects.
        '''

        subject_splits = []
        for n in range(n_repeats):

            # If a specific random state is passed
            # Make sure each repeat is provided a different random_state
            if random_state is not None:
                random_state += 1

            subject_splits += self.k_fold(subjects, n_splits, random_state,
                                          return_index)

        return subject_splits

    def k_fold(self, subjects, n_splits, random_state=None,
               return_index=False):
        '''Perform a k-fold with class defined split behavior.

        Parameters
        ----------
        subjects : array-like
            `subjects` should be a pandas index or numpy array of subjects.
            They should correspond to any subject indexed groups or stratify.

        n_splits : int
            The number of splits to compute within each fold.
            Or rather the k in k-fold.

        random_state : int or None, optional
            Optionally can provide a random state, in
            order to be able to recreate exact splits.
            (default = None)

        return_index : bool, optional
            If true, then instead of returning subject ID's, this function
            will return the numerical index of the subject.
            (default = False)

        Returns
        ----------
        list of tuples of array-like
            This function returns a list,
            where each element in this outer list is another fold.
            E.g., if n_splits = 2, then this list will have a length of 2.
            Within each list, there is a tuple with two elements.
            The first index of the tuple contains the array-like
            list of either subjects or numerical index
            (This depends on the value of return index)
            of the training subjects in that fold.
            The second contains the testing subjects.
        '''

        original_subjects, subjects, train_only = self.get_train_only(subjects)

        # Special implementation for group K fold,
        # just do KFold on unique groups, and recover subjects
        # by group. This assume groups are roughly equal size.
        if self.groups is not None:

            groups = self.groups.loc[subjects]
            unique_groups = np.unique(groups)

            splitter = MS.KFold(n_splits=n_splits, shuffle=True,
                                random_state=random_state)

            [*inds] = splitter.split(unique_groups)

            subject_splits = [(
                np.concatenate([groups.index[groups.isin(unique_groups[i[0]])],
                                train_only]),
                groups.index[groups.isin(unique_groups[i[1]])]) for i in inds]

        else:

            # Stratify behavior just uses stratified k-fold
            if self.stratify is not None:

                splitter = MS.StratifiedKFold(n_splits=n_splits,
                                              shuffle=True,
                                              random_state=random_state)
                [*inds] = splitter.split(subjects,
                                         y=self.stratify.loc[subjects])

            # If no groups or stratify, just use random k-fold
            else:

                splitter = MS.KFold(n_splits=n_splits, shuffle=True,
                                    random_state=random_state)
                [*inds] = splitter.split(subjects)

            # Conv inds within subjects to subject names + train only subjects
            subject_splits = [(np.concatenate([subjects[i[0]], train_only]),
                               subjects[i[1]]) for i in inds]

        if return_index:
            return inds_from_names(original_subjects, subject_splits)

        return subject_splits

    def get_train_only(self, subjects, ignore_by_group=False):

        original_subjects = subjects.copy()

        # Make sure original_subjects is pandas index
        if not isinstance(original_subjects, pd.Index):
            original_subjects = pd.Index(original_subjects)

        if self.train_only is not None:

            # Calculate train only
            train_only = subjects.intersection(self.train_only)

            # If groups, then train only also includes any subjects
            # with the same group ID. Though don't apply for e.g.
            # leave_one_group_out, since could lead to funky behavior.
            if self.groups is not None and not ignore_by_group:

                groups = self.groups.loc[subjects]

                train_only_unique_groups =\
                    np.unique(self.groups.loc[train_only])

                train_only =\
                    groups[groups.isin(train_only_unique_groups)].index

            # Set subjects as the difference between subjects and train only
            subjects = subjects.difference(train_only)

        else:
            train_only = pd.Index(data=[])

        return original_subjects, subjects, train_only

    def repeated_leave_one_group_out(self, subjects, n_repeats, groups_series,
                                     return_index=False):

        subject_splits = []
        for _ in range(n_repeats):
            subject_splits += self.leave_one_group_out(subjects, groups_series,
                                                       return_index)

        return subject_splits

    def leave_one_group_out(self, subjects, groups_series, return_index=False):

        original_subjects, subjects, train_only =\
            self.get_train_only(subjects, ignore_by_group=True)

        logo = MS.LeaveOneGroupOut()
        [*inds] = logo.split(subjects, groups=groups_series.loc[subjects])

        subject_splits = [(np.concatenate([subjects[i[0]], train_only]),
                           subjects[i[1]]) for i in inds]

        if return_index:
            return inds_from_names(original_subjects, subject_splits)

        return subject_splits

    def get_num_groups(self, subjects, groups_series):
        '''Func to get number of leave one out groups'''

        _, subjects, _ =\
            self.get_train_only(subjects, ignore_by_group=True)
        groups = groups_series.loc[subjects]

        return len(np.unique(groups))

    def get_cv(self, fit_index, splits, n_repeats,
               splits_vals=None, random_state=None,
               return_index=False, only_fold=None):
        '''Always return as list of tuples'''

        if return_index == 'both':
            no_index = self.get_cv(fit_index, splits, n_repeats,
                                   splits_vals=splits_vals,
                                   random_state=random_state,
                                   return_index=False)

            index = self.get_cv(fit_index, splits, n_repeats,
                                splits_vals=splits_vals,
                                random_state=random_state,
                                return_index=True)

            return no_index, index

        # If split_vals passed, then by group
        if splits_vals is not None:

            cv_inds =\
                self.repeated_leave_one_group_out(fit_index,
                                                  n_repeats=n_repeats,
                                                  groups_series=splits_vals,
                                                  return_index=return_index)

        # K-fold is splits is an int
        elif isinstance(splits, int):

            cv_inds =\
                self.repeated_k_fold(fit_index, n_repeats,
                                     n_splits=splits,
                                     random_state=random_state,
                                     return_index=return_index)

        # Otherwise, as train test splits
        else:

            cv_inds =\
                self.repeated_train_test_split(fit_index, n_repeats,
                                               test_size=splits,
                                               random_state=random_state,
                                               return_index=return_index)

        return proc_only_fold(cv_inds, only_fold)


def proc_only_fold(cv_inds, only_fold):

    if isinstance(only_fold, int):
        only_fold = [only_fold]

    if only_fold is None:
        return cv_inds

    elif isinstance(only_fold, list):

        cv_subset_inds = []

        for of in only_fold:

            try:
                cv_subset_inds.append(cv_inds[of])
            except IndexError:
                raise IndexError(f'Padded only_fold={of} is out ' +
                                 'of range for the generated CV folds.')

        return cv_subset_inds

    raise RuntimeError(f'only_fold={only_fold} is invalid.')
