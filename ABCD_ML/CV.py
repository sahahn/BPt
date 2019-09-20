"""
CV.py
====================================
Class for performing train test splits and other cross validation for ABCD_ML
"""
import sklearn.model_selection as MS
import numpy as np


def inds_from_names(original_subjects, subject_splits):

    subject_inds = [[[original_subjects.get_loc(name) for name in s]
                    for s in split] for split in subject_splits]
    return subject_inds


class CV():
    '''Class for performing various cross validation functions'''

    def __init__(self, groups=None, stratify=None, train_only=None):
        '''
        CV class init, for defining split behavior. If groups is None
        and stratify is None, random splits are used.

        Parameters
        ----------
        groups : pandas Series or None, optional
            `groups` should be passed a pandas Series,
            by default passed from ABCD_ML.strat,
            where the Index are subject id's and the values are
            unique ids - where like ids are to be preserved within
            the same fold.
            E.g., GroupKFold

            (default = None)

        stratify : pandas Series or None, optional
            `groups` should be passed a pandas Series,
            by default passed from ABCD_ML.strat,
            where the Index are subject id's and the values are
            unique ids - where like ids are to be as evenly distributed
            as possible between folds.
            E.g., StratifiedKFold

            (default = None)

        train_only : array-like, optional
            An array of subjects that should be forced into the
            training set at every fold, or train test split.

            (default = None)

        See Also
        --------
        ABCD_ML.Define_Validation_Strategy : Main location to define CV object

        '''

        self.groups = groups
        self.stratify = stratify
        self.train_only = train_only

        if self.train_only is not None:
            if len(self.train_only) == 0:
                self.train_only = None

    def train_test_split(self, subjects, test_size=.2, random_state=None):
        '''Define a train test split on input subjects, with a given target test size.

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

        return train_subjects, test_subjects

    def repeated_k_fold(self, subjects, n_repeats, n_splits, random_state=None,
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

    def get_train_only(self, subjects):

        original_subjects = subjects.copy()

        if self.train_only is not None:
            train_only = np.intersect1d(subjects, self.train_only,
                                        assume_unique=True)
            subjects = np.setdiff1d(subjects, self.train_only,
                                    assume_unique=True)
        else:
            train_only = np.array([])

        return original_subjects, subjects, train_only

    def repeated_leave_one_group_out(self, subjects, n_repeats, groups_series,
                                     return_index=False):

        subject_splits = []
        for n in range(n_repeats):
            subject_splits += self.leave_one_group_out(subjects, groups_series,
                                                       return_index)

        return subject_splits

    def leave_one_group_out(self, subjects, groups_series, return_index=False):

        original_subjects, subjects, train_only = self.get_train_only(subjects)

        logo = MS.LeaveOneGroupOut()
        [*inds] = logo.split(subjects, groups=groups_series.loc[subjects])

        subject_splits = [(np.concatenate([subjects[i[0]], train_only]),
                          subjects[i[1]]) for i in inds]

        if return_index:
            return inds_from_names(original_subjects, subject_splits)

        return subject_splits
