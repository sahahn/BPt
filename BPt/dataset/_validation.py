from ..main.CV import CVStrategy
from .helpers import save_subjects
import pandas as pd


def _validate_cv_key(self, cv_key, name):
    '''Various input validation. '''

    # Make sure categorical - scopes should already be checked
    if not self._is_category(cv_key, check_scopes=False):
        raise RuntimeError('Passed ' + name + ': ' + cv_key + ' must '
                           'be of type category! This can be set '
                           'in a number of ways, e.g., via '
                           'binarize, ordinalize, or directly via '
                           'add_scope.')

    # Check to see if any NaN's in this requested column
    nan_subjects = self._get_nan_subjects(col=cv_key)
    if len(nan_subjects) > 0:
        raise RuntimeError('Requested ' + name + ' col: ' + cv_key + ' '
                           'has ' + str(len(nan_subjects)) + ' NaN values.'
                           ' It cannot have any!')


def _proc_cv_strategy(self, cv_params):
    '''This function accepts cv_params and returns
    a CV object.'''

    # Check scopes and roles
    self._check_sr()

    # If None, return base
    if cv_params is None:
        return CVStrategy()

    # @TODO Support scikit-learn style CV directly.

    # If already a CV object, return as is
    if isinstance(cv_params, CVStrategy):
        return cv_params

    # Load train_only_subjects as sorted numpy array
    train_only_subjects = self.get_subjects(cv_params.train_only_subjects,
                                            return_as='flat index')

    # @TODO verbose statement saying X number of train_only_subjects loaded,
    # and also how many of those actually overlap with loaded subjects, and
    # a warning about how non-overlapping ones will be ignored.

    # Set to None if None
    if len(train_only_subjects) == 0:
        train_only_subjects = None

    # Unpack
    group_key = cv_params.groups
    strat_key = cv_params.stratify

    # If groups
    if group_key is not None:

        # Overlapping validations
        self._validate_cv_key(cv_key=group_key, name='groups')

        # Make sure it is non input
        if self.roles[group_key] != 'non input':
            raise RuntimeError('Passed groups: ' + group_key + ' must have '
                               'role = non input. This can be set with '
                               'set_role(...)')

        # Return CV
        return CVStrategy(groups=self[group_key],
                          train_only=train_only_subjects)

    # If stratify
    elif strat_key is not None:

        # Overlapping validations
        self._validate_cv_key(cv_key=strat_key, name='stratify')

        # Make sure it is non input or target
        if self.roles[strat_key] != 'non input' and \
           self.roles[strat_key] != 'target':

            raise RuntimeError('Passed stratify: ' + strat_key + ' must have '
                               'role = non input or target. This can be set '
                               'with set_role(...)')

        # Return CV
        return CVStrategy(stratify=self[strat_key],
                          train_only=train_only_subjects)

    # If None
    else:
        return CVStrategy(train_only=train_only_subjects)


def _validate_split(self, size, subjects):

    self._check_test_subjects()
    self._check_train_subjects()

    if self.train_subjects is not None:
        self._print('Overriding existing train/test split.', level=1)

    if size is None and subjects is None:
        raise TypeError('Either size or subjects must be non-null.')
    elif size is not None and subjects is not None:
        raise TypeError('Only size or subjects can be passed, not both.')


def _finish_split(self):

    # Save in class, pd.Index style
    self.train_subjects = pd.Index(self.train_subjects,
                                   name=self.index.name)
    self.test_subjects = pd.Index(self.test_subjects,
                                  name=self.index.name)

    self._print('Performed train/test split', level=1)
    self._print('Train size:', len(self.train_subjects), level=1)
    self._print('Test size: ', len(self.test_subjects), level=1)


def set_test_split(self, size=None, subjects=None,
                   cv_strategy=None, random_state=None, inplace=False):
    '''Defines a set of subjects to be reserved as test subjects. This
    method includes utilities for either defining a new test split, or loading
    an existing one. See related :func:`save_train_split` and
    :func:`save_test_subjects` or to view
    directly any saved test subjects, you can check self.test_subjects,
    and likewise for train_subjects.

    Parameters
    ----------
    size : float, int or None, optional
        If float, should be between 0.0 and 1.0 and represent
        the proportion of the dataset to be included in the test split.
        If int, represents the absolute number (or target number) to
        include in the testing group.

        Either this parameter or subjects should be used, not both
        as they define different behaviors. Keep as default = None if using
        subjects.

        ::

            default = None

    subjects : :ref:`Subjects`, optional
        If instead of defining a new test split, you wish to load
        an existing set of test subjects, this parameter should be used.
        This argument accepts :ref:`Subjects` (see for more info) formatted
        input. This will define an explicit set of subjects to use
        as the test set.

        In the case that additional subjects are loaded here, i.e., ones
        not loaded in the current Dataset, they will be simply be ignored,
        and the functional test set will constitue the overlap of
        valid subjects.

        Either this parameter or subjects should be used, not both
        as they define different behaviors. Keep as default = None if using
        size.

        ::

            default = None

    cv_strategy : None or :class:`CVStrategy`, optional
        This parameter is only relevant when size is not None,
        and you are defining a new test split. In this case, it
        defines any validation criteria in which the test split should
        adhere to. If left as None, the test split will be be performed
        according to random splits.

        ::

            default = None

    random_state : int or None, optional
        This parameter is only relevant when size is not None,
        and you are defining a new test split.
        In this case, you may provide a random state argument,
        which allows for reproducing the same split.

        If kept as None, will perform the split with a random
        seed.

        ::

            default = None

    inplace : bool, optional
        If True, do operation inplace and return None.

        ::

            default = False
    '''

    if not inplace:
        return self._inplace('set_test_split', locals())

    # Input validation
    self._validate_split(size, subjects)

    # If size is passed
    if size is not None:

        # Check for if size 0
        if size == 0:
            self.test_subjects =\
                self.get_subjects(None, return_as='index')

            # If multi index, set as flat index
            if isinstance(self.index, pd.MultiIndex):
                self.train_subjects = self.index.to_flat_index()
            else:
                self.train_subjects = self.index

            self._print('Warning: Test size of 0 passed, all subjects will be',
                        'considered train subjects.', level=0)

        # Otherwise perform split according to size, cv and random_state
        else:

            # Process the passed cv params
            cv_obj = self._proc_cv_strategy(cv_strategy)

            # Get the overlap of train only subjects from loaded data
            _, valid_subjects, train_only_subjects =\
                cv_obj.get_train_only(self.index)

            # Print info on split
            self._print('Performing test split on:', len(valid_subjects),
                        'subjects', end='', level=1)

            if len(train_only_subjects) > 0:
                self._print(' with:', len(train_only_subjects), 'set as',
                            'train only', end='', level=1)
            self._print('.', level=1)

            self._print('random_state:', random_state, level=1)
            self._print('Test split size:', size, level=1)

            self.train_subjects, self.test_subjects = cv_obj.train_test_split(
                subjects=self.index, test_size=size,
                random_state=random_state, return_index=False)

        self._print(level=1)

    # If subjects is passed
    else:

        # Load all test subjects
        self.test_subjects = self.get_subjects(subjects, return_as='set')

        # Take only the overlap of the passed subjects with what is loaded
        self.test_subjects = [subject for subject in self.test_subjects
                              if subject in self.index]

        # Set remaining subjects to train subjects
        self.train_subjects = [subject for subject in self.index
                               if subject not in self.test_subjects]

    self._finish_split()


def set_train_split(self, size=None, subjects=None,
                    cv_strategy=None, random_state=None, inplace=False):
    '''Defines a set of subjects to be reserved as train subjects.
    This is a variation on function :func:`save_test_split`, where
    both set train and test subjects, but vary on if parameters
    specify how the training set should be defined (this function)
    or how the testing set should be defined.

    This method includes utilities for either defining a new train split,
    or loading an existing one. See related
    :func:`save_train_subjects` or to view
    directly any saved train subjects, you can check self.train_subjects,
    and likewise for test_subjects.

    Parameters
    ----------
    size : float, int or None, optional
        If float, should be between 0.0 and 1.0 and represent
        the proportion of the dataset to be included in the training split.
        If int, represents the absolute number (or target number) to
        include in the training group.

        Either this parameter or subjects should be used, not both
        as they define different behaviors. Keep as default = None if using
        subjects.

        ::

            default = None

    subjects : :ref:`Subjects`, optional
        If instead of defining a new train split, if you wish to load
        an existing set of train subjects, this parameter should be used.
        This argument accepts :ref:`Subjects` (see for more info) formatted
        input. This will define an explicit set of subjects to use
        as the training set.

        In the case that additional subjects are loaded here, i.e., ones
        not loaded in the current Dataset, they will be simply be ignored,
        and the functional test set will constitue the overlap of
        valid subjects.

        Either this parameter or subjects should be used, not both
        as they define different behaviors. Keep as default = None if using
        size.

        ::

            default = None

    cv_strategy : None or :class:`CVStrategy`, optional
        This parameter is only relevant when size is not None,
        and you are defining a new train split. In this case, it
        defines any validation criteria in which the test split should
        adhere to. If left as None, the train split will be be performed
        according to random splits.

        ::

            default = None

    random_state : int or None, optional
        This parameter is only relevant when size is not None,
        and you are defining a new train split.
        In this case, you may provide a random state argument,
        which allows for reproducing the same split.

        If kept as None, will perform the train split with a random
        seed.

        ::

            default = None

    inplace : bool, optional
        If True, do operation inplace and return None.

        ::

            default = False
    '''

    if not inplace:
        return self._inplace('set_train_split', locals())

    # Input validation
    self._validate_split(size, subjects)

    # If size is passed
    if size is not None:

        # Check for if size 0
        if size == 0:
            raise RuntimeError('Train size of 0 not allowed!')

        # Otherwise perform split according to size, cv and random_state
        else:

            # Process the passed cv params
            cv_obj = self._proc_cv_strategy(cv_strategy)

            # Get the overlap of train only subjects from loaded data
            _, valid_subjects, train_only_subjects =\
                cv_obj.get_train_only(self.index)

            # Print info on split
            self._print('Performing train split on:', len(valid_subjects),
                        'subjects', end='', level=1)

            if len(train_only_subjects) > 0:
                self._print(' with:', len(train_only_subjects), 'set as',
                            'train only', end='', level=1)
            self._print('.', level=1)

            self._print('random_state:', random_state, level=1)
            self._print('Train split size:', size, level=1)

            # Convert size to test size
            if isinstance(size, int):
                test_size = len(valid_subjects) - size
            else:
                test_size = 1 - size

            self.train_subjects, self.test_subjects = cv_obj.train_test_split(
                subjects=self.index, test_size=test_size,
                random_state=random_state, return_index=False)

        self._print(level=1)

    # If subjects is passed
    else:

        # Load all train subjects
        self.train_subjects = self.get_subjects(subjects, return_as='set')

        # Take only the overlap of the passed subjects with what is loaded
        self.train_subjects = [subject for subject in self.train_subjects
                               if subject in self.index]

        # Set remaining subjects to test subjects
        self.test_subjects = [subject for subject in self.index
                              if subject not in self.train_subjects]

    self._finish_split()


def save_test_subjects(self, loc):
    '''Saves the currently defined test
    subjects in a text file with one subject / index
    per line.

    Parameters
    ----------
    loc : str or Path
        The location in which to save the test subjects
    '''

    self._check_test_subjects()

    if self.test_subjects is None:
        raise RuntimeError('No train test split defined')

    save_subjects(loc, self.test_subjects)


def save_train_subjects(self, loc):
    '''Saves the currently defined train
    subjects in a text file with one subject / index
    per line.

    Parameters
    ----------
    loc : str or Path
        The location in which to save the train subjects
    '''

    self._check_train_subjects()

    if self.train_subjects is None:
        raise RuntimeError('No train test split defined')

    save_subjects(loc, self.train_subjects)
