from ..main.CV import CVStrategy
from .helpers import save_subjects
import pandas as pd
from .Dataset import _shared_docs
from pandas.util._decorators import doc

_val_docs = {}

_val_docs['size'] = '''size : float, int or None, optional
        | This parameter represents the size of the train/test split
          to apply. If passed a floating type, this parameter
          should be between 0.0 and 1.0, and it will specify the
          proportion / percent of the data to bet set in the train/test split.
          If passed as an integer type, this value will be intrpetted as
          the absolute number of subjects to be set in the train/test split.

        | Note that either this parameter or subjects should be used, not both
          as they define different behaviors. Keep as default = None if using
          subjects.

        ::

            default = None
'''

_val_docs['subjects'] = '''subjects : :ref:`Subjects`, optional
        | This parameter can be optionally used instead of size in
          the case that a specific set of subjects should be used to define
          the split. This argument can accept any valid :ref:`Subjects` style
          input. Explicitly, either this parameter or size should be used,
          not both as they define different behaviors.

        | In the case that additional subjects are specified here, i.e., ones
          not loaded in the current Dataset, they will be simply be ignored,
          and the functional splits set to the overlap of passed subjects
          with loaded subjects.
          valid subjects.

        ::

            default = None
'''

_val_docs['cv_strategy'] = '''cv_strategy : None or :class:`CVStrategy`, optional
        This parameter is only relevant when size is used, i.e., a new split is
        defined (and subjects is not used). In this case,
        an instance of :class:`CVStrategy` defining any validation
        behavior the train/test split should be performed according
        to should be passed - or left as None (the default), which
        will use random splits. This parameter is typically
        used to define behavior like making sure the same distribution
        of target variable is present in both folds, or that
        members of the same family are preserved across folds.

        ::

            default = None
'''

_val_docs['random_state'] = '''random_state : int or None, optional
        This parameter is only relevant when size is used, i.e., a new split is
        defined (and subjects is not used). In this case, this
        parameter represents the random state in which the split should
        be performed according to. Random states allow for
        reproducing the same train/test splits across different
        runs if given the same input Dataset. If left as
        None, the train/test split will be performed with
        just a random random seed (that is to say a different random state
        each time the function is called.)

        ::

            default = None
'''


def _validate_group_key(self, cv_key, name):
    '''Various input validation.'''

    if cv_key not in list(self):
        raise KeyError(f'Passed {cv_key} must be a specific loaded column')

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
        self._validate_group_key(cv_key=group_key, name='groups')

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
        self._validate_group_key(cv_key=strat_key, name='stratify')

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


@doc(**_val_docs, inplace=_shared_docs['inplace'])
def set_test_split(self, size=None, subjects=None,
                   cv_strategy=None, random_state=None, inplace=False):
    '''| Defines a set of subjects to be reserved as test subjects. This
         method includes utilities for either defining a new test split,
         or loading an existing one.

    | This method applies the passed parameters in order define a test set
      which is stored in the current Dataset.

    Parameters
    ----------
    {size}

    {subjects}

    {cv_strategy}

    {random_state}

    {inplace}

    See Also
    ----------
    set_train_split : Apply a train/test split but via specifying which
        subjects are training subjects.
    test_split : Apply a test split returning two separate Train and Test
        Datasets.
    save_test_split : Save the test subjects from a split to a text file.

    Examples
    ---------

    .. ipython:: python

        import BPt as bp

        # Load example data
        data = bp.read_pickle('data/example1.dataset')
        data

        data.set_test_split(size=.6, inplace=True)
        data.train_subjects
        data.test_subjects

    Note that the split is stored in the dataset itself.
    We can also pass specific subjects.

    .. ipython:: python

        data = data.set_test_split(subjects=[0, 1])
        data.train_subjects
        data.test_subjects

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


@doc(**_val_docs, inplace=_shared_docs['inplace'])
def set_train_split(self, size=None, subjects=None,
                    cv_strategy=None, random_state=None, inplace=False):
    '''| Defines a set of subjects to be reserved as train subjects. This
         method includes utilities for either defining a new train split,
         or loading an existing one.

    | This method applies the passed parameters in order define a train set
      which is stored in the current Dataset.

    Parameters
    ----------
    {size}

    {subjects}

    {cv_strategy}

    {random_state}

    {inplace}

    See Also
    ----------
    set_test_split : Apply a train/test split but via specifying which
        subjects are test subjects.
    train_split : Apply a train split returning two separate Train and Test
        Datasets.
    save_train_split : Save the train subjects from a split to a text file.

    Examples
    ---------

    .. ipython:: python

        import BPt as bp

        # Load example data
        data = bp.read_pickle('data/example1.dataset')
        data

        data.set_train_split(size=.6, inplace=True)
        data.train_subjects
        data.test_subjects

    Note that the split is stored in the dataset itself.
    We can also pass specific subjects.

    .. ipython:: python

        data = data.set_train_split(subjects=[0, 1])
        data.train_subjects
        data.test_subjects

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


@doc(**_val_docs)
def test_split(self, size=None, subjects=None,
               cv_strategy=None, random_state=None):
    '''| This method defines and returns a Train and Test Dataset
         based on the passed parameters, allowing for defining
         the test set as a new split or from an existing set of
         subjects.

    | This method's parameters describe how the test set should be generated,
      where the training set is then defined by just which subjects are not in
      the test set.

    Parameters
    ----------
    {size}

    {subjects}

    {cv_strategy}

    {random_state}

    Returns
    -------
    train_data : :class:`Dataset`
        The current :class:`Dataset` as indexed (i.e., only the subjects from)
        the requested training set, as defined by the passed parameters.
        This :class:`Dataset` will have all the same metadata as the original
        :class:`Dataset`, though if changes are made to it, they will not
        influence the original :class:`Dataset`.

    test_data : :class:`Dataset`
        The current :class:`Dataset` as indexed (i.e., only the subjects from)
        the requested test set, as defined by the passed parameters.
        This :class:`Dataset` will have all the same metadata as the original
        :class:`Dataset`, though if changes are made to it, they will not
        influence the original :class:`Dataset`.

    See Also
    ----------
    train_split : Return a train/test split but via specifying which
        subjects are training subjects.
    set_test_split : Apply a test split, but storing the split information
        in the Dataset.
    save_test_split : Save the test subjects from a split to a text file.

    Examples
    ---------

    .. ipython:: python

        import BPt as bp

        # Load example data
        data = bp.read_pickle('data/example1.dataset')
        data

        tr_data, test_data = data.test_split(size=.2)
        tr_data
        test_data

    We can also define a split by passing specific subjects.

    .. ipython:: python

        tr_data, test_data = data.test_split(subjects=[3, 4])
        tr_data
        test_data

    We see that the parameters are used to define which subjects
    are set as test subjects.

    '''

    # Apply split on copy of data
    data = self.set_test_split(size=size, subjects=subjects,
                               cv_strategy=cv_strategy,
                               random_state=random_state,
                               inplace=False)

    # Properly extracts splits as separate Datasets
    return split(data)


@doc(**_val_docs)
def train_split(self, size=None, subjects=None,
                cv_strategy=None, random_state=None):
    '''| This method defines and returns a Train and Test Dataset
         based on the passed parameters, allowing for defining
         the training set as a new split or from an existing set of
         subjects.

    | This method's parameters describe how the training set
      should be generated, where the testing set is then defined by
      just which subjects are not in the training set.

    Parameters
    ----------
    {size}

    {subjects}

    {cv_strategy}

    {random_state}

    Returns
    -------
    train_data : :class:`Dataset`
        The current :class:`Dataset` as indexed (i.e., only the subjects from)
        the requested training set, as defined by the passed parameters.
        This :class:`Dataset` will have all the same metadata as the original
        :class:`Dataset`, though if changes are made to it, they will not
        influence the original :class:`Dataset`.

    test_data : :class:`Dataset`
        The current :class:`Dataset` as indexed (i.e., only the subjects from)
        the requested test set, as defined by the passed parameters.
        This :class:`Dataset` will have all the same metadata as the original
        :class:`Dataset`, though if changes are made to it, they will not
        influence the original :class:`Dataset`.

    See Also
    ----------
    test_split : Return a train/test split but via specifying which
        subjects are test subjects.
    set_train_split : Apply a train split, but storing the split information
        in the Dataset.
    save_train_split : Save the train subjects from a split to a text file.

    Examples
    ---------

    .. ipython:: python

        import BPt as bp

        # Load example data
        data = bp.read_pickle('data/example1.dataset')
        data

        tr_data, test_data = data.train_split(size=.6)
        tr_data
        test_data

        tr_data, test_data = data.train_split(subjects=[0, 1])
        tr_data
        test_data

    '''

    # Apply split on copy of data
    data = self.set_train_split(size=size, subjects=subjects,
                                cv_strategy=cv_strategy,
                                random_state=random_state,
                                inplace=False)

    # Properly extracts splits as separate Datasets
    return split(data)


def save_test_split(self, loc):
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


def save_train_split(self, loc):
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


def split(data):

    # Get the train and test subjects
    train_subjects = data.get_subjects('train')
    test_subjects = data.get_subjects('test')

    # Reset split to None
    data.train_subjects, data.test_subjects = None, None

    # Get as copies - since meta data should be separate
    tr_data = data.loc[train_subjects].copy(deep=False)
    test_data = data.loc[test_subjects].copy(deep=False)

    # Return Datasets as train then test
    return tr_data, test_data
