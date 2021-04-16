import pandas as pd
from pandas.util._decorators import doc
from .Dataset import _shared_docs
from .helpers import (base_load_subjects)
from ..util import conv_to_list
from ..main.input_operations import Intersection, ValueSubset
from itertools import product


def _apply_only_level(self, subjects, only_level):

    # If only level not None and MultiIndex - only keep
    if isinstance(self.index, pd.MultiIndex) and only_level is not None:
        drop_names = list(self.index.names)
        drop_names.remove(self.index.names[only_level])
        subjects = subjects.droplevel(drop_names)

    return set(list(subjects))


def _get_nan_loaded_subjects(self, only_level):

    # Get nan subjects
    nan_subjects = self[pd.isnull(self[:]).any(axis=1)].index

    # Apply only level + convert to sat
    nan_subjects = self._apply_only_level(nan_subjects, only_level)

    return nan_subjects


def _get_value_subset_loaded_subjects(self, subjects, only_level):

    if subjects.name not in list(self):
        raise KeyError('Passed ValueSubset name: ' +
                       repr(subjects.name) +
                       ' is not loaded')

    # Get the relevant series
    data = self.get_values(subjects.name, dropna=False,
                           decode_values=subjects.decode_values)

    # Extract the values as list
    values = conv_to_list(subjects.values)

    # Get subjects by values
    loaded_subjects = data[data.isin(values)].index

    # Apply only level + convert to set
    loaded_subjects =\
        self._apply_only_level(loaded_subjects, only_level)

    return loaded_subjects


def _get_base_loaded_subjects(self, subjects, only_level):

    loaded_subjects = base_load_subjects(subjects)

    # In multi-index case
    if isinstance(self.index, pd.MultiIndex):

        # If want multi-index returned
        if only_level is None:
            loaded_subjects = set(self.loc[loaded_subjects].index)

    # Non multi-index case try to cast to correct type
    else:

        dtype = self.index.dtype.name
        if 'int' in dtype:
            loaded_subjects = {int(s) for s in loaded_subjects}
        elif 'float' in dtype:
            loaded_subjects = {float(s) for s in loaded_subjects}
        elif dtype == 'object':
            loaded_subjects = {str(s) for s in loaded_subjects}
        else:
            raise RuntimeError('Index data type:' + dtype + ' '
                               'is not currently supported!')

    return loaded_subjects


def _return_subjects_as(self, subjects, return_as, only_level):

    if return_as == 'set':
        return subjects

    # If not set, treat as pandas index or flat index case
    subjects = sorted(list(subjects))

    # If not multi index, just cast and return
    if not isinstance(self.index, pd.MultiIndex):
        return pd.Index(data=subjects, name=self.index.name)

    # Multi index case, with only level
    if only_level is not None:
        return pd.Index(data=subjects,
                        name=self.index.names[only_level])

    # RConvert to multi index
    subjects = list(map(tuple, subjects))
    multi_index = pd.MultiIndex.from_tuples(
        subjects, names=self.index.names)

    if return_as == 'flat index':
        return multi_index.to_flat_index()
    else:
        return multi_index


@doc(subjects=_shared_docs['subjects'])
def get_subjects(self, subjects, return_as='set', only_level=None):
    '''Method to get a set of subjects, from
    a set of already loaded ones, or from a saved location.

    Parameters
    -----------
    {subjects}

    return_as : ['set', 'index', 'flat index'], optional

        - 'set'
            Return as set of subjects.

        - 'index'
            Return as sorted :class:`pandas.Index`,
            or if underlying multi-index as a
            :class:`pandas.MultiIndex`.

        - 'flat index'
            Return as sorted :class:`pandas.Index`,
            note that if not an underlying multi-index
            this will return the same result as
            'index', when MultiIndex, will return the index as a flat
            Index of tuples.

    only_level : int or None, optional
        This parameter is only relevant when the
        underlying index is a MultiIndex.

        Note: this param is not relevant
        when using special tuple style input for subjects.

        ::

            default = None

    Returns
    ----------
    subjects : set, :class:`pandas.Index` or :class:`pandas.MultiIndex`
        Based on value of return_as, returns as
        a set of subjects, sorted :class:`pandas.Index`,
        sorted :class:`pandas.MultiIndex` or flattened and
        sorted :class:`pandas.Index` representing a
        :class:`pandas.MultiIndex`.

    '''

    # input validation
    if return_as not in ['set', 'index', 'flat index']:
        raise TypeError('Invalid parameter passed to return as!')

    # Check for passed intersection case
    if isinstance(subjects, Intersection):

        subjects_list =\
            [self.get_subjects(s, return_as='set', only_level=only_level)
                for s in subjects]

        loaded_subjects = set.intersection(*subjects_list)

        # Return as requested
        return self._return_subjects_as(loaded_subjects,
                                        return_as=return_as,
                                        only_level=only_level)

    if isinstance(subjects, tuple):

        if len(subjects) != len(self.index.names):
            raise RuntimeError('Passed special tuple must match the '
                               'number of MultiIndex levels.')

        # Proc each ind seperately
        inds = []
        for i, subject_arg in enumerate(subjects):
            inds.append(self.get_subjects(subject_arg,
                                          return_as='set',
                                          only_level=i))
        # Create set of sets
        loaded_subjects = set(product(*inds))

        # Return as requested, note only_level fixed as None
        return self._return_subjects_as(loaded_subjects,
                                        return_as=return_as,
                                        only_level=None)

    # Compute all subjects at correct only level
    all_subjects = self._apply_only_level(self.index, only_level)

    # Check if None
    if subjects is None:
        loaded_subjects = set()

    # Check for special keywords
    elif isinstance(subjects, str) and subjects == 'nan':
        loaded_subjects =\
            self._get_nan_loaded_subjects(only_level=only_level)

    elif isinstance(subjects, str) and subjects in ['all', 'default']:
        loaded_subjects = all_subjects

    elif isinstance(subjects, str) and subjects == 'train':
        if not hasattr(self, 'train_subjects') or self.train_subjects is None:
            raise RuntimeError('Train subjects undefined')
        loaded_subjects =\
            self._apply_only_level(self.train_subjects, only_level)

    elif isinstance(subjects, str) and subjects == 'test':
        if not hasattr(self, 'test_subjects') or self.test_subjects is None:
            raise RuntimeError('Test subjects undefined')
        loaded_subjects =\
            self._apply_only_level(self.test_subjects, only_level)

    # Check for Value Subset or Values Subset
    elif isinstance(subjects, ValueSubset):
        loaded_subjects =\
            self._get_value_subset_loaded_subjects(subjects,
                                                   only_level=only_level)
    else:
        loaded_subjects =\
            self._get_base_loaded_subjects(subjects, only_level=only_level)

    # loaded_subjects are a set here, though still need to only
    # get the overlap of actually loaded subjects here
    loaded_subjects.intersection_update(all_subjects)

    # Return based on return as value
    return self._return_subjects_as(loaded_subjects, return_as,
                                    only_level=only_level)
