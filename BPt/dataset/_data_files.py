import os
import shutil
from joblib import Parallel, delayed
from .data_file import DataFile
from .helpers import (proc_file_input, mp_consol_save, wrap_load_func)
from pandas.util._decorators import doc
from .Dataset import _shared_docs, _sip_docs
import numpy as np
import pandas as pd


def get_file_mapping(self):
    '''This function is used to access the
    up to date file mapping.

    Returns
    --------
    file_mapping : dict
        Return a dictionary with keys as
        integer's loaded in the Dataset referring
        to Data Files.

    See Also
    --------
    to_data_file : Cast existing columns to type Data File.
    add_data_files : Method for adding new data files

    '''

    self._check_file_mapping()
    return self.file_mapping


@doc(load_func=_shared_docs['load_func'], inplace=_shared_docs['inplace'])
def add_data_files(self, files, file_to_subject,
                   load_func=np.load, inplace=False):
    '''This method allows adding columns of type
    'data file' to the Dataset class.

    Parameters
    ----------
    files : dict
        | This argument specifies the files to be loaded as :ref:`data_files`.
          Files must be passed as a python dict where
          each key refers to the name of that feature / column of data files
          to load, and the value is either a list-like of
          str file paths, or a single globbing str which will
          be used to determine the files.

        | In addition to this parameter, you must also pass a
          python function to the file_to_subject param,
          which specifies how to convert from passed
          file path, to a subject name.

    file_to_subject : python function, dict of or 'auto'
        | This parameter represents how the subject name should
          be determined from the passed file paths. This
          parameter can be passed any python function, where
          the first argument on the function takes a full
          file path and returns a subject name.

        | This parameter should be passed as either a single function
          or argument to be used for all columns, or as a dictionary
          corresponding to the passed files dictionary in the case
          that each column requires a different function mapping path
          to subject. If just one function is passed, it will be used
          for to load all dictionary entries. For example:

        | You may also pass the custom str 'auto' to
          specify that the subject name should be the base
          file name with the extension removed. For example
          if the path is '/some/path/subj16.npy' then the auto
          subject will be 'subj16'.

        | In the case that the underlying index is a MultiIndex, this
          function should be designed to return the subject in correct
          tuple form. See Examples below.

    {load_func}

    {inplace}

    See Also
    --------
    to_data_file : Cast existing columns to type Data File.
    get_file_mapping : Returns the raw file mapping.

    Examples
    ---------
    Consider the brief example below for loading two fake subjects,
    with the files parameter.

    ::

        files = dict()
        files['feat1'] = ['f1/subj_0.npy', 'f1/subj_1.npy']
        files['feat2'] = ['f2/subj_0.npy', 'f2/subj_1.npy']

    This could be matched with file_to_subject as:

    ::

        def file_to_subject_func(file):
            subject = file.split('/')[1].replace('.npy', '')
            return subject

        file_to_subject = file_to_subject_func
        # or
        file_to_subject = dict()
        file_to_subject['feat1'] = file_to_subject_func
        file_to_subject['feat2'] = file_to_subject_func

    In this example, subjects are loaded as 'subj_0' and 'subj_1',
    and they have associated loaded data files 'feat1' and 'feat2'.

    Next, we consider an example with fake data.
    In this example we will first generate and save some fake data files.
    These fake files will correspond to left hemisphere vertex files.

    .. ipython:: python

        import numpy as np
        import os

        dr = 'data/fake_surface/'
        os.makedirs(dr, exist_ok=True)

        # 20 subjects each with 10,242 vertex values
        X = np.random.random(size=(20, 10242))

        # Save the data as numpy arrays
        for x in range(len(X)):
            np.save(dr + str(x), X[x])

        os.listdir(dr)[:5]

    Next, we will use add data files to add these to
    a :class:`Dataset`.

    .. ipython:: python

        data = bp.Dataset()
        files = dict()
        files['fake_surface'] = dr + '*' # Add * for file globbing


        data = data.add_data_files(files=files, file_to_subject='auto')
        data.head(5)

    Let's also consider lastly a MultiIndex example:

    ::

        # The underlying dataset is indexed by subject and event
        data.set_index(['subject', 'event'], inplace=True)

        # Only one feature
        files = dict()
        files['feat1'] = ['f1/s0_e0.npy',
                          'f1/s0_e1.npy',
                          'f1/s1_e0.npy',
                          'f1/s1_e1.npy']

        def file_to_subject_func(file):

            # This selects the substring
            # at the last part seperated by the '/'
            # so e.g. the stub, 's0_e0.npy', 's0_e1.npy', etc...
            subj_split = file.split('/')[-1]

            # This removes the .npy from the end, so
            # stubs == 's0_e0', 's0_e1', etc...
            subj_split = subj_split.replace('.npy', '')

            # Set the subject name as the first part
            # and the eventname as the second part
            subj_name = subj_split.split('_')[0]
            event_name = subj_split.split('_')[1]

            # Lastly put it into the correct return style
            # This is tuple style e.g., ('s0', 'e0'), ('s0', 'e1')
            ind = (subj_name, eventname)

            return ind

    '''

    if not inplace:
        return self._inplace('add_data_files', locals())

    # Wrap load func if needed
    wrapped_load_func = wrap_load_func(load_func, _print=self._print)

    # Init if needed
    self._check_file_mapping()

    # Get dict of key to files
    file_series = proc_file_input(files, file_to_subject)

    # For each column
    for file in file_series:

        # For each subject, fill in with Data File
        series = file_series[file]
        self._series_to_data_file(col=file, series=series,
                                  load_func=wrapped_load_func)


@doc(**_sip_docs, load_func=_shared_docs['load_func'])
def to_data_file(self, scope,
                 load_func=np.load,
                 inplace=False):
    '''This method can be used to cast any existing columns
    where the values are file paths, to a data file.

    Parameters
    ----------
    {scope}

    {load_func}

    {inplace}

    Examples
    ----------
    This method can be used as a the primary way to prepare data files.
    We will perform a simple example here.

    .. ipython:: python

        import BPt as bp
        data = bp.Dataset()
        data['files'] = ['loc1.npy', 'loc2.npy']
        data

    We now have a :class:`Dataset`, but out column 'files' is not
    quite ready, as by default it won't know what to do with str.
    To get it to treat it as as a data file we will cast it.

    .. ipython:: python

        data = data.to_data_file('files')
        data

    What's happened here? Now it doesn't show paths anymore, but instead
    shows integers. That's actually the desired behavior though, we
    can check it out in file_mapping.

    .. ipython:: python

        data.file_mapping

    The file_mapping is then used internally with :class:`Loader`
    to load objects on the fly.

    See Also
    --------
    add_data_files : Method for adding new data files
    consolidate_data_files : Merge existing data files into one column.

    '''

    if not inplace:
        return self._inplace('to_data_file', locals())

    # Wrap load func if needed
    wrapped_load_func = wrap_load_func(load_func, _print=self._print)

    # Init if needed
    self._check_file_mapping()

    # Cast to data file
    for col in self.get_cols(scope):
        self._series_to_data_file(col=col, series=self[col],
                                  load_func=wrapped_load_func)


def _series_to_data_file(self, col, series, load_func):

    # Get next file mapping ind
    cnt = self._get_next_ind()

    for subject in series.index:

        # Create data file and add to file mapping
        data_file = DataFile(series[subject], load_func)
        self.file_mapping[cnt] = data_file

        # Replace cnt index in data
        self.at[subject, col] = cnt

        # Increment
        cnt += 1

    # Set scope
    self.add_scope(col, 'data file', inplace=True)


@doc(scope=_shared_docs['scope'])
def consolidate_data_files(self, save_dr, replace_with=None,
                           scope='data file', cast_to=None,
                           clear_existing='fail', n_jobs=-1):
    '''This function is designed as helper to consolidate all
    or a subset of the loaded data files into one column. While this
    removes information, in can provide a speed up in terms of downstream
    loading and reduce the number of files cached when using
    :class:`Loader`.

    This method assumes that the underlying data files
    can be stacked with ::

        np.stack(data, axis=-1)

    After they have been loaded. If this is not the case,
    then this function will break.

    Parameters
    -----------
    save_dr : str or Path
        The file directory in which to
        save the consolidated files. If it
        doesn't exist, then it will be created.

    replace_with : str or None, optional
        By default, if replace_with is left
        as None, then just a saved version of
        the files will be made. Instead,
        if a column name passed as a str is passed,
        then the original data files which were
        consolidated will be deleted, and the new
        consolidated column loaded instead.

        ::

            default = None

    {scope}
        ::

            default = 'data file'

    cast_to : None or numpy dtype, optional
        If not None, then this should be a
        numpy dtype in which the stacked data
        will be cast to before saving.

        ::

            default = None

    clear_existing : bool or 'fail', optional
        If True, then if the save dr already
        has files in it, delete them. If False,
        just overwrite them.

        If 'fail' then if there
        are already files in the save directory,
        raise an error.

        ::

            default = 'fail'

    n_jobs : int, optional
        The number of jobs to use while stacking
        and saving each file.

        If -1, then will try to use all avaliable cpu's.

        ::

            default == -1

    See Also
    --------
    to_data_file : Convert existing column to data file.
    add_data_files : Method for adding new data files
    update_data_file_paths : Update data path saved paths

    '''

    # Make sure file mapping up to date
    self._check_file_mapping()

    # If clear existing and exists
    if clear_existing is True:
        if os.path.exists(save_dr):
            self._print('Removing existing save directory:',
                        str(save_dr), level=0)
            shutil.rmtree(save_dr)

    # Make sure save_dr exists
    os.makedirs(save_dr, exist_ok=True)

    # If Fail.
    if clear_existing == 'fail':
        existing_files = len(os.listdir(save_dr))
        if existing_files > 0:
            raise RuntimeError('The save directory ' +
                               str(save_dr) + ' is not empty.'
                               ' Either change clear_existing or provide '
                               'a new save_dr.')

    # Get cols in scope
    cols = self.get_cols(scope)

    # For each subj / data point
    saved_locs = Parallel(n_jobs=n_jobs)(delayed(mp_consol_save)(
        data_files=[self.file_mapping[self.loc[index, key]]
                    for key in cols],
        index=index, cast_to=cast_to, save_dr=save_dr)
        for index in self.index)

    # If replace with
    if replace_with is not None:

        # Drop existing cols
        self.drop(cols, axis=1, inplace=True)

        # Create new series and add as new col
        self[replace_with] = pd.Series(saved_locs, index=self.index)

        # Cast to data file
        self.to_data_file(scope=replace_with,
                          inplace=True)


def update_data_file_paths(self, old, new):
    '''Go through and update saved file paths within
    the Datasets file mapping attribute.
    This function can be used
    when the underlying location of the data files has changed, or
    perhaps when loading a saved dataset on a different device.

    Note the old and new parameters work the same as those
    in the base python string.replace method.

    Parameters
    -----------
    old : str
        The substring in which to replace every instance found
        in every saved file path with new.

    new : str
        The substring in which to replace old with in
        every substring found.

    See Also
    --------
    to_data_file : Convert existing column to data file.
    add_data_files : Method for adding new data files
    '''

    self._check_file_mapping()

    for file_ind in self.file_mapping:
        self.file_mapping[file_ind].loc =\
            self.file_mapping[file_ind].loc.replace(old, new)


def _get_next_ind(self):

    if len(self.file_mapping) > 0:
        return np.nanmax(list(self.file_mapping.keys())) + 1
    else:
        return 0
