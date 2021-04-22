from ..data_file_type import DataFileDtype
from ..data_file import DataFile
from ..Dataset import Dataset
import numpy as np
import tempfile
import os
import pandas as pd


def test_basic():

    temp_dr = tempfile.gettempdir()

    data_files = []
    for i in range(3):
        file_loc = os.path.join(temp_dr, 'fake' + str(i) + '.npy')
        data = np.arange(1, 10)
        np.save(file_loc, data)
        data_files.append(DataFile(loc=file_loc, load_func=np.load))

    data_files_series = pd.Series(data_files, dtype='file')
    assert data_files_series.dtype.name == 'file'

    data = Dataset([1, 2, 3], columns=['1'])
    data['data'] = data_files
    data['data2'] = data['data'].astype('file')
    assert data['data2'].dtype.name == 'file'

    data = Dataset([1, 2, 3], columns=['1'])
    data['data'] = data_files_series
    assert data['data'].dtype.name == 'file'

    # Clean up
    for i in range(3):
        file_loc = os.path.join(temp_dr, 'fake' + str(i) + '.npy')
        os.remove(file_loc)
