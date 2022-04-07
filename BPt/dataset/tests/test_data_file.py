import numpy as np
from ..data_file import DataFile, mp_single_load, load_data_file_proxy
import tempfile
import pandas as pd
import os


def get_fake_data_file():

    temp_dr = tempfile.gettempdir()
    file_loc = os.path.join(temp_dr, 'fake.npy')

    data = np.arange(1, 10)
    np.save(file_loc, data)

    df = DataFile(loc=file_loc, load_func=np.load)

    return df


def test_Data_File():

    df = get_fake_data_file()
    assert df.loc.endswith('fake.npy')

    data = df.load()
    assert len(data) == 9
    assert np.max(data) == 9

    # Make sure equality based on file path works
    df2 = get_fake_data_file()
    assert df == df2

    # Make sure hash works
    assert hash(df) == hash(df2)


def test_Data_File_repr():

    df = get_fake_data_file()

    assert repr(df).startswith("DataFile(loc=")
    assert str(df).startswith("DataFile(loc=")


def test_df_sort():

    df = get_fake_data_file()

    assert not df < df


def test_mp_single_load():

    df = get_fake_data_file()
    files = [df, df]

    proxy = mp_single_load(files=files, reduce_func=np.max)
    assert all(proxy == 9)


def test_load_data_file_proxy():

    df = get_fake_data_file()

    # Fake file mapping
    file_mapping = {}
    for i in range(10):
        file_mapping[i] = df

    values = pd.Series(np.arange(10))

    proxy =\
        load_data_file_proxy(values, reduce_func=np.min,
                             file_mapping=file_mapping, n_jobs=1)
    assert proxy.sum() == 10
    assert proxy[0] == 1
    assert proxy[9] == 1

    # N jobs == 2
    proxy =\
        load_data_file_proxy(values, reduce_func=np.min,
                             file_mapping=file_mapping, n_jobs=2)
    assert proxy.sum() == 10
    assert proxy[0] == 1
    assert proxy[9] == 1
