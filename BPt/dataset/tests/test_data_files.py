import shutil
from .datasets import get_fake_dataset, get_fake_multi_index_dataset
import numpy as np
import tempfile
import os
import pytest


def test_add_data_files():

    df = get_fake_dataset()

    def file_to_subject(i):
        return int(i.split('_')[-1])

    files = {'q': ['a_0', 'b_1', 'c_2']}

    df = df.add_data_files(files=files,
                           file_to_subject=file_to_subject,
                           load_func=np.load)

    assert len(df['q']) == 3
    assert df.loc[2, 'q'] == 2

    assert 'a_0' in df.file_mapping[0].loc
    assert 'b_1' in df.file_mapping[1].loc
    assert 'c_2' in df.file_mapping[2].loc

    # Test in place
    df = get_fake_dataset()
    df.add_data_files(files=files,
                      file_to_subject=file_to_subject,
                      load_func=np.load, inplace=True)
    assert len(df['q']) == 3
    assert df.loc[2, 'q'] == 2
    assert 'a_0' in df.file_mapping[0].loc
    assert 'b_1' in df.file_mapping[1].loc
    assert 'c_2' in df.file_mapping[2].loc


def setup_datafiles():

    df = get_fake_dataset()

    def file_to_subject(i):

        if '/' in i:
            return int(i.split('/')[-1].replace('.npy', ''))

        # Windows version
        else:
            return int(i.split('\\')[-1].replace('.npy', ''))

    temp_dr = tempfile.gettempdir()
    locs = []
    for i in range(3):
        loc = os.path.join(temp_dr, str(i) + '.npy')
        locs.append(loc)

        data = np.zeros(shape=2)
        data[:] = i
        np.save(loc, data)

    files = {'data_files': locs}

    # Test add data files
    df = df.add_data_files(files=files,
                           file_to_subject=file_to_subject,
                           load_func=np.load)

    return df


def test_data_files_consolidate():

    df = setup_datafiles()
    assert df.file_mapping[0].load().shape == (2,)

    df['data_files2'] = df['data_files'].copy()
    df.add_scope('data_files2', 'data file', inplace=True)

    temp_dr = tempfile.gettempdir()
    save_dr = os.path.join(temp_dr, 'save_consol')

    if os.path.exists(save_dr):
        shutil.rmtree(save_dr)

    df.consolidate_data_files(save_dr, replace_with='test',
                              scope='data file', cast_to=None,
                              clear_existing='fail', n_jobs=-1)

    assert len(df.file_mapping) == 3
    assert df.shape == (3, 4)
    assert df.file_mapping[0].load().shape == ((2, 2))

    # Should fail if try again
    with pytest.raises(RuntimeError):
        df.consolidate_data_files(save_dr, replace_with='test',
                                  scope='data file', cast_to=None,
                                  clear_existing='fail', n_jobs=-1)


def test_data_files_consolidate2():

    df = setup_datafiles()
    assert df.file_mapping[0].load().shape == (2,)

    df['data_files2'] = df['data_files'].copy()
    df.add_scope('data_files2', 'data file', inplace=True)

    temp_dr = tempfile.gettempdir()
    save_dr = os.path.join(temp_dr, 'save_consol')

    if os.path.exists(save_dr):
        shutil.rmtree(save_dr)

    df.consolidate_data_files(save_dr, replace_with='test',
                              scope='data file', cast_to='float64',
                              clear_existing='fail', n_jobs=-1)

def test_data_files_integration():

    df = setup_datafiles()

    assert len(df['data_files']) == 3
    assert df.loc[0, 'data_files'] == 0
    assert df.loc[2, 'data_files'] == 2

    # Test get values
    values = df.get_values(col='data_files', reduce_func=np.mean)
    assert values.loc[0] == 0
    assert values.loc[1] == 1
    assert values.loc[2] == 2

    values = df.get_values(col='data_files', reduce_func=np.max, n_jobs=2)
    assert values.loc[0] == 0
    assert values.loc[1] == 1
    assert values.loc[2] == 2

    # Test auto detect categorical
    df = df.auto_detect_categorical()
    cat_cols = df.get_cols(scope='category')
    assert cat_cols == ['2', '3']

    # This also tests copy
    df_copy = df.copy(deep=True)

    # Drop outliers by std, should drop all by 1
    df_copy.filter_outliers_by_std(n_std=.5, scope='data file',
                                   drop=True, reduce_func=np.mean,
                                   n_jobs=1, inplace=True)

    # Make sure self._check_file_mapping works
    assert len(df_copy.file_mapping) == 1
    assert 1 in df_copy.file_mapping
    assert len(df_copy) == 1
    assert len(df.file_mapping) == 3

    # Try drop = False
    df_copy = df.copy(deep=True)
    df_copy = df_copy.filter_outliers_by_std(n_std=.5, scope='data file',
                                             drop=False, reduce_func=np.mean,
                                             n_jobs=1)

    assert np.nan in df_copy.file_mapping
    assert len(df_copy.file_mapping) == 2
    assert 1 in df_copy.file_mapping
    assert len(df_copy) == 3


def test_multi_index_add_data_files():

    df = get_fake_multi_index_dataset()

    def file_to_subject(i):

        subj = i.split('_')[1]
        event = i.split('_')[2]

        return (subj, event)

    # Leave c_s3_e2' as NaN
    files = {'files': ['a_s1_e1', 'a_s1_e2',
                       'b_s2_e1', 'b_s2_e2', 'c_s3_e1']}

    df = df.add_data_files(files=files,
                           file_to_subject=file_to_subject,
                           load_func=np.load)

    assert len(df['files']) == 6
    assert 'a_s1_e1' in df.file_mapping[0].loc


def test_to_data_file():

    df = get_fake_dataset()
    df['file'] = ['loc1', 'loc2', 'loc3']

    df.to_data_file('file', inplace=True)
    file_mapping = df.get_file_mapping()

    assert len(file_mapping) == 3
    assert 0 in file_mapping
    assert file_mapping[1].loc.endswith('loc2')


def test_to_data_file_not_inplace():

    df = get_fake_dataset()
    df['file'] = ['loc1', 'loc2', 'loc3']

    df = df.to_data_file('file', inplace=False)
    file_mapping = df.get_file_mapping()

    assert len(file_mapping) == 3
    assert 0 in file_mapping
    assert file_mapping[2].loc.endswith('loc3')
