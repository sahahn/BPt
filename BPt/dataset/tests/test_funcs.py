from pandas.core.algorithms import value_counts
from ..funcs import concat
from ..Dataset import Dataset
from ..data_file import DataFile
import pytest
import numpy as np
import os

def test_concat_axis_1_error_duplicate_col_names():

    x = Dataset(['1'], columns=['1'])
    y = Dataset(['1'], columns=['1'])

    with pytest.raises(RuntimeError):
        concat([x, y], axis=1)

def test_concat_scopes1():

    x = Dataset([0], columns=['1'], scopes={'1': '1'})
    y = Dataset([1], columns=['2'], scopes={'2': '2'})

    data = concat([x, y], axis=0)

    assert data.shape == (2, 2)
    assert len(data.get_scopes()) == 2
    assert data.get_scopes()['1'] == '1'
    assert data.get_scopes()['2'] == '2'


def test_concat_roles1():

    x = Dataset([0], columns=['1'], roles={'1': 'data'})
    y = Dataset([1], columns=['2'], roles={'2': 'data'})

    data = concat([x, y], axis=0)
    assert data.shape == (2, 2)

    assert len(data.get_roles()) == 2
    assert data.get_roles()['1'] == 'data'
    assert data.get_roles()['2'] == 'data'


def test_concat_roles2():

    x = Dataset([[1, 1], [2, 2], [3, 3]], columns=['1', '2'], roles={'1': 'data', '2': 'target'})
    y = Dataset([[1, 1], [2, 2], [3, 3]], columns=['1', '2'], roles={'1': 'data', '2': 'data'})

    data = concat([x, y], axis=0)
    assert data.shape == (6, 2)

    assert len(data.get_roles()) == 2
    assert data.get_roles()['1'] == 'data'
    assert data.get_roles()['2'] == 'data'

def test_concat_roles3():

    x = Dataset([[1, 1], [2, 2], [3, 3]], columns=['1', '2'], roles={'1': 'data', '2': 'target'})
    y = Dataset([[1, 1], [2, 2], [3, 3]], columns=['1', '2'], roles={'1': 'data', '2': 'target'})

    data = concat([x, y], axis=0)
    assert data.shape == (6, 2)

    assert len(data.get_roles()) == 2
    assert data.get_roles()['1'] == 'data'
    assert data.get_roles()['2'] == 'target'

def test_concat_roles4():

    x = Dataset([[1, 1], [2, 2], [3, 3]], columns=['1', '2'], roles={'1': 'data', '2': 'target'})
    y = Dataset([[1, 1], [2, 2], [3, 3]], columns=['1', '2'], roles={'1': 'data', '2': 'non input'})

    data = concat([x, y], axis=0)
    assert data.shape == (6, 2)

    assert len(data.get_roles()) == 2
    assert data.get_roles()['1'] == 'data'
    assert data.get_roles()['2'] == 'target'

def test_concat_roles5():

    x = Dataset([[1, 1], [2, 2], [3, 3]], columns=['1', '2'], roles={'1': 'data', '2': 'non input'})
    y = Dataset([[1, 1], [2, 2], [3, 3]], columns=['1', '2'], roles={'1': 'data', '2': 'non input'})

    data = concat([x, y], axis=0)
    assert data.shape == (6, 2)

    assert len(data.get_roles()) == 2
    assert data.get_roles()['1'] == 'data'
    assert data.get_roles()['2'] == 'non input'

def test_concat_encoders1():

    x = Dataset([['1', '11'], ['2', '22'], ['3', '33']],
                columns=['1', '2'])
    x = x.ordinalize('all')

    y = Dataset([['x', 'xx'], ['y', 'yy'], ['z', 'zz']],
                columns=['1', '2'])
    y = y.ordinalize('all')

    data = concat([x, y], axis=0)
    assert 'category' in data.get_scopes()['1'] 
    assert 'category' in data.get_scopes()['2'] 

    encoders = data._get_encoders()
    assert len(encoders['1']) == 6
    assert encoders['1'][2] == '3'
    assert encoders['1'][4] == 'y'
    assert encoders['2'][4] == 'yy'
    assert encoders['2'][3] == 'xx'
    
def test_concat_encoders1():

    x = Dataset([[1, 1], [2, 2], [3, 3]], columns=['1', '2'])
    x = x.binarize('1', threshold=1.5)

    y = Dataset([[2, 2], [1, 1], [5, 5]], columns=['1', '2'])
    y = y.binarize('1', threshold=1.5)

    data = concat([x, y], axis=0)
    assert len(data.encoders['1']) == 2
    assert data.encoders['1'][0] == '<1.5'

    
def test_tr_test_subjs1():

    x = Dataset([[1, 1], [2, 2], [3, 3]], columns=['1', '2'])
    x = x.set_index('1')
    x = x.set_test_split(subjects=[1])

    y = Dataset([[4, 2], [5, 3], [6, 4]], columns=['1', '2'])
    y = y.set_index('1')
    y = y.set_test_split(subjects=[4])

    data = concat([x, y], axis=0)
    assert list(data.test_subjects) == [1, 4]
    assert list(data.train_subjects) == [2, 3, 5, 6]

def test_tr_test_subjs2():

    x = Dataset([[1, 1], [2, 2], [3, 3]], columns=['1', '2'])
    x = x.set_index('1')
    x = x.set_train_split(subjects=[1])

    y = Dataset([[4, 2], [5, 3], [6, 4]], columns=['1', '2'])
    y = y.set_index('1')
    y = y.set_train_split(subjects=[4])

    z = Dataset([['7', 2], ['8', 3], ['9', 4]], columns=['1', '2'])
    z = z.set_index('1')
    z = z.set_train_split(subjects=['7'])

    # Concat
    data = concat([x, y, z], axis=0)

    assert list(data.test_subjects) == [2, 3, 5, 6, '8', '9']
    assert list(data.train_subjects) == [1, 4, '7']

def test_concat_datasets_axis0():

    x = Dataset([1, 2, 3], columns=['1'])
    x = x.add_scope('1', 'data file')
    x.file_mapping = {1: DataFile('1', np.load),
                      2: DataFile('2', np.load),
                      3: DataFile('3', np.load)}

    y = Dataset([1, 2, 3], columns=['1'])
    y = y.add_scope('1', 'data file')
    y.file_mapping = {1: DataFile('5', np.load),
                      2: DataFile('6', np.load),
                      3: DataFile('7', np.load)}

    # Concat
    data = concat([x, y], axis=0)

    # Make sure lines up - each data file to new unique
    for val in list(data['1']):
        assert os.path.basename(data.file_mapping[val].loc) == str(val)

    assert len(data.file_mapping) == 6


def test_concat_datasets_axis1():

    x = Dataset([1, 2, 3], columns=['1'])
    x = x.add_scope('1', 'data file')
    x.file_mapping = {1: DataFile('1', np.load),
                      2: DataFile('2', np.load),
                      3: DataFile('3', np.load)}

    y = Dataset([1, 2, 3], columns=['2'])
    y = y.add_scope('2', 'data file')
    y.file_mapping = {1: DataFile('5', np.load),
                      2: DataFile('6', np.load),
                      3: DataFile('7', np.load)}

    # Concat
    data = concat([x, y], axis=1)

    # Make sure lines up - each data file to new unique
    for val in list(data['1']):
        assert os.path.basename(data.file_mapping[val].loc) == str(val)
    for val in list(data['2']):
        assert os.path.basename(data.file_mapping[val].loc) == str(val)

    assert len(data.file_mapping) == 6