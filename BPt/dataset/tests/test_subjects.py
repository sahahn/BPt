import numpy as np
import pandas as pd
import tempfile
import os
import pytest
from ...main.input_operations import ValueSubset, Intersection
from .datasets import (get_fake_dataset, get_fake_dataset2, get_full_dataset,
                       get_fake_multi_index_dataset, get_full_int_index_dataset,
                       get_fake_dataset4)


def test_get_subjects_None():

    df = get_fake_dataset()

    subjects = df.get_subjects(None, return_as='set')
    assert len(subjects) == 0

    subjects = df.get_subjects(None, return_as='index')
    assert len(subjects) == 0

    subjects = df.get_subjects(None, return_as='flat index')
    assert len(subjects) == 0


def test_get_subjects_tr_test_case():

    df = get_full_dataset()

    assert len(df.get_subjects('all')) == 5
    assert len(df.get_subjects('train')) == 3
    assert len(df.get_subjects('test')) == 2

    # Make sure updates to dropped subjects
    df = df.drop('s1')
    assert len(df.get_subjects('train')) == 2
    df = df.drop('s4')
    assert len(df.get_subjects('test')) == 1


def test_get_subjects_int_index_tr_test_case():

    df = get_full_int_index_dataset()

    assert len(df.get_subjects('all')) == 5
    assert len(df.get_subjects('train')) == 3
    assert len(df.get_subjects('test')) == 2

    # Make sure updates to dropped subjects
    df = df.drop(1)
    assert len(df.get_subjects('train')) == 2
    df = df.drop(4)
    assert len(df.get_subjects('test')) == 1


def test_get_subjects_intersection():

    df = get_fake_dataset4()
    assert len(df.get_subjects('all')) == 6
    assert len(df.get_subjects([1, 2])) == 2

    subjs = df.get_subjects(Intersection([[1, 2]]))
    assert len(subjs) == 2

    subjs = df.get_subjects(Intersection([[1, 2], [2, 3]]))
    assert len(subjs) == 1

    subjs = df.get_subjects(Intersection([[1, 2], [2, 3], [4, 5]]))
    assert len(subjs) == 0

    subjs = df.get_subjects(Intersection([[1, 2], [2, 3], [2, 3]]))
    assert len(subjs) == 1


def test_get_subjects_nan():

    # Subject id == 0 is only one with NaN
    df = get_fake_dataset()
    index_dtype = df.index.dtype.name

    subjects = df.get_subjects('nan', return_as='set')
    assert len(subjects) == 1
    assert df.loc[subjects].shape == (1, 3)
    subj = subjects.pop()
    assert subj == 0
    assert isinstance(subj, int)

    subjects = df.get_subjects('nan', return_as='flat index')
    assert len(subjects) == 1
    assert df.loc[subjects].shape == (1, 3)
    assert subjects[0] == 0
    assert subjects[0].dtype == index_dtype

    subjects = df.get_subjects('nan', return_as='index')
    assert len(subjects) == 1
    assert df.loc[subjects].shape == (1, 3)
    assert subjects[0] == 0
    assert subjects[0].dtype == index_dtype

    # Dataset with no Nan's
    df = get_fake_dataset2()
    subjects = df.get_subjects('nan', return_as='set')
    assert len(subjects) == 0


def test_get_subjects_ValueSubset():

    df = get_fake_dataset()

    # Int column test 1 value
    vs = ValueSubset(name='1', values=1, decode_values=False)
    subjects = df.get_subjects(vs, return_as='set')
    assert len(subjects) == 1
    assert subjects.pop() == 0

    # Int column test 2 values
    vs = ValueSubset(name='1', values=[1, 2], decode_values=False)
    subjects = df.get_subjects(vs, return_as='set')
    assert len(subjects) == 2

    # Str column test 1 value
    vs = ValueSubset(name='2', values=['6'], decode_values=False)
    subjects = df.get_subjects(vs, return_as='set')
    assert len(subjects) == 1
    assert subjects.pop() == 0

    # Str column test 2 values
    vs = ValueSubset(name='2', values=['6', '7'], decode_values=False)
    subjects = df.get_subjects(vs, return_as='set')
    assert len(subjects) == 2

    # Str / cat column test extra values
    vs = ValueSubset(name='2', values=['6', '7', '9'], decode_values=False)
    subjects = df.get_subjects(vs, return_as='set')
    assert len(subjects) == 2

    # Column w/ nan
    vs = ValueSubset(name='3', values=np.nan, decode_values=False)
    subjects = df.get_subjects(vs, return_as='set')
    assert len(subjects) == 1
    assert subjects.pop() == 0

    # Column w/ nan 2 values
    vs = ValueSubset(name='3', values=[np.nan, 2], decode_values=False)
    subjects = df.get_subjects(vs, return_as='set')
    assert len(subjects) == 2
    assert df.loc[subjects].shape == (2, 3)

    # Bad name col
    vs = ValueSubset(name=1, values=[np.nan, 2], decode_values=False)
    with pytest.raises(KeyError):
        subjects = df.get_subjects(vs, return_as='set')

    # Bad input
    with pytest.raises(ValueError):
        vs = ValueSubset(name=[1, 2], values=2)


def test_get_subjects_base():

    df = get_fake_dataset()

    subjs = [0]
    subjects = df.get_subjects(subjs, return_as='set')
    assert len(subjects) == 1
    assert df.loc[subjects].shape == (1, 3)

    subjs = ['0']
    subjects = df.get_subjects(subjs, return_as='set')
    assert len(subjects) == 1
    assert df.loc[subjects].shape == (1, 3)

    subjs = np.array([0])
    subjects = df.get_subjects(subjs, return_as='set')
    assert len(subjects) == 1
    assert df.loc[subjects].shape == (1, 3)

    subjs = pd.Index(data=[0], name=df.index.name)
    subjects = df.get_subjects(subjs, return_as='set')
    assert len(subjects) == 1
    assert df.loc[subjects].shape == (1, 3)

    subjs = pd.Index(data=np.array([0, 2]), name=df.index.name)
    subjects = df.get_subjects(subjs, return_as='index')
    assert len(subjects) == 2
    assert df.loc[subjects].shape == (2, 3)
    assert np.array_equal(np.array([0, 2]), subjects)

    subjects = df.get_subjects(subjs, return_as='flat index')
    assert len(subjects) == 2
    assert df.loc[subjects].shape == (2, 3)
    assert np.array_equal(np.array([0, 2]), subjects)


def test_get_subjects_base_file():

    df = get_fake_dataset()

    temp_loc = os.path.join(tempfile.gettempdir(), 'temp.txt')
    with open(temp_loc, 'w') as f:
        f.write('0\n')

    subjects = df.get_subjects(temp_loc, return_as='set')
    assert len(subjects) == 1
    assert df.loc[subjects].shape == (1, 3)

    temp_loc = os.path.join(tempfile.gettempdir(), 'temp.txt')
    with open(temp_loc, 'w') as f:
        f.write('0\n')
        f.write('1\n')

    subjects = df.get_subjects(temp_loc, return_as='set')
    assert len(subjects) == 2
    assert df.loc[subjects].shape == (2, 3)


def test_multi_index_get_subjects():

    df = get_fake_multi_index_dataset()

    subjs = df.get_subjects(subjects=None)
    assert len(subjs) == 0

    subjs = df.get_subjects(subjects=['s1'])
    assert len(subjs) == 2
    assert ('s1', 'e1') in subjs
    assert ('s1', 'e2') in subjs

    subjs = df.get_subjects(subjects=['s1'], only_level=0)
    assert len(subjs) == 1
    assert 's1' in subjs

    subjs = df.get_subjects(subjects=['s1'], return_as='index',
                            only_level=0)
    assert 's1' in subjs
    assert subjs.name == 'subj'

    subjs = df.get_subjects(subjects=['s1'], return_as='index',
                            only_level=None)
    assert ('s1', 'e1') in subjs
    assert ('s1', 'e2') in subjs
    assert subjs.names == ['subj', 'event']

    subjs = df.get_subjects(subjects=(['s1', 's2'], ['e1']),
                            return_as='set')

    assert len(subjs) == 2
    assert ('s1', 'e1') in subjs
    assert ('s2', 'e1') in subjs

    subjs = df.get_subjects(subjects=('all', ['e1']),
                            return_as='set')
    assert len(subjs) == 3
    assert df.loc[subjs].shape == (3, 3)
