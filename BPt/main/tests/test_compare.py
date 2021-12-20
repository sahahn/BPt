from ..compare import (Option, CompareDict, MultipleSummary,
                       compare_dict_from_existing)
from ...dataset.Dataset import Dataset
from ..funcs import evaluate
import numpy as np
from ..input import (Model, Pipeline)
import tempfile
import os
import shutil
import random
from copy import deepcopy


def test_option_compares():

    # Should compare by name and key, not value

    o1 = Option(value='10', name='10', key='something')
    o2 = Option(value='20', name='10', key='something')

    assert o1 == o2
    assert not o1 > o2
    assert not o1 < o2


def test_option_compares_not_equal():

    o1 = Option(value='10', name='10', key='something2')
    o2 = Option(value='20', name='10', key='something')
    assert o1 != o2

    o1 = Option(value='10', name='10', key='something')
    o2 = Option(value='20', name='20', key='something')
    assert o1 != o2


def get_results(problem_type='regression', cv=2, n_subjs=20):

    fake = Dataset()

    fake['1'] = np.ones((n_subjs))
    fake['2'] = np.ones((n_subjs))
    fake['3'] = np.ones((n_subjs))
    fake['3'][:n_subjs//2] = 0
    fake = fake.set_role('3', 'target')

    pipe = Pipeline(Model('dt'))

    results = evaluate(pipeline=pipe,
                       dataset=fake,
                       cv=cv,
                       problem_type=problem_type,
                       progress_bar=False)

    return results

def test_compare_dict_mixed_summary():

    r1 = get_results('regression')
    r2 = get_results('binary')

    cd = compare_dict_from_existing([r1, r2])

    ms = cd.summary(show_timing=True)
    assert isinstance(ms, MultipleSummary)
    assert len(list(ms.summary_dfs)) == 2

def test_compare_dict_from_existing_case1():
    # Dict case

    r1 = get_results()
    results = {'1': r1, '2': deepcopy(r1), '3': deepcopy(r1)}

    cd = compare_dict_from_existing(results)
    assert isinstance(cd, CompareDict)
    assert cd['3'] is not None
    assert cd.summary(show_timing=True).shape == (3, 6)


def test_compare_dict_from_existing_case2():
    # list case

    r1 = get_results()
    results = [r1, deepcopy(r1), deepcopy(r1)]

    cd = compare_dict_from_existing(results)
    assert isinstance(cd, CompareDict)
    assert cd['0'] is not None
    assert cd['2'] is not None
    assert cd.summary(show_timing=True).shape == (3, 6)
    assert cd.summary().shape == (3, 4)


def test_compare_dict_from_existing_case3():
    # Pass as files cases

    r1 = get_results()

    # Save
    temp_dr = tempfile.gettempdir()
    s_loc1 = os.path.join(temp_dr, '1.pkl')
    s_loc2 = os.path.join(temp_dr, '2.pkl')
    r1.to_pickle(s_loc1)
    r1.to_pickle(s_loc2)

    # Dict case
    cd = compare_dict_from_existing({'x': s_loc1, 'y': s_loc2})
    assert isinstance(cd, CompareDict)
    assert cd['x'] is not None
    assert cd['y'] is not None
    assert cd.summary(show_timing=True).shape == (2, 6)

    # List case
    cd = compare_dict_from_existing([s_loc1, s_loc2])
    assert isinstance(cd, CompareDict)
    assert cd['1'] is not None
    assert cd['2'] is not None
    assert cd.summary(show_timing=True).shape == (2, 6)

    # Alt list mixed case
    cd = compare_dict_from_existing([s_loc1, s_loc2, r1])
    assert isinstance(cd, CompareDict)
    assert cd['0'] is not None
    assert cd['1'] is not None
    assert cd['2'] is not None
    assert cd.summary(show_timing=True).shape == (3, 6)

    # Clean up
    os.remove(s_loc1)
    os.remove(s_loc2)


def test_compare_dict_from_existing_case4():
    # Pass as file dr case

    r1 = get_results()

    # Save
    temp_dr = os.path.join(tempfile.gettempdir(), str(random.random()))
    os.makedirs(temp_dr, exist_ok=True)

    s_loc1 = os.path.join(temp_dr, 'abcX.pkl')
    s_loc2 = os.path.join(temp_dr, 'abcY.pkl')
    s_loc3 = os.path.join(temp_dr, 'abcZ.pkl')

    r1.to_pickle(s_loc1)
    r1.to_pickle(s_loc2)
    r1.to_pickle(s_loc3)

    # Load from dir
    cd = compare_dict_from_existing(temp_dr)
    assert cd['X'] is not None
    assert cd['Y'] is not None
    assert cd['Z'] is not None
    assert cd.summary(show_timing=True).shape == (3, 6)

    # Clean up
    shutil.rmtree(temp_dr)

def test_compare_dict_summary_df_cv():
    # list case

    r1 = get_results()
    r2 = get_results(cv=3)

    results = [r1, r2]

    cd = compare_dict_from_existing(results)
    assert isinstance(cd, CompareDict)
    assert cd['0'] is not None
    assert cd['1'] is not None

    assert 'n_folds' in cd._check_evaluator_difs()
    assert cd.summary(show_timing=True).shape == (2, 7)


def test_compare_dict_summary_df_cv_dif2():
    # list case

    r1 = get_results()
    r2 = get_results(cv=3, n_subjs=30)

    results = [r1, r2]

    cd = compare_dict_from_existing(results)
    assert isinstance(cd, CompareDict)
    assert cd['0'] is not None
    assert cd['1'] is not None

    assert 'n_folds' in cd._check_evaluator_difs()
    assert 'n_subjects' in cd._check_evaluator_difs()
    assert cd.summary(show_timing=True).shape == (2, 8)
