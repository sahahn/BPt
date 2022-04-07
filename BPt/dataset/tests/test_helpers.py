from ..helpers import proc_fop, proc_file_input, auto_determine_subjects
from ..dataset import Dataset
import pytest


def test_proc_fop():

    assert proc_fop(1) == (.01, .99)
    assert proc_fop(2) == (.02, .98)
    assert proc_fop((None, 5)) == (None, .95)
    assert proc_fop((5, None)) == (.05, None)

    res = proc_fop((3, 7))
    assert res[0] > .029
    assert res[0] < .031
    assert res[1] > .929
    assert res[1] < .931


def test_proc_file_input():

    with pytest.raises(ValueError):
        proc_file_input(files='not dict',
                        file_to_subject='auto')

    files = {'something': ['loc1.npy', 'loc2.npy']}

    with pytest.raises(RuntimeError):
        proc_file_input(files=files,
                        file_to_subject=None)

    with pytest.raises(ValueError):
        proc_file_input(files=files,
                        file_to_subject={'dif_key': 'auto'})


def test_auto_determine_subjects_template1():

    data = Dataset()
    data['1'] = [1, 2]
    data.index = ['subj-0001', 'subj-0002']
    existing_index = data.index
    file_paths =\
        ['data/ds003097/derivatives/proc_fmri/sub-0001/func_timeseries.npy',
         'data/ds003097/derivatives/proc_fmri/sub-0002/func_timeseries.npy']

    auto_determine_subjects(file_paths, existing_index) == ['subj-0001',
                                                            'subj-0002']


def test_auto_determine_subjects_template2():

    data = Dataset()
    data['1'] = [1, 2, 3]
    data.index = ['subjs_grp1-0001', 'subjs_grp1-0002', 'subjs_grp1-0003']
    existing_index = data.index
    file_paths = ['some_loc/subjs_grp1-0001', 'some_loc/subjs_grp1-0002',
                  'some_loc/subjs_grp1-0003']
    auto_determine_subjects(file_paths, existing_index) == ['subjs_grp1-0001',
                                                            'subjs_grp1-0002',
                                                            'subjs_grp1-0003']

def test_auto_determine_subjects_template_int_case():

    data = Dataset()
    data['1'] = [1, 2, 3]
    data.index = [1, 2, 3]
    
    existing_index = data.index

    # Make sure to test with and w/o leading 0's
    file_paths = ['some_loc/subjs_grp1-0001',
                  'some_loc/subjs_grp1-0002',
                  'some_loc/subjs_grp1-3']


    # Makes sure index works
    subjs = auto_determine_subjects(file_paths, existing_index)
    assert data.loc[subjs].shape == (3, 1)