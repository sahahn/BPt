from ..helpers import proc_fop, proc_file_input
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

    file_series = proc_file_input(files={'something': '*.npy'},
                                  file_to_subject='auto')
    assert isinstance(file_series, dict)
