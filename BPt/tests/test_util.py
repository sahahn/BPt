from ..util import conv_to_list, save_docx_table, get_top_substrs
import pandas as pd
import tempfile
import os


def test_conv_to_list_None():

    assert conv_to_list(None) is None


def test_save_docx_table():

    loc = os.path.join(tempfile.gettempdir(), 't.docx')

    df = pd.DataFrame(['1', '2', '3'], columns=['1'])
    save_docx_table(df, loc)

    assert os.path.exists(loc)
    os.remove(loc)


def test_get_top_substrs():

    e = get_top_substrs(['apple_123', 'dog_123'])
    assert len(e) == 1
    assert e[0] == '_123'


def test_get_top_substrs_no_overlap():

    e = get_top_substrs(['n', 'g', 'x'])
    assert len(e) == 0
