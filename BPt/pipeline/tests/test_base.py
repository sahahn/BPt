from ..base import _needs, _get_est_fit_params
from ..BPtLoader import BPtLoader
from ..ScopeObjs import ScopeTransformer
from .helpers import ToFixedTransformer


def test_needs():
    # Needs, everybodies got them

    loader = BPtLoader(None, None, None)

    # Should have _needs_mapping and fit
    assert _needs(loader, '_needs_mapping', 'mapping', 'fit')
    assert _needs(loader, '_needs_fit_index', 'fit_index', 'fit')
    assert not _needs(loader, '_needs_transform_index',
                      'transform_index', 'transform')

    trans = ToFixedTransformer(1)

    assert not _needs(trans, '_needs_mapping', 'mapping', 'fit')
    assert not _needs(trans, '_needs_fit_index', 'fit_index', 'fit')
    assert not _needs(trans, '_needs_transform_index',
                      'transform_index', 'transform')

    st = ScopeTransformer(trans, None)
    assert _needs(st, '_needs_mapping', 'mapping', 'fit')
    assert not _needs(st, '_needs_transform_index',
                      'transform_index', 'transform')

    ToFixedTransformer._needs_transform_index = True
    n_trans = ToFixedTransformer(1)
    st = ScopeTransformer(n_trans, None)

    assert _needs(st, '_needs_mapping', 'mapping', 'fit')
    assert _needs(st, '_needs_transform_index',
                  'transform_index', 'transform')


def test_get_est_fit_params_copy_mapping():

    loader = BPtLoader(None, None, None)

    fit_params = _get_est_fit_params(loader, mapping=None, fit_index=None,
                                     other_params=None, copy_mapping=True)

    assert isinstance(fit_params, dict)
    assert len(fit_params) == 0

    mapping = {1: 1}
    fit_params = _get_est_fit_params(loader, mapping=mapping, fit_index=None,
                                     other_params=None, copy_mapping=True)
    assert isinstance(fit_params, dict)
    fit_params['mapping'][2] = 2
    assert 2 not in mapping

    mapping = {1: 1}
    fit_params = _get_est_fit_params(loader, mapping=mapping, fit_index=None,
                                     other_params=None, copy_mapping=False)
    assert isinstance(fit_params, dict)
    fit_params['mapping'][2] = 2
    assert 2 in mapping
