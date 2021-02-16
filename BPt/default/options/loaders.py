from ..helpers import get_obj_and_params
from ...extensions.Loaders import Identity, SurfLabels

LOADERS = {
    'identity': (Identity, ['default']),
    'surface rois': (SurfLabels, ['default']),
}

# If nilearn dependencies
try:
    from nilearn.input_data import NiftiLabelsMasker
    from ...extensions.Loaders import Connectivity
    LOADERS['volume rois'] = (NiftiLabelsMasker, ['default'])
    LOADERS['connectivity'] = (Connectivity, ['default'])

except ImportError:
    pass


def get_loader_and_params(loader_str, extra_params, params, **kwargs):

    loader, extra_loader_params, loader_params =\
        get_obj_and_params(loader_str, LOADERS, extra_params, params)

    return loader(**extra_loader_params), loader_params
