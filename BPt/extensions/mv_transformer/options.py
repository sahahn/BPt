from ...default.helpers import get_obj_and_params, all_from_objects


MV_TRANSFORMERS = {}

try:
    from mvlearn.embed import CCA
    MV_TRANSFORMERS['cca'] = (CCA, ['base mv'])


except ImportError:
    pass


def get_mv_transformer_and_params(trans_str, extra_params, params, **kwargs):

    obj, extra_obj_params, obj_params =\
        get_obj_and_params(trans_str, MV_TRANSFORMERS, extra_params, params)

    return obj(**extra_obj_params), obj_params


all_obj_keys = all_from_objects(MV_TRANSFORMERS)
