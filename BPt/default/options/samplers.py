from ..helpers import get_obj_and_params, all_from_objects
from ...extensions.samplers import OverSampler

SAMPLERS = {
    'oversample': (OverSampler, ['default']),
}

def get_sampler_and_params(obj_str, extra_params, params, **kwargs):


    obj, extra_obj_params, obj_params =\
        get_obj_and_params(obj_str, SAMPLERS, extra_params, params)

    return obj(**extra_obj_params), obj_params


all_obj_keys = all_from_objects(SAMPLERS)