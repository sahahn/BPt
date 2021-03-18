
from copy import deepcopy
from .model_params import P as model_params
from .ensemble_params import P as ensemble_params
from .feat_selector_params import P as feat_selector_params
from .imputer_params import P as imputer_params
from .scaler_params import P as scaler_params
from .transformer_params import P as transformer_params

# Extensions
from ...extensions.mv_transformer.mv_transformer_params import P as mv_params

PARAMS = {}
PARAMS.update(model_params)
PARAMS.update(ensemble_params)
PARAMS.update(feat_selector_params)
PARAMS.update(scaler_params)
PARAMS.update(imputer_params)
PARAMS.update(transformer_params)
PARAMS.update(mv_params)


def get_base_params(str_indicator):

    base_params = deepcopy(PARAMS[str_indicator])
    return base_params


def proc_params(base_params, prepend):

    if isinstance(base_params, int):
        return {}

    params = {prepend + '__' + key: base_params[key] for key in
              base_params}

    return params
