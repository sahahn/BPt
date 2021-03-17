from ...pipeline.constructors import Constructor
from .MVTransformer import MVTransformer


def get_mv_transformer_and_params():

    return None


class MVTransformerConstructor(Constructor):

    name = 'mv_transformers'

    def _process(self, params):

        # Then call get objs and params
        objs, obj_params =\
            self._get_objs_and_params(get_mv_transformer_and_params,
                                      params)

        return self._make_col_version(objs, obj_params,
                                      params, Wrapper=MVTransformer)
