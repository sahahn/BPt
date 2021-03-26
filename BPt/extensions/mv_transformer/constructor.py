from ...pipeline.constructors import Constructor


class MVTransformerConstructor(Constructor):

    name = 'transformers_mv'

    def _process(self, params):

        from .options import get_mv_transformer_and_params
        from .transformer import BPtTransformerMV

        # Then call get objs and params
        objs, obj_params =\
            self._get_objs_and_params(get_mv_transformer_and_params,
                                      params)

        return self._make_col_version(objs, obj_params,
                                      params, Wrapper=BPtTransformerMV)
