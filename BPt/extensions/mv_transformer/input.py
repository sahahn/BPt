from ...main.input import Piece


class MVTransformer(Piece):

    def __init__(self, obj, scopes, params=0, cache_loc=None,
                 **extra_params):

        self.obj = obj
        self.params = params
        self.scope = scopes
        self.cache_loc = cache_loc
        self.extra_params = extra_params

        self._check_args()
