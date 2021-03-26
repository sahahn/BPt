from ...main.input import Piece
from .constructor import MVTransformerConstructor


class MVTransformer(Piece):

    _constructor = MVTransformerConstructor

    def __init__(self, obj, scopes, params=0, cache_loc=None,
                 **extra_params):

        self.obj = obj
        self.params = params
        self.scopes = scopes
        self.cache_loc = cache_loc
        self.extra_params = extra_params

        self._check_args()
