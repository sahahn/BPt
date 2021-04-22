from pandas.core.dtypes.base import ExtensionDtype, register_extension_dtype
import numpy as np


class DataFileDtypeType(type):
    pass


@register_extension_dtype
class DataFileDtype(ExtensionDtype):

    kind = 'O'
    str = "|O08"
    base = np.dtype("O")
    subdtype = None
    type = DataFileDtypeType
    name = 'file'

    isbuiltin = 0
    isnative = 0

    _metadata = ()
    na_value = np.nan

    def __str__(self):
        """
        Return a string representation for a particular Object
        """
        return self.name

    def __repr__(self):
        """
        Return a string representation for a particular object.
        """
        return self.name

    def __getstate__(self):
        return {k: getattr(self, k, None) for k in self._metadata}

    @classmethod
    def construct_array_type(cls):
        """
        Return the array type associated with this dtype.
        Returns
        -------
        type
        """

        from .data_file_array import DataFileArray
        return DataFileArray

    @classmethod
    def construct_from_string(cls, string):
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )

        if string != cls.name:
            raise TypeError(
                f"Cannot construct a 'DataFileDType' from '{string}'")

        return cls()

    def __hash__(self) -> int:
        return hash(str(self))

    def __setstate__(self, state) -> None:
        # for pickle compat. __get_state__ is defined in the
        # PandasExtensionDtype superclass and uses the public properties to
        # pickle -> need to set the settable private ones here (see GH26067)
        # self._tz = state["tz"]
        # self._unit = state["unit"]

        pass

    @classmethod
    def _from_values_or_dtype(cls, values=None, categories=None,
                              ordered=None, dtype=None):
        print('_from_values_or_dtype')

    def _get_common_dtype(self, dtypes):
        return self

    @property
    def _is_boolean(self) -> bool:
        return False
