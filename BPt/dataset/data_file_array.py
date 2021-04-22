from pandas.core.arrays import ExtensionArray
from .data_file_type import DataFileDtype
from .data_file import DataFile
import numpy as np
from pandas._libs import lib


class DataFileArray(ExtensionArray):

    _typ = "datafilearray"
    _scalar_type = DataFile

    ndim = 1
    can_hold_na = True

    _dtype = DataFileDtype

    def __init__(self, data):

        # If passed DataFileArray
        if isinstance(data, DataFileArray):
            self._set_data(data._data)

        # Otherwise set
        self._set_data(data)

    def _set_data(self, data):

        # Data should be a numpy area with dtype
        # object, filled with DataFile objects

        self._data = np.empty(len(data), dtype=object)
        for i, file in enumerate(data):
            if not isinstance(file, self._scalar_type):
                self._data[i] = self._scalar_type(file)
            else:
                self._data[i] = file

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        """
        Construct a new ExtensionArray from a sequence of scalars.
        Parameters
        ----------
        scalars : Sequence
            Each element will be an instance of the scalar type for this
            array, ``cls.dtype.type`` or be converted into this type
            in this method.
        dtype : dtype, optional
            Construct for this particular dtype. This should be a Dtype
            compatible with the ExtensionArray.
        copy : bool, default False
            If True, copy the underlying data.
        Returns
        -------
        DataFileArray
        """

        result = cls(data=scalars)

        return result

    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype=None, copy=False):
        """
        Construct a new ExtensionArray from a sequence of strings.
        .. versionadded:: 0.24.0
        Parameters
        ----------
        strings : Sequence
            Each element will be an instance of the scalar type for this
            array, ``cls.dtype.type``.
        dtype : dtype, optional
            Construct for this particular dtype. This should be a Dtype
            compatible with the ExtensionArray.
        copy : bool, default False
            If True, copy the underlying data.
        Returns
        -------
        ExtensionArray
        """

        return cls()

    def _from_factorized(cls, values, original):
        return cls()

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        raise NotImplementedError(
            f"{type(self)} does not implement __setitem__.")

    def __len__(self) -> int:
        """
        Length of this array
        Returns
        -------
        length : int
        """
        return len(self._data)

    def __iter__(self):
        """
        Iterate over elements of the array.
        """
        # This needs to be implemented so that pandas recognizes extension
        # arrays as list-like. The default implementation makes successive
        # calls to ``__getitem__``, which may be slower than necessary.
        for i in range(len(self)):
            yield self[i]

    def __eq__(self, other):
        """
        Return for `self == other` (element-wise equality).
        """

        return self._data == other._data

    def __ne__(self, other):
        """
        Return for `self != other` (element-wise in-equality).
        """
        return ~(self._data == other._data)

    def to_numpy(self, dtype=None, copy=False, na_value=lib.no_default):
        print('to numpy !!!')

    # ------------------------------------------------------------------------
    # Required attributes
    # ------------------------------------------------------------------------

    @property
    def shape(self):
        """
        Return a tuple of the array dimensions.
        """
        return self._data.shape

    @property
    def size(self) -> int:
        """
        The number of elements in the array.
        """
        return np.prod(self.shape)

    @property
    def ndim(self) -> int:
        """
        Extension Arrays are only allowed to be 1-dimensional.
        """
        return 1

    @property
    def nbytes(self) -> int:
        return self._data.nbytes

    @property
    def dtype(self):
        return DataFileDtype

    def astype(self, dtype, copy=True):
        """
        Cast to a NumPy array with 'dtype'.
        Parameters
        ----------
        dtype : str or dtype
            Typecode or data-type to which the array is cast.
        copy : bool, default True
            Whether to copy the data, even if not necessary. If False,
            a copy is made only if the old dtype does not match the
            new dtype.
        Returns
        -------
        array : ndarray
            NumPy ndarray with 'dtype' for its dtype.
        """
        print('astype!')

    def copy(self):
        """
        Return a copy of the array.
        Returns
        -------
        ExtensionArray
        """

        return DataFileArray(self._data.copy())
