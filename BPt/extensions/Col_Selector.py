from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.utils import _safe_indexing
from ..helpers.ML_Helpers import proc_mapping
import numpy as np


class ColTransformer(ColumnTransformer):

    def _resave_transformers(self, new_idx):

        self.transformers =\
            [(self.transformers[0][0], self.transformers[0][1], new_idx)]

    def _proc_mapping(self, mapping):

        if hasattr(self, '_mapping'):
            return
        else:
            self._mapping = mapping

        if len(mapping) > 0:
            new_indx = proc_mapping(self.transformers[0][2], mapping)
            self._resave_transformers(new_indx)

        return

    def fit(self, X, y=None, mapping=None):

        if mapping is None:
            mapping = {}

        self._proc_mapping(mapping)
        super().fit(X, y)

        return self

    def fit_transform(self, X, y=None, mapping=None):

        if mapping is None:
            mapping = {}

        self._proc_mapping(mapping)
        return super().fit_transform(X, y)

    def _fit_transform(self, X, y, func, fitted=False):

        transformers = list(
            self._iter(fitted=fitted, replace_strings=True))

        res = []
        for idx, (name, trans, column, weight) in enumerate(
                self._iter(fitted=fitted, replace_strings=True), 1):

            try:
                res.append(func(
                    transformer=clone(trans) if not fitted else trans,
                    X=_safe_indexing(X, column, axis=1),
                    y=y,
                    weight=weight,
                    message_clsname='ColumnTransformer',
                    message=self._log_message(name, idx, len(transformers)),
                    mapping=self._mapping))

            except TypeError:
                res.append(func(
                    transformer=clone(trans) if not fitted else trans,
                    X=_safe_indexing(X, column, axis=1),
                    y=y,
                    weight=weight,
                    message_clsname='ColumnTransformer',
                    message=self._log_message(name, idx, len(transformers))))

            except ValueError as e:
                if "Expected 2D array, got 1D array instead" in str(e):
                    raise ValueError()
                else:
                    raise

        return res


class ColDropStrat(ColTransformer):

    def fit(self, X, y=None, mapping=None):
        self.n_X_feats_ = X.shape[1]
        return super().fit(X, y, mapping)

    def fit_transform(self, X, y=None, mapping=None):
        self.n_X_feats_ = X.shape[1]
        return super().fit_transform(X, y, mapping)

    def inverse_transform(self, X):

        keep_inds = self.transformers[0][2]
        Xt = np.zeros((X.shape[0], self.n_X_feats_), dtype=X.dtype)
        Xt[:, keep_inds] = X
        return Xt


class InPlaceColTransformer(ColTransformer):

    def _reverse_X(self, X):

        if len(self.transformers) > 1:
            raise RuntimeError('InPlaceColumnTransformer only works with',
                               'one transformer!')

        idx = self.transformers[0][2]

        remaining_idx = list(set(range(X.shape[1])) - set(idx))
        remaining_idx = sorted(remaining_idx)

        map_back = {idx[c]: c for c in range(len(idx))}
        map_back.update({remaining_idx[c]: c + len(idx)
                         for c in range(len(remaining_idx))})
        mb = [map_back[c] for c in range(X.shape[1])]

        return X[:, mb]

    def fit_transform(self, X, y=None, mapping=None):

        if mapping is None:
            mapping = {}

        return self._reverse_X(super().fit_transform(X, y, mapping=mapping))

    def transform(self, X):
        return self._reverse_X(super().transform(X))



