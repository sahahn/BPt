from sklearn.compose import ColumnTransformer


class InPlaceColumnTransformer(ColumnTransformer):

    def reverse_X(self, X):

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

    def fit_transform(self, X, y=None):
        return self.reverse_X(super().fit_transform(X, y))

    def transform(self, X):
        return self.reverse_X(super().transform(X))
