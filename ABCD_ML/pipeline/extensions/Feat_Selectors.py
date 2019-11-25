from sklearn.feature_selection import RFE


class RFE(RFE):
    def fit(self, X, y):
        '''Override the fit function from base
           specifically allow passing in float % to keep.
        '''

        if isinstance(self.n_features_to_select, float):

            if self.n_features_to_select <= 0:
                self.n_features_to_select = 1

            if self.n_features_to_select < 1:
                divide_by = self.n_features_to_select ** -1
                self.n_features_to_select = X.shape[1] // divide_by

        return self._fit(X, y)
