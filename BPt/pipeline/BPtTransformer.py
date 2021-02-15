from ..helpers.ML_Helpers import update_mapping
from .ScopeObjs import ScopeTransformer
from sklearn.preprocessing import OneHotEncoder


class BPtTransformer(ScopeTransformer):

    def fit(self, X, y=None, mapping=None,
            train_data_index=None, **fit_params):

        # Need the output from a transform to full fit,
        # so when fit is called, call fit_transform instead
        self.fit_transform(X=X, y=y, mapping=mapping,
                           train_data_index=train_data_index,
                           **fit_params)

        return self

    def fit_transform(self, X, y=None, mapping=None,
                      train_data_index=None, **fit_params):

        if mapping is None:
            mapping = {}

        # Call parent fit
        super().fit(X, y=y, mapping=mapping,
                    train_data_index=train_data_index,
                    **fit_params)

        # If skip
        if self.estimator_ is None:
            return X

        # Transform X
        X_trans = self.transform(X)

        # Need to update the mapping before returning

        # Many to many case for transformer,
        # override existing out_mapping_
        self.out_mapping_ = {}
        X_trans_inds = list(range(self.n_trans_feats_))

        # Many to many case, each ind is mapped
        # to all output'ed X_trans_inds
        for i in self.inds_:
            self.out_mapping_[i] = X_trans_inds

        # Fill the remaining spots sequentially,
        # for each of the rest inds.
        for c in range(len(self.rest_inds_)):
            ind = self.rest_inds_[c]
            self.out_mapping_[ind] = self.n_trans_feats_ + c

        # Update the original mapping, this is the mapping which
        # will be passed to the next piece of the pipeline
        update_mapping(mapping, self.out_mapping_)

        # Now return X_trans
        return X_trans

    def transform_df(self, df, base_name='transformer', encoders=None):
        return super().transform_df(df, base_name=base_name, encoders=encoders)

    def _proc_new_names(self, feat_names, base_name, encoders=None):

        # Check for one hot encoder
        if isinstance(self.estimator_, OneHotEncoder):
            new_names =\
                self._proc_one_hot_new_names(feat_names, encoders=encoders)

        # Get new names
        else:

            if len(self.inds_) == 1:
                alt_name = feat_names[self.inds_[0]]
            else:
                alt_name = base_name

            try:
                new_names = [alt_name + '_' + str(i)
                             for i in range(self.n_trans_feats_)]
            except IndexError:
                new_names = [base_name + '_' + str(i)
                             for i in range(self.n_trans_feats_)]

        # Remove old names - using parent method
        feat_names = self._remove_old_names(feat_names)

        # New names come first, then rest of names
        all_names = new_names + feat_names

        return all_names

    def _proc_one_hot_new_names(self, feat_names, encoders=None):

        def get_display_cat(name, cat):

            # If encoders passed, and name in encoder
            # use de-coded name as the cat
            if encoders is not None and name in encoders:

                try:
                    cat = encoders[name][cat]

                # If error, keep as is
                except KeyError:
                    cat = cat
                    print(cat, encoders[name])

            # If Float, turn to int before cast to str
            if isinstance(cat, float):
                cat = int(cat)

            return name + '=' + str(cat)

        new_names = []

        # If no drop
        if self.estimator_.drop_idx_ is None:
            for name_ind, category in zip(self.inds_,
                                          self.estimator_.categories_):
                name = feat_names[name_ind]
                for cat in category:
                    new_names.append(get_display_cat(name, cat))

        # Otherwise if drop index
        else:
            for name_ind, category, to_drop in zip(self.inds_,
                                                   self.estimator_.categories_,
                                                   self.estimator_.drop_idx_):
                name = feat_names[name_ind]
                for i, cat in enumerate(category):
                    if i != to_drop:
                        new_names.append(get_display_cat(name, cat))

        return new_names
