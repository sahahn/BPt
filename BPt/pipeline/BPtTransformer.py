from .helpers import update_mapping
from .ScopeObjs import ScopeTransformer
from sklearn.preprocessing import OneHotEncoder


class BPtTransformer(ScopeTransformer):

    def fit(self, X, y=None, mapping=None,
            fit_index=None, **fit_params):

        # Need the output from a transform to full fit,
        # so when fit is called, call fit_transform instead
        self.fit_transform(X=X, y=y, mapping=mapping,
                           fit_index=fit_index,
                           **fit_params)

        return self

    def _update_transformer_mapping(self, mapping):

        # Need to update the mapping before returning

        # Many to many case for transformer,
        # override existing out_mapping_
        self.out_mapping_ = {}
        X_trans_inds = list(range(self.n_trans_feats_))

        # Many to many case, each ind is mapped
        # to all outputted X_trans_inds
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

        return self

    def _all_case_update_transformer_mapping(self, X, mapping):

        # Get as list of
        X_trans_inds = list(range(self.n_trans_feats_))

        # All case out mapping
        self.out_mapping_ = {i: X_trans_inds for i in range(X.shape[1])}

        # Since no rest inds, update mapping
        update_mapping(mapping, self.out_mapping_)

        return self

    def fit_transform(self, X, y=None, mapping=None,
                      fit_index=None,
                      transform_index=None, **fit_params):

        if mapping is None:
            mapping = {}

        # Call parent fit
        super().fit(X, y=y, mapping=mapping,
                    fit_index=fit_index,
                    **fit_params)

        # If skip
        if self.estimator_ is None:
            return X

        # Transform X - since fit_transform, index is fit index
        X_trans = self.transform(X, transform_index=fit_index)

        # Update mapping and set out_mapping_
        # special all case
        if self.inds_ is Ellipsis:
            self._all_case_update_transformer_mapping(X, mapping)
        else:
            self._update_transformer_mapping(mapping)

        # Now return X_trans
        return X_trans

    def transform_df(self, df, base_name='transformer', encoders=None):
        return super().transform_df(df, base_name=base_name, encoders=encoders)

    def _proc_new_names(self, feat_names, base_name, encoders=None):

        # If skip, return passed names as is
        if self.estimator_ is None:
            return feat_names

        # Check for one hot encoder
        if isinstance(self.estimator_, OneHotEncoder):
            new_names =\
                self._proc_one_hot_new_names(feat_names, encoders=encoders)

        # Get new names
        else:

            if self.inds_ is Ellipsis:
                alt_name = base_name
            elif len(self.inds_) == 1:
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

            # If Float, turn to int before cast to str
            if isinstance(cat, float):
                cat = int(cat)

            return str(name) + '=' + repr(cat)

        # Scope all case, set inds as identity
        # over all passed feat names
        if self.inds_ is Ellipsis:
            inds = list(range(len(feat_names)))

        # Otherwise use self.inds_
        else:
            inds = self.inds_

        # Save new names in new_names
        new_names = []

        # If no drop
        if self.estimator_.drop_idx_ is None:
            for name_ind, category in zip(inds,
                                          self.estimator_.categories_):
                name = feat_names[name_ind]
                for cat in category:
                    new_names.append(get_display_cat(name, cat))

        # Otherwise if drop index
        else:
            for name_ind, category, to_drop in zip(inds,
                                                   self.estimator_.categories_,
                                                   self.estimator_.drop_idx_):
                name = feat_names[name_ind]
                for i, cat in enumerate(category):
                    if i != to_drop:
                        new_names.append(get_display_cat(name, cat))

        return new_names
