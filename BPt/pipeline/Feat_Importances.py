import pandas as pd
import numpy as np

from .Perm_Feat_Importance import Perm_Feat_Importance
from sklearn.inspection import permutation_importance
from ..helpers.ML_Helpers import get_obj_and_params


class Feat_Importances():

    def __init__(self, importance_info, params, n_jobs, scorer):

        self.name = importance_info['name']
        self.scopes = importance_info['scopes']
        self.split = importance_info['split']

        # Unpack params
        self.shap_params = params.shap_params
        self.n_perm = params.n_perm

        self.inverse_global = params.inverse_global
        self.inverse_local = params.inverse_local

        self.valid = True
        self.test = False

        self.n_jobs = n_jobs
        self.scorer = scorer

        self.global_df = None
        self.local_df = None
        self.local_dfs = []

        self.inverse_global_fis = []
        self.inverse_local_fis = []
        self.warning = False

    def get_name(self):

        name = self.name[0].upper() + self.name[1:]
        return name

    def get_data_needed_flags(self, flags):

        # Save flag info
        self.flags = flags

        # Proc by combination of flags and type of feat importance
        if self.name == 'base':

            if flags['tree'] or flags['linear']:
                return False
            else:
                self.valid = False
                return False

        # If shap, return True
        elif self.name == 'shap':

            tfp =\
                self.shap_params.tree_feature_perturbation == 'interventional'
            if flags['tree'] and tfp:
                return False
            return True

    def init_global(self, X, y):

        if not self.valid:
            return

        if 'global' in self.scopes:

            feat_names = list(X)
            self.global_df = pd.DataFrame(columns=feat_names)

    def init_local(self, X, y, test=False, n_splits=None):

        if not self.valid:
            return

        if test:
            self.test = True

        if 'local' in self.scopes:

            # If test or, split as test,
            if self.split == 'test' or self.test:

                self.local_df = X.copy()
                for col in self.local_df.columns:
                    self.local_df[col].values[:] = 0

            # Train or both
            else:

                self.local_df = []
                for i in range(n_splits):

                    df = X.copy()
                    for col in df.columns:
                        df[col].values[:] = 0

                    self.local_df.append(df)

    def get_avg(self, dfs):

        if isinstance(dfs, list):

            df = dfs[0].copy()

            df_arrays = [np.array(df) for df in dfs]
            mean_array = np.nanmean(df_arrays, axis=0)

            # Set to average
            df[list(df)] = mean_array
            return df

        else:
            return dfs

    def add_to_global(self, feat_names, feat_imps):

        feat_imp_dict = {name: imp for name, imp in zip(feat_names, feat_imps)}

        self.global_df = self.global_df.append(feat_imp_dict,
                                               ignore_index=True)
        self.global_df = self.global_df.fillna(0)

    def _add_to_local(self, X_test, vals, fold, class_df):

        val_df = X_test.copy()
        val_df[list(X_test)] = vals

        if self.split == 'test':
            class_df.update(val_df)

        elif self.split == 'train':
            class_df[fold].update(val_df)

            # Replace any rows with all zeros as NaN instead of 0
            ind_df = self.local_df[fold]
            to_nan = ind_df[ind_df.sum(axis=1) == 0].index
            class_df[fold].loc[to_nan] = np.nan

        elif self.split == 'all':
            class_df[fold].update(val_df)

    def add_to_local(self, X_test, vals, fold):
        self._add_to_local(X_test, vals, fold, self.local_df)

    def proc_local(self):

        if not self.valid:
            return

        if 'local' in self.scopes:

            # Get avg then save in self.local_dfs
            df = self.get_avg(self.local_df)
            self.local_dfs.append(df)

            # Reset to None once added
            self.local_df = None

    def proc_importances(self, base_model, X_test, y_test=None,
                         X_train=None, fold=0, random_state=None):
        '''X_test should be a df, and X_train either None or as np array.'''

        if not self.valid:
            return None, None

        if self.name == 'base':
            feat_imps = self.get_base_feat_importances(base_model)
            self.add_to_global(list(X_test), feat_imps)
            return feat_imps, None

        elif self.name == 'perm':
            feat_imps = self.get_perm_feat_importances(base_model,
                                                       np.array(X_test),
                                                       y_test)
            self.add_to_global(list(X_test), feat_imps)
            return feat_imps, None

        elif self.name == 'sklearn perm':
            feat_imps = self.get_perm_feat_importances2(base_model,
                                                        np.array(X_test),
                                                        y_test,
                                                        random_state)
            self.add_to_global(list(X_test), feat_imps)
            return feat_imps, None

        elif self.name == 'shap':
            shap_vals = self.get_shap_feature_importance(base_model, X_test,
                                                         X_train)
            global_shap_vals = self.global_from_local(shap_vals)

            # Add to local
            self.add_to_local(X_test, shap_vals, fold)

            # Add to global
            self.add_to_global(list(X_test), global_shap_vals)
            return global_shap_vals, shap_vals

    def global_from_local(self, vals):
        return self.col_abs_mean(vals)

    def col_abs_mean(self, vals):

        if self.shap_params.avg_abs:
            col_means = np.mean(np.abs(vals), axis=0)

        else:
            col_means = np.mean(vals, axis=0)

        return col_means

    def get_base_feat_importances(self, base_model):

        if self.flags['linear']:
            feat_imps = np.squeeze(base_model.coef_)
        elif self.flags['tree']:
            feat_imps = base_model.feature_importances_

        return feat_imps

    def get_perm_feat_importances(self, base_model, X_test,
                                  y_test):

        perm_import = Perm_Feat_Importance(n_perm=self.n_perm,
                                           n_jobs=self.n_jobs)
        feat_imps = perm_import.compute(base_model, self.scorer,
                                        X_test, y_test)
        return feat_imps

    def get_perm_feat_importances2(self, base_model, X_test, y_test,
                                   random_state):

        results = permutation_importance(base_model, X_test, y_test,
                                         scoring=self.scorer,
                                         n_repeats=self.n_perm,
                                         n_jobs=self.n_jobs,
                                         random_state=random_state)
        return results.importances_mean

    def get_shap_feature_importance(self, base_model, X_test, X_train):

        try:
            import shap
        except ImportError:
            raise ImportError('You must have shap installed to use shap')

        if self.flags['tree'] or self.flags['linear']:

            if self.flags['linear']:

                fp = self.shap_params.linear_feature_perturbation
                n = self.shap_params.linear_nsamples
                explainer = shap.LinearExplainer(base_model, X_train,
                                                 nsamples=n,
                                                 feature_perturbation=fp)

                shap_values = explainer.shap_values(X_test)

            elif self.flags['tree']:

                tmo = self.shap_params.tree_model_output
                tfp = self.shap_params.tree_feature_perturbation
                explainer =\
                    shap.TreeExplainer(
                        base_model, X_train,
                        model_output=tmo,
                        feature_perturbation=tfp)

                ttl = self.shap_params.tree_tree_limit
                shap_values =\
                    explainer.shap_values(X_test,
                                          tree_limit=ttl)

        # Kernel
        else:

            nkmean = self.shap_params.kernel_nkmean

            if nkmean is not None:
                X_train_summary = shap.kmeans(X_train, nkmean)
            else:
                X_train_summary = X_train

            explainer =\
                self.get_kernel_explainer(base_model, X_train_summary,
                                          self.shap_params.kernel_link)

            klr = self.shap_params.kernel_l1_reg
            kns = self.shap_params.kernel_nsamples

            shap_values =\
                explainer.shap_values(np.array(X_test),
                                      l1_reg=klr,
                                      n_samples=kns)

        return self.proc_shap_vals(shap_values)

    def proc_shap_vals(self, shap_values):
        return shap_values

    def get_kernel_explainer(self, model, X_train_summary, link):

        try:
            import shap
        except ImportError:
            raise ImportError('You must have shap installed to use shap')

        if link == 'default':
            link = 'logic'

        explainer = shap.KernelExplainer(model.predict_proba, X_train_summary,
                                         link=link)

        return explainer

    def set_final_local(self):

        if 'local' in self.scopes:

            df = self.get_avg(self.local_dfs)
            self.local_df = df

        self.local_dfs = None  # Save memory

    def get_global_label(self):

        if self.name == 'base':
            if self.flags['linear']:
                return 'Mean Beta Weight'
            elif self.flags['tree']:
                return 'Mean Tree Split Based Importance'

        elif self.name == 'shap':
            return 'Mean Shap Value'

    def _get_warning_str(self, as_html=False):

        lb = '\n'
        if as_html:
            lb = '<br>'

        content = 'Warning! Base pipeline contains either loaders '
        content += 'or transformers. Any self.global_df or self.local_df '
        content += 'might therefore be incorrect, or atleast hard to '
        content += 'make sense of. In this case you should likely use the '
        content += 'the inverse feat importances!' + lb

        return content

    def _repr_html_(self):

        content = 'Feature_Importances Object<br>'

        if not self.valid:
            return content + 'Invalid type for passed classifier!<br>'

        content += 'name: ' + self.name + '<br>'

        if self.name == 'base':

            content += 'importance type: '
            if self.flags['linear']:
                content += 'beta weight <br>'
            elif self.flags['tree']:
                content += 'tree based <br>'

        if self.warning:

            content += self._get_warning_str(as_html=True)

            if len(self.inverse_global_fis) > 0:
                content += 'contains inverse global feat importances: '
                content += 'self.inverse_global_fis <br>'
            if len(self.inverse_local_fis) > 0:
                content += 'contains inverse local feat importances: '
                content += 'self.inverse_local_fis <br>'

        else:
            if 'global' in self.scopes:
                content += 'contains global feat importances: '
                content += 'self.global_df <br>'
            if 'local' in self.scopes:
                content += 'contains local feat importances: '
                content += 'self.local_df <br>'

        content += 'importance computed on: ' + self.split + '<br>'
        return content

    def set_target(self, target):
        self.target = target

    def set_run_name(self, run_name):
        self.run_name = run_name


class Regression_Feat_Importances(Feat_Importances):

    def get_kernel_explainer(self, model, X_train_summary, link):

        try:
            import shap
        except ImportError:
            raise ImportError('You must have shap installed to use shap')

        if link == 'default':
            link = 'identity'

        explainer = shap.KernelExplainer(model.predict, X_train_summary,
                                         link=link)

        return explainer


class Cat_Feat_Importances(Feat_Importances):

    def init_global(self, X, y):

        if not self.valid:
            return

        if 'global' in self.scopes:

            self.n_classes = len(np.unique(y))
            self.global_df = []
            feat_names = list(X)

            for j in range(self.n_classes):
                self.global_df.append(pd.DataFrame(columns=feat_names))

    def init_local(self, X, y=None, test=False, n_splits=None):

        if not self.valid:
            return

        if test:
            self.test = True

        if 'local' in self.scopes:

            self.local_df = []
            self.n_classes = len(np.unique(y))

            # If test or, split as test,
            if self.split == 'test' or self.test:

                for j in range(self.n_classes):

                    df = X.copy()

                    for col in df.columns:
                        df[col].values[:] = 0

                    self.local_df.append(df)

            # Train or both
            else:

                for j in range(self.n_classes):
                    dfs = []

                    for i in range(n_splits):

                        df = X.copy()
                        for col in df.columns:
                            df[col].values[:] = 0

                        dfs.append(df)

                    self.local_df.append(dfs)

    def add_to_global(self, feat_names, feat_imps):

        # In cases like base feat importance for trees
        # just set each class to the global values
        if len(feat_names) == len(feat_imps):
            feat_imps = [feat_imps for j in range(self.n_classes)]

        for j in range(self.n_classes):

            feat_imp_dict = {name: imp for name, imp in
                             zip(feat_names, feat_imps[j])}

            self.global_df[j] = self.global_df[j].append(feat_imp_dict,
                                                         ignore_index=True)
            self.global_df[j] = self.global_df[j].fillna(0)

    def add_to_local(self, X_test, vals, fold):

        for j in range(self.n_classes):
            self._add_to_local(X_test, vals[j], fold, self.local_df[j])

    def global_from_local(self, vals):

        global_vals = []

        for j in range(self.n_classes):
            global_vals.append(self.col_abs_mean(vals[j]))

        return global_vals

    def proc_local(self):

        if not self.valid:
            return

        if 'local' in self.scopes:

            dfs = []
            for j in range(self.n_classes):
                dfs.append(self.get_avg(self.local_df[j]))

            self.local_dfs.append(dfs)

            # Reset to None once added
            self.local_df = None

    def set_final_local(self):

        if not self.valid:
            return

        if 'local' in self.scopes:

            if len(self.local_dfs) > 0:

                dfs = []

                for j in range(self.n_classes):
                    class_dfs = [d[j] for d in self.local_dfs]
                    dfs.append(self.get_avg(class_dfs))

                self.local_df = dfs

            else:
                self.local_df = self.local_dfs[0]

        self.local_dfs = None  # Save memory


IMPORTANCES = {
    'base': ({'name': 'base',
              'scopes': ('global'),
              'split': 'train'}, ['default']),

    'shap': ({'name': 'shap',
              'scopes': ('local', 'global'),
              'split': 'test'}, ['base shap']),

    'shap train': ({'name': 'shap',
                    'scopes': ('local', 'global'),
                    'split': 'train'}, ['base shap']),

    'shap all': ({'name': 'shap',
                  'scopes': ('local', 'global'),
                  'split': 'all'}, ['base shap']),

    'perm': ({'name': 'perm',
              'scopes': ('global'),
              'split': 'test'}, ['base perm']),

    'perm train': ({'name': 'perm',
                    'scopes': ('global'),
                    'split': 'train'}, ['base perm']),

    'perm all': ({'name': 'perm',
                  'scopes': ('global'),
                  'split': 'all'}, ['base perm']),

    'sklearn perm': ({'name': 'sklearn perm',
                      'scopes': ('global'),
                      'split': 'test'}, ['base perm']),

    'sklearn perm train': ({'name': 'sklearn perm',
                            'scopes': ('global'),
                            'split': 'train'}, ['base perm']),

    'sklearn perm all': ({'name': 'sklearn perm',
                          'scopes': ('global'),
                          'split': 'all'}, ['base perm']),
}


def get_feat_importances_and_params(params, problem_type, n_jobs, scorer):

    # Search type should always be None
    imp_info, _, _ =\
        get_obj_and_params(params.obj, IMPORTANCES, {}, params.params)

    problem_types = {'binary': Feat_Importances,
                     'regression': Regression_Feat_Importances,
                     'categorical': Cat_Feat_Importances}

    # Get right base object by problem type
    FI = problem_types[problem_type]

    # Return the instance of the feat_importance obj
    return FI(imp_info, params, n_jobs, scorer)
