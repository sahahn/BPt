from ..helpers.ML_Helpers import get_obj_and_params, proc_mapping, update_mapping, show_objects
import numpy as np
from .Transformers import Transformer_Wrapper
from .extensions.Loaders import Identity
from joblib import Parallel, delayed

class Loader_Wrapper(Transformer_Wrapper):

    def __init__(self, wrapper_transformer,
                 wrapper_inds, file_mapping,
                 wrapper_n_jobs=1, **params):

        super().__init__(wrapper_transformer,
                         wrapper_inds, **params)

        self.file_mapping = file_mapping
        self.wrapper_n_jobs = wrapper_n_jobs
    
    def _fit(self, X, y=None):
        
        fit_fm_key = X[0, self.wrapper_inds[0]]
        fit_data = self.file_mapping[int(fit_fm_key)].load()

        self.wrapper_transformer.fit(fit_data, y)

        return self
        
    def fit_transform(self, X, y=None, mapping={}):

        # If any changes to mapping, update
        self._proc_mapping(mapping)
        
        # Fit on the first data-point only
        self._fit(X, y)
        
        # Transform X
        X_trans, self._X_trans_inds = self._get_X_trans(X)
        
        # Then create new mapping
        new_mapping = {}
        
        # Add changed X_trans by col
        for c in range(len(self.wrapper_inds)):
            ind = self.wrapper_inds[c]
            new_mapping[ind] = self._X_trans_inds[c]
            
        # Update rest of inds, as just shifted over
        rest_inds = [i for i in range(X.shape[1])
                     if i not in self.wrapper_inds]
        
        for c in range(len(rest_inds)):
            ind = rest_inds[c]
            new_mapping[ind] = X_trans.shape[1] + c
        
        # Update mapping
        update_mapping(mapping, new_mapping)
        
        return np.hstack([X_trans, X[:, rest_inds]])
    
    def _get_trans_chunk(self, data_files):
        
        X_trans_chunk = []
        
        for data_file in data_files:
            
            data = data_file.load()
            trans_data = np.squeeze(self.wrapper_transformer.transform(data))
            X_trans_chunk.append(trans_data)
        
        return X_trans_chunk
    
    def get_chunks(self, data_files):

        per_chunk = len(data_files) // self.wrapper_n_jobs
        chunks = [list(range(i * per_chunk, (i+1) * per_chunk))
                  for i in range(self.wrapper_n_jobs)]

        last = chunks[-1][-1]
        chunks[-1] += list(range(last+1, len(data_files)))
        return [[data_files[i] for i in c] for c in chunks]
    
    def _get_trans_col(self, fm_keys):
        
        data_files = [self.file_mapping[int(fm_key)] for fm_key in fm_keys]
        
        if self.wrapper_n_jobs == 1:
             X_trans_cols = self._get_trans_chunk(data_files)
          
        else:
            chunks = self.get_chunks(data_files)
            
            X_trans_chunks =\
                Parallel(n_jobs=self.wrapper_n_jobs)(
                    delayed(self._get_trans_chunk)(
                        data_files=chunk) for chunk in chunks)
            
            X_trans_cols = []
            for chunk in X_trans_chunks:
                X_trans_cols += chunk
        
        return X_trans_cols
    
    def _get_X_trans(self, X):
        
        # Init lists + mappings
        X_trans, X_trans_inds = [], []
        cnt = 0
        
        # For each column to fit_transform
        for col in self.wrapper_inds:
            
            # Get transformer column
            fm_keys = [key for key in X[:,col]]
            X_trans_cols = self._get_trans_col(fm_keys)
                
            # Stack + append new features
            X_trans_cols = np.stack(X_trans_cols)
            X_trans.append(X_trans_cols)
            
            # Add + append inds
            X_trans_cols_inds = [i for i in range(cnt, X_trans_cols.shape[1] + cnt)]
            X_trans_inds.append(X_trans_cols_inds)
            
            # Update cnt
            cnt = X_trans_cols.shape[1] + cnt
            
        # Stack final
        X_trans = np.hstack(X_trans)
        
        return X_trans, X_trans_inds

    def transform(self, X):
        
        # Transform X
        X_trans, _ = self._get_X_trans(X)
        rest_inds = [i for i in range(X.shape[1])
                     if i not in self.wrapper_inds]
        
        return np.hstack([X_trans, X[:, rest_inds]])

    def _get_new_df_names(self, feat_names):
        '''Create new feature names for the transformed features,
        in loaders this is done per feature/column'''

        new_names = []
        for c in range(len(self.wrapper_inds)):

            ind = self.wrapper_inds[c]
            base_name = feat_names[ind]

            new_inds = self._X_trans_inds[c]
            new_names += [base_name + '_' + str(i) for i in range(len(new_inds))]

        return new_names

    def set_params(self, **params):

        if 'file_mapping' in params:
            self.file_mapping = params.pop('file_mapping')
            
        if 'wrapper_n_jobs' in params:
            self.wrapper_n_jobs = params.pop('wrapper_n_jobs')

        return super().set_params(**params)

    def get_params(self, deep=False):

        params = super().get_params()
        params['file_mapping'] = self.file_mapping
        params['wrapper_n_jobs'] = self.wrapper_n_jobs

        return params


LOADERS = {
    'identity': (Identity, ['default']),
}


def get_loader_and_params(loader_str, extra_params, params, search_type,
                          random_state=None, num_feat_keys=None):

    loader, extra_loader_params, loader_params =\
        get_obj_and_params(loader_str, LOADERS, extra_params, params,
                           search_type)

    return loader(**extra_loader_params), loader_params


def Show_Loaders(self, loader=None, show_params_options=False,
                 show_object=False,
                 show_all_possible_params=False):
    '''Print out the avaliable data loaders.

    Parameters
    ----------
    loader : str or list, optional
        Provide a str or list of strs, where
        each str is the exact loader str indicator
        in order to show information for only that (or those)
        data loaders

    show_params_options : bool, optional
        Flag, if set to True, then will display the ABCD_ML
        param ind options for each data loader.

        (default = False)

    show_object : bool, optional
        Flag, if set to True, then will print the raw data loader
        object.

        (default = False)

    show_all_possible_params: bool, optional
        Flag, if set to True, then will print all
        possible arguments to the classes __init__

        (default = False)
    '''


    show_objects(problem_type=None, obj=loader,
                 show_params_options=show_params_options,
                 show_object=show_object,
                 show_all_possible_params=show_all_possible_params,
                 AVALIABLE=None, OBJS=LOADERS)