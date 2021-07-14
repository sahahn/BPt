from .Dataset import Dataset
import warnings


class FakeDataset(Dataset):

    def get_cols(self, scope, limit_to=None):

        if scope == 'target':
            return ['compat']

        raise RuntimeError('get_cols not supported for FakeDataset')

    def _get_problem_type(self, col=None):
        return 'regression'

    def _get_data_inds(self, ps_scope, scope):

        # If all, return empty
        # will match
        if scope == 'all':
            return []

        # If scope matches ps_scope,
        # then treat also as all
        elif ps_scope == scope:
            return []

        # If float treat as all, but add warning
        elif scope == 'float':
            warnings.warn('Treating scope of float as "all".')
            return []

        raise RuntimeError(f'scope={scope} not supported by FakeDataset!')

# @TODO also passing no dataset here
# as long as all scopes are all, i.e., make
# a fake dataset class and it should have a function
# _get_data_inds, and _get_cols that throw errors
# when invalid. Add this option to get_estimator also.
