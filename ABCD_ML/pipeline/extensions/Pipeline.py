from imblearn.pipeline import Pipeline


class ABCD_Pipeline(Pipeline):
    
    def __init__(self, steps, memory=None, verbose=False,
                 mapping=False, to_map=[]):
        
        self.mapping = mapping
        self.to_map = to_map
        
        super().__init__(steps, memory, verbose)

    def get_params(self, deep=True):
        params = super()._get_params('steps', deep=deep)
        return params

    def set_params(self, **kwargs):
        super()._set_params('steps', **kwargs)
        return self

    def fit(self, X, y=None, **fit_params):

        # If yes to mapping, then create mapping as initially 1:1
        if self.mapping:
            self._mapping = {i:i for i in range(X.shape[1])}
        else:
            self._mapping = {}

        for name in self.to_map:
            fit_params[name + '__mapping'] = self._mapping
            
        super().fit(X, y, **fit_params)