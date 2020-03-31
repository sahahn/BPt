from sklearn.neural_network import MLPRegressor, MLPClassifier
import numpy as np

class MLPRegressor(MLPRegressor):

    def _fit(self, X, y, incremental=False):

        self.hidden_layer_sizes = np.array(self.hidden_layer_sizes).astype(int)
        super()._fit(X, y, incremental=False)

class MLPClassifier(MLPClassifier):

    def _fit(self, X, y, incremental=False):

        self.hidden_layer_sizes = np.array(self.hidden_layer_sizes).astype(int)
        super()._fit(X, y, incremental=False)

