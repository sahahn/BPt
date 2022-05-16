from .feat_selectors import FeatureSelector
from .loaders import Identity, get_loader_pipe

from .mlp import MLPRegressor_Wrapper, MLPClassifier_Wrapper
from .scalers import Winsorizer
from .residualizer import LinearResidualizer

try:
    from .loaders import SingleConnectivityMeasure
except ImportError:
    class SingleConnectivityMeasure():
        pass

try:
    from .loaders import ThresholdNetworkMeasures
except ImportError:
    class ThresholdNetworkMeasures():
        pass

__all__ = ['FeatureSelector', 'get_loader_pipe',
           'Identity', 'SingleConnectivityMeasure',
           'MLPRegressor_Wrapper',
           'MLPClassifier_Wrapper', 'LinearResidualizer',
           'Winsorizer', 'ThresholdNetworkMeasures']
