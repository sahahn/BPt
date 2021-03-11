from .FeatSelectors import RFEWrapper, FeatureSelector
from .loaders import Identity, SurfLabels, SurfMaps

try:
    from .loaders import Connectivity
except ImportError:
    class Connectivity():
        pass

try:
    from .loaders import ThresholdNetworkMeasures
except ImportError:
    class ThresholdNetworkMeasures():
        pass

from .MLP import MLPRegressor_Wrapper, MLPClassifier_Wrapper
from .random_parcellation import RandomParcellation
from .Scalers import Winsorizer
from .residualizer import LinearResidualizer

__all__ = ['RFEWrapper',
           'FeatureSelector',
           'Identity', 'SurfLabels', 'Connectivity', 'MLPRegressor_Wrapper',
           'MLPClassifier_Wrapper', 'LinearResidualizer',
           'RandomParcellation', 'Winsorizer',
           'ThresholdNetworkMeasures', 'SurfMaps']
