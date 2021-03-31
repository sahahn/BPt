from .FeatSelectors import FeatureSelector
from .loaders import Identity, SurfLabels, SurfMaps

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

from .MLP import MLPRegressor_Wrapper, MLPClassifier_Wrapper
from .random_parcellation import RandomParcellation
from .Scalers import Winsorizer
from .residualizer import LinearResidualizer


__all__ = ['FeatureSelector',
           'Identity', 'SurfLabels', 'SingleConnectivityMeasure',
           'MLPRegressor_Wrapper',
           'MLPClassifier_Wrapper', 'LinearResidualizer',
           'RandomParcellation', 'Winsorizer',
           'ThresholdNetworkMeasures', 'SurfMaps']
