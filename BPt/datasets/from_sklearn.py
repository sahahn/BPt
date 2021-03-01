
import pandas as pd
from BPt import Dataset


def load_boston():
    '''Loads the boston dataset using the sklearn helper function
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html#sklearn.datasets.load_boston
    and returns it as a BPt Dataset.'''

    from sklearn.datasets import load_boston

    # Load as dataframe
    boston_data = load_boston()
    df = pd.DataFrame(boston_data['data'],
                      columns=boston_data['feature_names'])
    df['PRICE'] = boston_data['target']

    # Convert to dataset
    dataset = Dataset(df)
    dataset = dataset.set_role('PRICE', 'target')

    # Set test split
    dataset = dataset.set_test_split(.2, random_state=3)

    return dataset
