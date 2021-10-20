
import pandas as pd
from BPt import Dataset


def load_cali():
    '''Loads the california dataset using the sklearn helper function
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
    and returns it as a BPt Dataset. This is just used for examples
    '''
    from sklearn.datasets import fetch_california_housing

    # Load as dataframe
    cali_data = fetch_california_housing()
    df = pd.DataFrame(cali_data['data'],
                      columns=cali_data['feature_names'])
    df['MedHouseVal'] = cali_data['target']

    # Convert from df to dataset
    dataset = Dataset(df)
    dataset = dataset.set_role('MedHouseVal', 'target')

    # Set test split
    dataset = dataset.set_test_split(.2, random_state=3)

    return dataset
