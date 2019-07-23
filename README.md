# ABCD_ML
Python based Machine Learning library for tabular Neuroimaging data, specifically geared towards the ABCD dataset.

The library is setup to be some-what modular, but the reccomended steps are shown below. Within each section, steps are ordered, as are phases, if two steps have the same number, it means it doesn't matter the order they are done.


Init Phase (1):
-----------

-Import ABCD_ML (1)
import ABCD_ML

-Initialize a main class object (2)
ML = ABCD_ML.ABCD_ML()


Data Loader Phase (2):
-----------

-Load a column name mapping (1) (optional) 
ML.load_name_map()

-Load in subjects to exclude from all analysis (1) (optional) 
ML.load_exclusions()

-Load in neuroimaging ROI data (2)
ML.load_data()

-Load in target data / the data you want to predict (2)
ML.load_targets()

-Load in 'co-variate' type data (2) (optional) 
ML.load_covars()

-Load in potential stratification values, for use in different CV stratagies (3) (optional)
ML.load_strat()


Validation Phase (3):
-----------

-Define an overarching validation stratagy (1) (optional)
ML.define_validation_strategy()

-Define a global train/test split (2) (required, but if the user really wants no testing set, then can set test_size=0)
ML.train_test_split()


Modeling Phase (3):
-----------

-Set default ML parameters for calls to Evaluate and Test (1) (optional)
ML.set_default_ML_params()

-Evaluate different choices of models and data scaling over different metrics on the training set (2)
ML.Evaluate()


Testing Phase (4):
-----------

-Test a final model/scaling stratagy as trained on the train set and evalauted on the test set
ML.Test()







