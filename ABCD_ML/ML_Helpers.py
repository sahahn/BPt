"""
ML_Helpers.py
====================================
File with various ML helper functions for ABCD_ML.
These are non-class functions that are used in _ML.py and Scoring.py
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from ABCD_ML.Models import AVALIABLE


def process_model_type(problem_type, model_type, extra_params):
    '''Function to take care of processing passed in model_type to
    best select the right model type key word, and to check when
    categorical to use multilabel or multiclass models.

    Parameters
    ----------
    problem_type : {'regression', 'binary', 'categorical'}

        - 'regression' : For ML on float or ordinal target score data
        - 'binary' : For ML on binary target score data
        - 'categorical' : For ML on categorical target score data,
                          as either multilabel or multiclass.

    model_type : str or list of str
        Each string refers to a type of model to train.
        If a list of strings is passed then an ensemble model
        will be created over all individual models.
        For a full list of supported options call:
        self.show_model_types(), with optional problem type parameter.

    extra_params : dict, optional
            Any extra params being passed. Typically, extra params are
            added when the user wants to provide a specific model/classifier,
            or data scaler, with updated (or new) parameters.
            These can be supplied by creating another dict within extra_params.
            E.g., extra_params[model_name] = {'model_param' : new_value}
            Where model param is a valid argument for that model,
            and model_name in this case is the str indicator
            passed to model_type.
            (default={})

    Returns
    ----------
    str or list
        The converted model type, as str or list depending on what was
        passed in.

    dict
        The extra_params dict, as updated if neccisary to reflect params passed
        to a specific model type

    bool
        The cat_conv_flag, which if set to True means that the problem type
        is categorical, and based on the model type(s) selected,
        the score columns need to be transformed to ordinal.
    '''

    # Only in the case of categorical, with model types that just support
    # multiclass and not multilabel will this flag be set to True.
    # It means score must be converted back to ordinal
    # Will this flag be set to true, which means score must be converted
    cat_conv_flag = False

    if type(model_type) == list:
        conv_model_types = [m.replace('_', ' ').lower() for m in model_type]

        if problem_type == 'categorical':

            # Check first to see if all model names are in multilabel
            if np.array([m in AVALIABLE['categorical']['multilabel']
                         for m in conv_model_types]).all():

                conv_model_types = [AVALIABLE['categorical']['multilabel'][m]
                                    for m in conv_model_types]

            # Then check for multiclass, if multilabel not avaliable
            elif np.array([m in AVALIABLE['categorical']['multiclass']
                          for m in conv_model_types]).all():

                conv_model_types = [AVALIABLE['categorical']['multiclass'][m]
                                    for m in conv_model_types]
                cat_conv_flag = True

            else:
                assert 0 == 1, "Selected model type not avaliable."

        else:

            # Ensure for binary/regression the models passed exist,
            # and change names.
            assert np.array([m in AVALIABLE[problem_type]
                            for m in conv_model_types]).all(), \
                "Selected model type not avaliable with problem type"

            conv_model_types = [AVALIABLE[problem_type][m]
                                for m in conv_model_types]

        # If any extra params passed for the model, change to conv'ed name
        for m in range(len(conv_model_types)):
            if model_type[m] in extra_params:
                extra_params[conv_model_types[m]] = extra_params[model_type[m]]

        return conv_model_types, extra_params, cat_conv_flag

    else:
        conv_model_type = model_type.replace('_', ' ').lower()

        if problem_type == 'categorical':

            # Check multilabel first
            if conv_model_type in AVALIABLE['categorical']['multilabel']:
                conv_model_type = \
                    AVALIABLE['categorical']['multilabel'][conv_model_type]

            # Then multi class
            elif conv_model_type in AVALIABLE['categorical']['multiclass']:
                conv_model_type = \
                    AVALIABLE['categorical']['multiclass'][conv_model_type]
                cat_conv_flag = True

            else:
                assert 0 == 1, \
                    "Selected model type not avaliable with problem type"

        else:

            # Ensure for binary/regression the model passed exist,
            # and change name.
            assert conv_model_type in AVALIABLE[problem_type], \
                "Selected model type not avaliable with problem type"

            conv_model_type = AVALIABLE[problem_type][conv_model_type]

            if conv_model_type in extra_params:
                extra_params[conv_model_type] = extra_params[model_type]

        return conv_model_type, extra_params, cat_conv_flag


def get_scaler(method, extra_params=None):
    '''Returns a scaler based on the method passed,

    Parameters
    ----------
    method : str
        `method` refers to the type of scaling to apply
        to the saved data during model evaluation.
        For a full list of supported options call:
        self.show_data_scalers()

    extra_params : dict, optional
        Any extra params being passed.
        These can be supplied by creating another dict within extra_params.
        E.g., extra_params[method] = {'method param' : new_value}
        Where method param is a valid argument for that method,
        and method in this case is the str indicator.
        (default = {})

    Returns
    ----------
    scaler
        A scaler object with fit and transform methods.
    '''

    method_lower = method.lower()
    params = {}

    if method_lower == 'standard':
        scaler = StandardScaler

    elif method_lower == 'minmax':
        scaler = MinMaxScaler

    elif method_lower == 'robust':
        scaler = RobustScaler
        params = {'quantile_range': (5, 95)}

    elif method_lower == 'power':
        scaler = PowerTransformer
        params = {'method': 'yeo-johnson', 'standardize': True}

    # Check to see if user passed in params,
    # otherwise params will remain default.
    if method in extra_params:
        params.update(extra_params[method])

    scaler = scaler(**params)
    return scaler


def scale_data(train_data, test_data, data_scaler, data_keys, extra_params):
    '''
    Wrapper function to take in train/test data,
    and if applicable fit + transform a data scaler on the train data,
    and then transform the test data.

    Parameters
    ----------
    train_data : pandas DataFrame
        ABCD_ML formatted df, with the subset of training data only

    test_data : pandas DataFrame
        ABCD_ML formatted df, with the subset of testing data only

    data_scaler : str
        `data_scaler` refers to the type of scaling to apply
        to the saved data during model evaluation.
        For a full list of supported options call:
        self.show_data_scalers()

    data_keys : list or array-like
        The column names within the data to be scaled, so
        typically the neuroimaging data and not the co-variate columns.

    extra_params : dict, optional
        Any extra params being passed.
        These can be supplied by creating another dict within extra_params.
        E.g., extra_params[method] = {'method param' : new_value}
        Where method param is a valid argument for that method,
        and method in this case is the str indicator.
        (default = {})

    Returns
    ----------
    pandas DataFrame
        ABCD_ML formatted, the scaled training data

    pandas DataFrame
        ABCD_ML formatted, the scaled testing data
    '''

    if data_scaler is not None:

        scaler = get_scaler(data_scaler, extra_params)
        train_data[data_keys] = scaler.fit_transform(train_data[data_keys])
        test_data[data_keys] = scaler.transform(test_data[data_keys])

    return train_data, test_data


def compute_macro_micro(scores, n_repeats, n_splits):
    '''Compute and return scores, as computed froma repeated k-fold.

    Parameters
    ----------
    scores : list or array-like
        Should contain all of the scores
        and have a length of `n_repeats` * `n_splits`

    n_repeats : int
        The number of repeats

    n_splits : int
        The number of splits per repeat

    Returns
    ----------
    float
        The mean macro score

    float
        The standard deviation of the macro score

    float
        The mean micro score

    float
        The standard deviation of the micro score
    '''

    scores = np.array(scores)
    macro_scores = np.mean(np.reshape(scores, (n_repeats, n_splits)), axis=1)

    return (np.mean(macro_scores), np.std(macro_scores),
            np.mean(scores), np.std(scores))
