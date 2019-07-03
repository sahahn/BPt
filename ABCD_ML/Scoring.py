''' 
File with functions related to calculating score from ML models
'''
from ABCD_ML.ML_Helpers import metric_from_string

def get_regression_score(model, X_test, y_test, metric):
    '''Computes a regression score, provided a model and testing data with labels and metric'''
    
    preds = model.predict(X_test)
    metric_func = metric_from_string(metric)
    
    score = use_metric(metric_func, y_test, preds)
    return score

def get_binary_score(model, X_test, y_test, metric):
    '''Computes a binary score, provided a model and testing data with labels and metric'''

    #Get the metric function and if metric needs proba, or preds
    metric_func, needs_proba = metric_from_string(metric, return_needs_proba=True)

    if needs_proba:
        preds = model.predict_proba(X_test)[:,1]
    else:
        preds = model.predict(X_test)

    score = use_metric(metric_func, y_test, preds)
    return score

def get_categorical_score(model, X_test, y_test, metric):
    '''Computes a categorical score, provided a model and testing data with labels and metric'''

    metric_func, needs_proba = metric_from_string(metric, return_needs_proba=True)

    if needs_proba:
        pred_proba = clf.predict_proba(X_test)
        preds = np.stack([p[:,1] for p in pred_proba], axis=1)
    else:
        preds = model.predict(X_test)

    score = use_metric(metric_func, y_test, preds)
    return score


def use_metric(metric_func, y_test, preds):
    '''
    Check for extra metric params wrapped up in function to compute actual metric.
    Returns score.
    '''

    metric_params = {}

    if type(metric_func) == list:
        metric_params = metric_func[1]
        metric_func = metric_func[0]
    
    score = metric_func(y_test, preds, **metric_params)
    return score

def get_score(problem_type,
              model,
              test_data,
              score_key,
              metric,
              score_encoder=None
              ):
    '''
    Computes a score for a given model with given test data.

    Parameters
    ----------
    problem_type : str, either 'regression', 'binary' or 'categorical'

    model : sklearn model,
        A trained model must be passed in to be evaluated.

    test_data : pandas DataFrame,
        ABCD_ML formatted df.

    score_key : str or numpy array,
        The score column key or keys within test_data

    metric : str,
        Indicator for which metric to use for parameter selection
        and model evaluation. For a full list of supported metrics,
        call self.show_metrics(problem_type=problem_type)

    score_encoder : sklearn encoder, optional (default=None)
        A sklearn api encoder, for optionally transforming the target
        variable. Used either in the case of categorical data in converting from
        one-hot encoding to ordinal, or in the case of a target transformation in
        a regression problem.

    Returns
    -------
    score : returns a score for whatever metric was passed
    '''

    X_test = np.array(test_data.drop(score_key, axis=1))
    y_test = np.array(test_data[score_key])

    #If a score encoder is passed, transform the encoded y back for testing
    if score_encoder is not None:
        y_test = score_encoder.inverse_transform(y_test).squeeze()
    
    if problem_type == 'regression':
        score = get_regression_score(model, X_test, y_test, metric)
    elif problem_type == 'binary':
        score = get_binary_score(model, X_test, y_test, metric)
    elif problem_type == 'categorical':
        score = get_categorical_score(model, X_test, y_test, metric)

    return score
