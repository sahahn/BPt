''' 
File with functions related to calculating score from ML models
'''
from ABCD_ML.ML_Helpers import metric_from_string

def get_regression_score(model, X_test, y_test, metric):
    '''Computes a regression score, provided a model and testing data with labels and metric'''
    
    preds = model.predict(X_test)

    metric_func = metric_from_string(metric)
    score = metric_func(y_test, preds)

    return score

def get_binary_score(model, X_test, y_test, metric):
    '''Computes a binary score, provided a model and testing data with labels and metric'''

    preds = model.predict_proba(X_test)
    preds = [p[1] for p in preds]
    
    metric_func = metric_from_string(metric)
    
    try:
        score = metric_func(y_test, preds)
    except ValueError:
        score = metric_func(y_test, np.round(preds))

    return score

def get_categorical_score(model, X_test, y_test, metric):

    pass

def get_score(problem_type,
              model,
              test_data,
              score_key,
              metric
              ):
    '''
    Compute a score for a given model with given test data,
    depending on the type of problem.
    '''

    X_test = np.array(test_data.drop(score_key, axis=1))
    y_test = np.array(test_data[score_key])
    
    if problem_type == 'regression':
        score = get_regression_score(model, X_test, y_test, metric)
    elif problem_type == 'binary':
        score = get_binary_score(model, X_test, y_test, metric)
    elif problem_type == 'categorical':
        score = get_categorical_score(model, X_test, y_test, metric)

    return score