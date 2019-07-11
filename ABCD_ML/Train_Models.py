'''
ABCD_ML Project

Scripts for training models
'''
import numpy as np

from ABCD_ML.Models import MODELS

from ABCD_ML.Train_Light_GBM import Train_Light_GBM
from ABCD_ML.Scoring import scorer_from_string


def replace(params, base_int_cv, scorer, class_weight, n_jobs, estimator=None, base_model=False):

    if 'cv' in params:
        if params['cv'] == 'base_int_cv':
            params['cv'] = base_int_cv
    
    if 'scoring' in params:
        if params['scoring'] == 'scorer':
            params['scoring'] = scorer
    
    if 'class_weight' in params:
        if params['class_weight'] == 'class_weight':
            params['class_weight'] = class_weight
    
    if 'n_jobs' in params:
        if params['n_jobs'] == 'n_jobs':
            if base_model:
                del params['n_jobs']
            else:
                params['n_jobs'] = n_jobs
    
    if 'estimator' in params:
        if type(params['estimator']) == str and estimator is not None:
            params['estimator'] = estimator

    return params

def get_model(model_type, base_int_cv, scorer, class_weight, n_jobs, extra_params, base_model=False):

    print(model_type)

    estimator = None

    #If gs or rs in name (grid or random search) recursively build the base_model / estimator
    if ' gs' in model_type or ' rs' in model_type:
        
        base_model_type = MODELS[model_type][1]['estimator']
        estimator = get_model(base_model_type, base_int_cv, scorer, class_weight, n_jobs, extra_params, base_model=True)

    #Grab the right model and params
    model, model_params = MODELS[model_type][0], MODELS[model_type][1].copy()
    model_params = replace(model_params, base_int_cv, scorer, class_weight, n_jobs, estimator, base_model=base_model)

    #Check to see if there are any user passed model params to update
    ex_params = {}
    if model_type in extra_params:
        ex_params = extra_params[model_type]
    model_params.update(ex_params)

    #Create model
    model = model(**model_params)

    return model

def train_model(data,
                score_key,
                CV,
                model_type,
                int_cv,
                metric,
                class_weight='balanced',
                random_state=None,
                score_encoder=None,
                n_jobs=1,
                extra_params={}
                ):

    '''Function for training a specific model.

    Parameters
    ----------
    data : pandas DataFrame
        ABCD_ML formatted df.

    score_key : str or list
        The name(s) of the column(s) within data to be
        set as score.

    CV : ABCD_ML CV
        A custom ABCD_ML CV object, which stores the
        desired split behavior

    model_type : str
        Where each string refers to a type of model to train.
        Assumes final processed model_type names here.
        For a full list of supported options call:
        self.show_model_types(), with optional problem type parameter.

    int_cv : int
        The number of internal folds to use during
        model k-fold parameter selection, if the chosen model requires
        parameter selection. A value greater
        then 2 must be passed.

    metric : str
        Indicator for which metric to use for calculating
        score and during model parameter selection.
        Note, some metrics are only avaliable for certain problem types.
        For a full list of supported metrics call:
        self.show_metrics, with optional problem type parameter.

    class weight : {dict, 'balanced', None}, optional (default='balanced')
        Only used for binary and categorical problem types.
        Follows sklearn api class weight behavior. Typically, either use
        'balanced' in the case of class distribution imbalance, or None.

    random_state : int, or None, optional (default=None)
        Random state, either as int for a specific seed,
        or if None then the random seed is set by np.random.

    score_encoder : sklearn encoder, optional (default=None)
        A sklearn api encoder, for optionally transforming the target
        variable. Used in the case of categorical data in converting from
        one-hot encoding to ordinal.

    n_jobs : int, optional (default = 1)
        Number of processors to use during training.

    extra_params : dict, optional (default={})
        Any extra params being passed. Typically, extra params are
        added when the user wants to provide a specific model/classifier,
        or data scaler, with updated (or new) parameters.
        These can be supplied by creating another dict within extra_params.
        E.g., extra_params[model_name] = {'model_param' : new_value}
        Where model param is a valid argument for that model,
        and model_name in this case is the str indicator
        passed to model_type.
    '''
    
    #Create the internal base k-fold and scorer
    base_int_cv = CV.k_fold(data.index, int_cv, random_state=random_state, return_index=True)
    scorer = scorer_from_string(metric)

    #Create the model
    model = get_model(model_type, base_int_cv, scorer, class_weight, n_jobs, extra_params)

    #Fit the model
    X, y = np.array(data.drop(score_key, axis=1)), np.array(data[score_key])

    #If a score encoder is passed, inverse transform y back to ordinal
    if score_encoder is not None:
        y = score_encoder.inverse_transform(y).squeeze()

    model.fit(X, y)

    return model