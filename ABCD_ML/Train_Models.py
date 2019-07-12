'''
ABCD_ML Project

Scripts for training models
'''
import numpy as np
from ABCD_ML.Models import MODELS
from ABCD_ML.Train_Light_GBM import Train_Light_GBM


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

