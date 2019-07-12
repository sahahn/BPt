"""
_ML.py
====================================
Main class extension file for the Machine Learning functionality
"""
import numpy as np
from ABCD_ML.ML_Helpers import compute_macro_micro
from Model import Regression_Model, Binary_Model, Categorical_Model


def Evaluate(self, problem_type, model_type,
             data_scaler='standard', n_splits=3, n_repeats=2,
             int_cv=3, metric='default', class_weight='balanced',
             random_state=None, extra_params={}):
    '''Class method to be called during the model selection phase.
    Used to evaluated different combination of models and scaling, ect...

    Parameters
    ----------
    problem_type : {'regression', 'binary', 'categorical'}

        - 'regression' : For ML on float or ordinal target score data
        - 'binary' : For ML on binary target score data
        - 'categorical' : For ML on categorical target score data,
                          as either multilabel or multiclass.

    model_type : str or list of str,
        Each string refers to a type of model to train.
        If a list of strings is passed then an ensemble model
        will be created over all individual models.
        For a full list of supported options call:
        self.show_model_types(), with optional problem type parameter.

    data_scaler : str, optional
        `data_scaler` refers to the type of scaling to apply
        to the saved data during model evaluation.
        For a full list of supported options call:
        self.show_data_scalers()
        (default = 'standard')

    n_splits : int, optional
        evaluate_model performs a repeated k-fold model evaluation,
        `n_splits` refers to the k. E.g., if set to 3, then a 3-fold
        repeated CV will be performed. This parameter is typically
        chosen as a trade off between bias and variance, in addition to
        as a function of sample size.
        (default = 3)

    n_repeats : int, optional
        evaluate_model performs a repeated k-fold model evaluation,
        `n_repeats` refers to the number of times to repeat the
        k-fold CV. This parameter is typical chosen as a balance between
        run time, and accuratly accessing model performance.
        (default = 2)

    int_cv : int, optional
        The number of internal folds to use during
        model k-fold parameter selection, if the chosen model requires
        parameter selection. A value greater
        then 2 must be passed.
        (default = 3)

    metric : str, optional
        Indicator for which metric to use for calculating
        score and during model parameter selection.
        If `metric` left as 'default', then the default metric/scorer
        for that problem types will be used.
        'regression'  : 'r2',
        'binary'      : 'roc',
        'categorical' : 'weighted roc auc'
        Note, some metrics are only avaliable for certain problem types.
        For a full list of supported metrics call:
        self.show_metrics, with optional problem type parameter.
        (default = 'default')

    class weight : {dict, 'balanced', None}, optional
        Only used for binary and categorical problem types.
        Follows sklearn api class weight behavior. Typically, either use
        'balanced' in the case of class distribution imbalance, or None.
        (default = 'balanced')

    random_state : int, RandomState instance or None, optional
        Random state, either as int for a specific seed, or if None then
        the random seed is set by np.random.
        (default = None)

    extra_params : dict, optional
        Any extra params being passed. Typically, extra params are
        added when the user wants to provide a specific model/classifier,
        or data scaler, with updated (or new) parameters.
        These can be supplied by creating another dict within extra_params.
        E.g., extra_params[model_name] = {'model_param' : new_value}
        Where model param is a valid argument for that model, and model_name in
        this case is the str indicator passed to model_type.
        (default = {})

    Returns
    ----------
    float
        The mean macro score (as set by input metric) across each
        repeated K-fold.

    float
        The standard deviation of the macro score (as set by input metric)
        across each repeated K-fold.

    float
        The mean micro score (as set by input metric) across each
        fold with the repeated K-fold.

    float
        The standard deviation of the micro score (as set by input metric)
        across each fold with the repeated K-fold.
    '''

    # Perform pre-modeling data check
    self._premodel_check()

    # Init the Model object with modeling params
    self._init_model(problem_type, model_type, data_scaler, int_cv, metric,
                     class_weight, random_state, extra_params)

    # Evaluate the model
    scores = self.Model.evaluate_model(self.all_data, self.train_subjects,
                                       n_splits, n_repeats)

    # Compute macro / micro summary of scores
    summary_scores = compute_macro_micro(scores, n_repeats, n_splits)

    self._print('Macro mean score: ', summary_scores[0])
    self._print('Macro std in score: ', summary_scores[1])
    self._print('Micro mean score: ', summary_scores[2])
    self._print('Micro std in score: ', summary_scores[3])

    # Return the computed macro and micro means and stds
    return summary_scores


def Test(self, problem_type, model_type, train_subjects=None,
               test_subjects=None, data_scaler='standard', int_cv=3,
               metric='default', class_weight='balanced',
               random_state=None, return_model=False, extra_params={}):
    '''Class method used to evaluate a specific model / data scaling
    setup on an explicitly defined train and test set.

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

    train_subjects : array-like or None, optional
        If passed None, (default), then the class defined train subjects will
        be used. Otherwise, an array or pandas Index of
        valid subjects should be passed.
        (default = None)

    test_subjects : array-like or None, optional
        If passed None, (default), then the class defined test subjects will
        be used. Otherwise, an array or pandas Index of
        valid subjects should be passed.
        (default = None)

    data_scaler : str, optional
        `data_scaler` refers to the type of scaling to apply
        to the saved data during model evaluation.
        For a full list of supported options call:
        self.show_data_scalers()
        (default = 'standard')

    int_cv : int, optional
        The number of internal folds to use during
        model k-fold parameter selection, if the chosen model requires
        parameter selection. A value greater
        then 2 must be passed.
        (default = 3)

    metric : str, optional
        Indicator for which metric to use for calculating
        score and during model parameter selection.
        If `metric` left as 'default', then the default metric/scorer
        for that problem types will be used.
        'regression'  : 'r2',
        'binary'      : 'roc',
        'categorical' : 'weighted roc auc'
        Note, some metrics are only avaliable for certain problem types.
        For a full list of supported metrics call:
        self.show_metrics, with optional problem type parameter.
        (default = 'default')

    class weight : {dict, 'balanced', None}, optional
        Only used for binary and categorical problem types.
        Follows sklearn api class weight behavior. Typically, either use
        'balanced' in the case of class distribution imbalance, or None.
        (default = 'balanced')

    random_state : int, RandomState instance or None, optional
        Random state, either as int for a specific seed, or if None then
        the random seed is set by np.random.
        (default = None)

    return_model : bool, optional
        If `return_model` is True, then model constructed and tested
        will be returned in addition to the score. If False,
        just the score will be returned.
        (default = False)

    extra_params : dict, optional
        Any extra params being passed. Typically, extra params are
        added when the user wants to provide a specific model/classifier,
        or data scaler, with updated (or new) parameters.
        These can be supplied by creating another dict within extra_params.
        E.g., extra_params[model_name] = {'model_param' : new_value}
        Where model param is a valid argument for that model, and model_name in
        this case is the str indicator passed to model_type.
        (default = {})

    Returns
    ----------
    float
        The score as determined by the passed metric/scorer on the
        provided testing set.

    model (if return_model == True)
        The sklearn api trained model object.
    '''

    # Perform pre-modeling data check
    self._premodel_check()

    # Init the Model object with modeling params
    self._init_model(problem_type, model_type, data_scaler, int_cv, metric,
                     class_weight, random_state, extra_params)

    # If not train subjects or test subjects passed, use class
    if train_subjects is None:
        train_subjects = self.train_subjects
    if test_subjects is None:
        test_subjects = self.test_subjects

    # Train the model w/ selected parameters and test on test subjects
    score = self.Model.test_model(self.all_data, train_subjects, test_subjects)

    # Optionally return the model object itself
    if return_model:
        return score, self.Model.model

    return score


def _premodel_check(self):
    '''Internal helper function to ensure that self._prepare_data()
       has been called, and to force a train/test split if not already done.
    '''

    if self.all_data is None:
        self._prepare_data()

    if self.train_subjects is None:

        print('No train-test set defined! \
              Performing one automatically with default test split =.25')
        print('If no test set is intentional, \
              call self.train_test_split(test_size=0)')

        self.train_test_split(test_size=.25)


def _init_model(self, problem_type, model_type, data_scaler, int_cv,
                metric, class_weight, random_state, extra_params):

    problem_types = {'binary': Binary_Model, 'regression': Regression_Model,
                     'categorical': Categorical_Model}

    assert problem_type in problem_types, \
        "Invalid problem type!"

    Model = problem_types[problem_type]

    self.Model = Model(self.CV, model_type, data_scaler, self.data_keys,
                       self.score_keys, self.score_encoder, int_cv, metric,
                       class_weight, random_state, self.n_jobs, extra_params,
                       self.verbose)
