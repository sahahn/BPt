"""
_ML.py
====================================
Main class extension file for the Machine Learning functionality
"""
import numpy as np
from ABCD_ML.Ensemble_Model import Ensemble_Model
from ABCD_ML.Train_Models import train_model
from ABCD_ML.ML_Helpers import (scale_data, compute_macro_micro,
                                process_model_type)
from ABCD_ML.Scoring import get_score


def evaluate_model(self, problem_type, model_type,
                   data_scaler='standard', n_splits=3, n_repeats=2,
                   int_cv=3, metric='default', class_weight='balanced',
                   random_state=None, extra_params={}):
    '''Class method to be called during the model selection phase.
    Used to evaluated different combination of models and scaling, ect...

    Parameters
    ----------
    problem_type : {'regression', 'binary', 'categorical'}
        Regression for float or ordinal target data, binary for binary,
        categorical for categorical.

    model_type : str or list of str,
        Each string refers to a type of model to train.
        If a list of strings is passed then an ensemble model
        will be created over all individual models.
        For a full list of supported options call:
        self.show_model_types(), with optional problem type parameter.

    data_scaler : str, optional (default = 'standard')
        `data_scaler` refers to the type of scaling to apply
        to the saved data during model evaluation.
        For a full list of supported options call:
        self.show_data_scalers()

    n_splits : int, optional (default = 3)
        evaluate_model performs a repeated k-fold model evaluation,
        `n_splits` refers to the k. E.g., if set to 3, then a 3-fold
        repeated CV will be performed. This parameter is typically
        chosen as a trade off between bias and variance, in addition to
        as a function of sample size.

    n_repeats : int, optional (default = 2)
        evaluate_model performs a repeated k-fold model evaluation,
        `n_repeats` refers to the number of times to repeat the
        k-fold CV. This parameter is typical chosen as a balance between
        run time, and accuratly accessing model performance.

    int_cv : int, optional (default = 3)
        The number of internal folds to use during
        model k-fold parameter selection, if the chosen model requires
        parameter selection. A value greater
        then 2 must be passed.

    metric : str, optional (default = 'default'),
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

    class weight : {dict, 'balanced', None}, optional (default = 'balanced')
        Only avaliable for binary and categorical problem types.
        Follows sklearn api class weight behavior. Typically, either use
        'balanced' in the case of class distribution imbalance, or None.

    random_state : int, RandomState instance or None, optional (default = None)
        Random state, either as int for a specific seed, or if None then
        the random seed is set by np.random.

    extra_params : dict, optional (default = {})
        Any extra params being passed. Typically, extra params are
        added when the user wants to provide a specific model/classifier,
        or data scaler, with updated (or new) parameters.
        These can be supplied by creating another dict within extra_params.
        E.g., extra_params[model_name] = {'model_param' : new_value}
        Where model param is a valid argument for that model, and model_name in
        this case is the str indicator passed to model_type.

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

    # Setup the desired splits, using the class defined train subjects
    subject_splits = self.CV.repeated_k_fold(subjects=self.train_subjects,
                                             n_repeats=n_repeats,
                                             n_splits=n_splits,
                                             random_state=random_state)

    scores = []

    # For each split with the repeated K-fold
    for train_subjects, test_subjects in subject_splits:

        score = self.test_model(problem_type=problem_type,
                                model_type=model_type,
                                train_subjects=train_subjects,
                                test_subjects=test_subjects,
                                data_scaler=data_scaler, int_cv=int_cv,
                                metric=metric, class_weight=class_weight,
                                random_state=random_state, return_model=False,
                                extra_params=extra_params)

        scores.append(score)

    # Return the computed macro and micro means and stds
    return compute_macro_micro(scores, n_repeats, n_splits)


def test_model(self, problem_type, model_type, train_subjects=None,
               test_subjects=None, data_scaler='standard', int_cv=3,
               metric='default', class_weight='balanced',
               random_state=None, return_model=False, extra_params={}):
    '''Class method used to evaluate a specific model / data scaling
    setup on an explicitly defined train and test set.

    Parameters
    ----------
    problem_type : {'regression', 'binary', 'categorical'}
        Regression for float or ordinal target data, binary for binary,
        categorical for categorical.

    model_type : str or list of str,
        Each string refers to a type of model to train.
        If a list of strings is passed then an ensemble model
        will be created over all individual models.
        For a full list of supported options call:
        self.show_model_types(), with optional problem type parameter.

    train_subjects : array-like or None, optional (default = None)
        If passed None, (default), then the class defined train subjects will
        be used. Otherwise, an array or pandas Index of
        valid subjects should be passed.

    test_subjects : array-like or None, optional (default = None)
        If passed None, (default), then the class defined test subjects will
        be used. Otherwise, an array or pandas Index of
        valid subjects should be passed.

    data_scaler : str, optional (default = 'standard')
        `data_scaler` refers to the type of scaling to apply
        to the saved data during model evaluation.
        For a full list of supported options call:
        self.show_data_scalers()

    int_cv : int, optional (default = 3)
        The number of internal folds to use during
        model k-fold parameter selection, if the chosen model requires
        parameter selection. A value greater
        then 2 must be passed.

    metric : str, optional (default = 'default'),
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

    class weight : {dict, 'balanced', None}, optional (default = 'balanced')
        Only avaliable for binary and categorical problem types.
        Follows sklearn api class weight behavior. Typically, either use
        'balanced' in the case of class distribution imbalance, or None.

    random_state : int, RandomState instance or None, optional (default = None)
        Random state, either as int for a specific seed, or if None then
        the random seed is set by np.random.

    return_model : bool, optional (default = False)
        If `return_model` is True, then model constructed and tested
        will be returned in addition to the score. If False,
        just the score will be returned.

    extra_params : dict, optional (default = {})
        Any extra params being passed. Typically, extra params are
        added when the user wants to provide a specific model/classifier,
        or data scaler, with updated (or new) parameters.
        These can be supplied by creating another dict within extra_params.
        E.g., extra_params[model_name] = {'model_param' : new_value}
        Where model param is a valid argument for that model, and model_name in
        this case is the str indicator passed to model_type.

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

    assert problem_type in ['binary', 'regression', 'categorical'], \
        "Invalid problem type!"

    default_metrics = {'regression': 'r2',
                       'binary': 'roc',
                       'categorical': 'weighted roc auc'}

    # Set default metric based on problem type, if no metric passed
    if metric == 'default':
        metric = default_metrics[problem_type]

    # Split the data to train/test
    train_data, test_data = self._split_data(train_subjects, test_subjects)

    # If passed a data scaler, scale and transform the data
    train_data, test_data = scale_data(train_data, test_data, data_scaler,
                                       self.data_keys, extra_params)

    # Process model_type, and get the cat_conv flag
    model_type, extra_params, cat_conv_flag = process_model_type(problem_type,
                                                                 model_type,
                                                                 extra_params)

    score_encoder = None
    if cat_conv_flag:
        score_encoder = self.score_encoder[1]

    # Create a trained model based on the provided parameters
    model = self._get_trained_model(problem_type=problem_type, data=train_data,
                                    model_type=model_type, int_cv=int_cv,
                                    metric=metric, class_weight=class_weight,
                                    random_state=random_state,
                                    score_encoder=score_encoder,
                                    extra_params=extra_params)

    # Compute the score of the trained model on the testing data
    score = get_score(problem_type, model, test_data, self.score_key, metric)

    if return_model:
        return score, model

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


def _split_data(self, train_subjects, test_subjects):
    '''Function to create train_data and test_data from self.all_data,
       based on input.

    Parameters
    ----------
    train_subjects : array-like or None, optional (default = None)
        If passed None, (default), then the class defined train subjects will
        be used. Otherwise, an array or pandas Index of
        valid subjects should be passed.

    test_subjects : array-like or None, optional (default = None)
        If passed None, (default), then the class defined test subjects will
        be used. Otherwise, an array or pandas Index of
        valid subjects should be passed.

    Returns
    ----------
    pandas DataFrame
        ABCD_ML formatted, with just the subset of training subjects

    pandas DataFrame
        ABCD_ML formatted, with just the subset of testing subjects
    '''

    if train_subjects is None:
        train_subjects = self.train_subjects
    if test_subjects is None:
        test_subjects = self.test_subjects

    train_data = self.all_data.loc[train_subjects]
    test_data = self.all_data.loc[test_subjects]

    return train_data, test_data


def _get_trained_model(self, problem_type, data, model_type,
                       int_cv, metric, class_weight='balanced',
                       random_state=None, score_encoder=None,
                       extra_params={}):
    '''
    Helper function for training either an ensemble or one single model.
    Handles model type additionally.

    Parameters
    ----------
    problem_type : {'regression', 'binary', 'categorical'}
        Regression for float or ordinal target data, binary for binary,
        categorical for categorical.

    data : pandas DataFrame,
        ABCD_ML formatted df.

    model_type : str or list of str,
        Each string refers to a type of model to train.
        If a list of strings is passed then an ensemble model
        will be created over all individual models.
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

    class weight : {dict, 'balanced', None}, optional (default = 'balanced')
        Only avaliable for binary and categorical problem types.
        Follows sklearn api class weight behavior. Typically, either use
        'balanced' in the case of class distribution imbalance, or None.

    random_state : int, RandomState instance or None, optional (default = None)
        Random state, either as int for a specific seed, or if None then
        the random seed is set by np.random.

    score_encoder : sklearn encoder, optional (default=None)
        A sklearn api encoder, for optionally transforming the target
        variable. Used in the case of categorical data in converting from
        one-hot encoding to ordinal.

    extra_params : dict, optional (default = {})
        Any extra params being passed. Typically, extra params are
        added when the user wants to provide a specific model/classifier,
        or data scaler, with updated (or new) parameters.
        These can be supplied by creating another dict within extra_params.
        E.g., extra_params[model_name] = {'model_param' : new_value}
        Where model param is a valid argument for that model, and model_name in
        this case is the str indicator passed to model_type.

    Returns
    -------
    model : returns a trained model object.
    '''
    model_params = {'data': data,
                    'score_key': self.score_key,
                    'CV': self.CV,
                    'model_type': model_type,
                    'int_cv': int_cv,
                    'metric': metric,
                    'class_weight': class_weight,
                    'random_state': random_state,
                    'score_encoder': score_encoder,
                    'n_jobs': self.n_jobs,
                    'extra_params': extra_params}

    if type(model_type) == list:
        model = Ensemble_Model(**model_params)
    else:
        model = train_model(**model_params)

    return model
