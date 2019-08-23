"""
_ML.py
====================================
Main class extension file for the Machine Learning functionality
"""
import numpy as np
import pandas as pd
from ABCD_ML.ML_Helpers import compute_macro_micro
from ABCD_ML.Model import Regression_Model, Binary_Model, Categorical_Model


def Set_Default_ML_Params(self, problem_type='default', metric='default',
                          data_scaler='default', sampler='default',
                          feat_selector='default', n_splits='default',
                          n_repeats='default', int_cv='default',
                          search_type='default',
                          data_scaler_param_ind='default',
                          sampler_param_ind='default',
                          feat_selector_param_ind='default',
                          class_weight='default', n_jobs='default',
                          n_iter='default', data_to_use='default',
                          compute_train_score='default',
                          random_state='default',
                          calc_base_feature_importances='default',
                          calc_shap_feature_importances='default',
                          extra_params='default'):
    '''Sets the self.default_ML_params dictionary with user passed or default
    values. In general, if any argument is left as 'default' and it has
    not been previously defined, it will be set to a default value,
    sometimes passed on other values. See notes for rationale behind
    default ML params.

    Parameters
    ----------
    problem_type : {'regression', 'binary', 'categorical', 'default'}, optional

        - 'regression' : For ML on float or ordinal target data
        - 'binary' : For ML on binary target data
        - 'categorical' : For ML on categorical target data,\
                          as either multilabel or multiclass.
        - 'default' : Use 'regression' if nothing else already defined

        If 'default', and not already defined, set to 'regression'
        (default = 'default')

    metric : str or list, optional
        Indicator for which metric(s) to use for calculating
        score and during model parameter selection.
        If `metric` left as 'default', then the default metric/scorer
        for that problem types will be used.
        Note, some metrics are only avaliable for certain problem types.
        For a full list of supported metrics call:
        self.Show_Metrics, with optional problem type parameter.

        For a full list of supported metrics call:
        :func:`Show_Metrics`

         If 'default', and not already defined, set to default
        metric for the problem type.

        - 'regression'  : 'r2',
        - 'binary'      : 'roc',
        - 'categorical' : 'weighted roc auc'

        (default = 'default')

    data_scaler : str, list or None optional
        `data_scaler` refers to the type of scaling to apply
        to the saved data (just data, not covars) during model evaluation.
        If a list is passed, then scalers will be applied in that order.
        If None, then no scaling will be applied.

        For a full list of supported options call:
        :func:`Show_Data_Scalers`

        If 'default', and not already defined, set to 'standard'
        (default = 'default')

    sampler : str, list or none optional
        `sampler` refers optional to the type of resampling
        to apply within the model pipeline - to correct for
        imbalanced class distributions. These are different
        techniques for over sampling under distributed classed
        and under sampling over distributed ones.
        If a list is passed, then samplers will be fit and applied
        in that order.

        For a full list of supported options call:
        :func:`Show_Samplers`

        If 'default', and not already defined, set to None
        (default = 'default')

    feat_selector : str, list or None, optional
        `feat_selector` should be a str indicator or list of,
        for which feature selection to use, if a list, they will
        be applied in order.
        If None, then no feature selection will be used.

        For a full list of supported options call:
        :func:`Show_Feat_Selectors`

        If 'default', and not already defined, set to None
        (default = 'default')

    n_splits : int or 'default', optional
        evaluate_model performs a repeated k-fold model evaluation,
        `n_splits` refers to the k. E.g., if set to 3, then a 3-fold
        repeated CV will be performed. This parameter is typically
        chosen as a trade off between bias and variance, in addition to
        as a function of sample size.

        If 'default', and not already defined, set to 3
        (default = 'default')

    n_repeats : int or 'default', optional
        evaluate_model performs a repeated k-fold model evaluation,
        `n_repeats` refers to the number of times to repeat the
        k-fold CV. This parameter is typical chosen as a balance between
        run time, and accuratly accessing model performance.

        If 'default', and not already defined, set to 2
        (default = 2)

    int_cv : int or 'default', optional
        The number of internal folds to use during
        model k-fold parameter selection, if the chosen model requires
        parameter selection. A value greater
        then 2 must be passed.

        If 'default', and not already defined, set to 3
        (default = 'default')

    search_type : {'random', 'grid', None, 'default'}
        The type of parameter search to conduct if any.

        - 'random' : Uses sklearn RandomizedSearchCV
        - 'grid' : Uses sklearn GridSearchCV
        - None : No search

        .. WARNING::
            If search type is set to grid, and any of model_type_param_ind,
            data_scaler_param_ind and feat_selector_param_ind are set
            to a random distribution (rather then discrete values),
            this will lead to an error.

        If 'default', and not already defined, set to None
        (default = 'default')

    data_scaler_param_ind : int, str, or list of
        Each `data_scaler` has atleast one default parameter distribution
        saved with it. This parameter is used to select between different
        distributions to be used with `search_type` == 'random' or 'grid',
        when `search_type` == None, `data_scaler_param_ind` is automatically
        set to default 0.
        This parameter can be selected with either an integer index
        (zero based), or the str name for a given `data_scaler`.
        Likewise with `data_scaler`, if passed list input, this means
        a list was passed to `data_scaler` and the indices should correspond.

        The different parameter distributions avaliable for each
        `data_scaler`, can be shown by calling :func:`Show_Data_Scalers`

        If 'default', and not already defined, set to 0
        (default = 'default')

    sampler_param_ind :  int, str, or list of
        Each `sampler` has atleast one default parameter distribution
        saved with it. This parameter is used to select between different
        distributions to be used with `search_type` == 'random' or 'grid',
        when `search_type` == None, `sampler_param_ind` is automatically
        set to default 0.
        This parameter can be selected with either an integer index
        (zero based), or the str name for a given `sampler`.
        Likewise with `sampler`, if passed list input, this means
        a list was passed to `sampler` and the indices should correspond.

        The different parameter distributions avaliable for each
        `sampler`, can be shown by calling :func:`Show_Samplers`

        If 'default', and not already defined, set to 0
        (default = 'default')

    feat_selector_param_ind : int, str, or list of
         Each `feat_selector` has atleast one default parameter distribution
        saved with it. This parameter is used to select between different
        distributions to be used with `search_type` == 'random' or 'grid',
        when `search_type` == None, `feat_selector_param_ind` is automatically
        set to default 0.
        This parameter can be selected with either an integer index
        (zero based), or the str name for a given `feat_selector` param option.
        Likewise with `feat_selector`, if passed list input, this means
        a list was passed to `feat_selector` and the indices should correspond.

        The different parameter distributions avaliable for each
        `feat_selector`, can be shown by calling :func:`Show_Feat_Selectors`

        If 'default', and not already defined, set to 0
        (default = 'default')

    class_weight : {dict, 'balanced', None, 'default'}, optional
        Only used for binary and categorical problem types.
        Follows sklearn api class weight behavior. Typically, either use
        'balanced' in the case of class distribution imbalance, or None.

        If 'default', and not already defined, set to None
        (default = 'default')

    n_jobs : int or 'default', optional
        The number of jobs to use (if avaliable) during training ML models.
        This should be the number of procesors avaliable, if wanting to run
        as fast as possible.

        if 'default', and not already defined, set to 1.
        (default = 'default')

    n_iter : int or 'default', optional
        The number of random search parameters to try, used
        only if using random search.

        if 'default', and not already defined, set to 10.
        (default = 'default')

    data_to_use : {'all', 'data', 'covars'}, optional
        This setting allows the user to optionally
        run an expiriment with either only the loaded
        data and/or only the loaded covars. Likewise,
        both can be used with the default param of 'all'.

        - 'all' : Uses all data + covars loaded
        - 'data' : Uses only the loaded data, and drops covars if any
        - 'covars' : Uses only the loaded covars, and drops data if any

        if 'default', and not already defined, set to 'all'.
        (default = 'default')

    compute_train_score : bool, optional
        If set to True, then :func:`Evaluate` and :func:`Test`
        will compute, and print, training scores along with
        validation / test scores.

        if 'default', and not already defined, set to False.
        (default = 'default')

    random_state : int, RandomState instance, None or 'default', optional
        Random state, either as int for a specific seed, or if None then
        the random seed is set by np.random.

        If 'default', use the saved value within self,
        (defined when initing ABCD_ML class) ^,
        Or can define a different random state for use in ML.
        (default = 'default')

    calc_base_feature_importances : bool or 'default, optional
        If set to True, will store the base feature importances
        when running Evaluate or Test. Note, base feature importances
        are only avaliable for tree-based or linear models, specifically
        those with either coefs or feature_importance attributes.

        If 'default', and not already defined, set to True.
        (default = 'default')

    calc_shap_feature_importances : bool or 'default, optional
        If set to True, will calculate SHAP (SHapley Additive exPlanations)
        for the model when running Evaluate or Test.
        Note: For any case where the underlytin model is not tree or linear
        based, e.g. an ensemble of different methods, or non-linear svm,
        these values are estimated by a kernel explainer function which is
        very compute intensive.
        Read more about shap on their github:

        https://github.com/slundberg/shap

        If 'default', and not already defined, set to False.
        (default = 'default')

    extra_params : dict or 'default', optional
        Any extra params being passed. Typically, extra params are
        added when the user wants to provide a specific model/classifier,
        or data scaler, with updated (or new) parameters.
        These can be supplied by creating another dict within extra_params.
        E.g., ::

            extra_params[model_name] = {'model_param' : new_value}

        Where model param is a valid argument for that model, and model_name in
        this case is the str indicator passed to model_type.
        If 'default', and not already defined, set to empty dict.

        (default = 'default')
    '''

    default_metrics = {'binary': 'macro roc auc', 'regression': 'r2',
                       'categorical': 'weighted roc auc'}

    if problem_type != 'default':
        problem_type = problem_type.lower()
        assert problem_type in default_metrics, 'Invalid problem type passed!'
        self.default_ML_params['problem_type'] = problem_type

    elif 'problem_type' not in self.default_ML_params:
        self.default_ML_params['problem_type'] = 'regression'
        self._print('No default problem type passed, set to regression.')

    if metric != 'default':
        self.default_ML_params['metric'] = metric

    elif 'metric' not in self.default_ML_params:

        self.default_ML_params['metric'] = \
            default_metrics[self.default_ML_params['problem_type']]

        self._print('No default metric passed, set to,',
                    self.default_ML_params['metric'],
                    'based on default problem type.')

    if data_scaler != 'default':
        self.default_ML_params['data_scaler'] = data_scaler

    elif 'data_scaler' not in self.default_ML_params:
        self.default_ML_params['data_scaler'] = 'standard'
        self._print('No default data scaler passed, set to standard')

    if sampler != 'default':
        self.default_ML_params['sampler'] = sampler

    elif 'sampler' not in self.default_ML_params:
        self.default_ML_params['sampler'] = None
        self._print('No default sampler passed, set to None')

    if feat_selector != 'default':
        self.default_ML_params['feat_selector'] = feat_selector

    elif 'feat_selector' not in self.default_ML_params:
        self.default_ML_params['feat_selector'] = None
        self._print('No default feat selector passed, set to None')

    if n_splits != 'default':
        assert isinstance(n_splits, int), 'n_splits must be int'
        assert n_splits > 1, 'n_splits must be greater than 1'
        self.default_ML_params['n_splits'] = n_splits

    elif 'n_splits' not in self.default_ML_params:
        self.default_ML_params['n_splits'] = 3
        self._print('No default num CV splits passed, set to 3')

    if n_repeats != 'default':
        assert isinstance(n_repeats, int), 'n_repeats must be int'
        assert n_repeats > 0, 'n_repeats must be greater than 0'
        self.default_ML_params['n_repeats'] = n_repeats

    elif 'n_repeats' not in self.default_ML_params:
        self.default_ML_params['n_repeats'] = 2
        self._print('No default num CV repeats passed, set to 2')

    if int_cv != 'default':
        assert isinstance(int_cv, int), 'int_cv must be int'
        assert int_cv > 1, 'int_cv must be greater than 1'
        self.default_ML_params['int_cv'] = int_cv

    elif 'int_cv' not in self.default_ML_params:
        self.default_ML_params['int_cv'] = 3
        self._print('No default num internal CV splits passed, set to 3')

    if search_type != 'default':
        self.default_ML_params['search_type'] = search_type

    elif 'search_type' not in self.default_ML_params:
        self.default_ML_params['search_type'] = None
        self._print('No default search type passed, set to None')

    if data_scaler_param_ind != 'default':
        self.default_ML_params['data_scaler_param_ind'] = data_scaler_param_ind

    elif 'data_scaler_param_ind' not in self.default_ML_params:
        self.default_ML_params['data_scaler_param_ind'] = 0
        self._print('No default data scaler param ind passed, set to 0')

    if sampler_param_ind != 'default':
        self.default_ML_params['sampler_param_ind'] = sampler_param_ind

    elif 'sampler_param_ind' not in self.default_ML_params:
        self.default_ML_params['sampler_param_ind'] = 0
        self._print('No default sampler param ind passed, set to 0')

    if feat_selector_param_ind != 'default':
        self.default_ML_params['feat_selector_param_ind'] =\
            feat_selector_param_ind

    elif 'feat_selector_param_ind' not in self.default_ML_params:
        self.default_ML_params['feat_selector_param_ind'] = 0
        self._print('No default feat selector param ind passed, set to 0')

    if class_weight != 'default':
        self.default_ML_params['class_weight'] = class_weight

    elif 'class_weight' not in self.default_ML_params:
        self.default_ML_params['class_weight'] = None
        if self.default_ML_params['problem_type'] != 'regression':
            self._print('No default class weight setting passed,',
                        'set to None')

    if n_jobs != 'default':
        assert isinstance(n_jobs, int), 'n_jobs must be int'
        self.default_ML_params['n_jobs'] = n_jobs

    elif 'n_jobs' not in self.default_ML_params:
        self.default_ML_params['n_jobs'] = 1
        self._print('No default number of jobs passed, set to 1')

    if n_iter != 'default':
        assert isinstance(n_iter, int), 'n_iter must be int'
        assert n_iter > 0, 'n_iter must be greater than 0'
        self.default_ML_params['n_iter'] = n_iter

    elif 'n_iter' not in self.default_ML_params:
        self.default_ML_params['n_iter'] = 10
        self._print('No default number of random search iters passed,',
                    'set to 10')

    if data_to_use != 'default':
        assert data_to_use in ['all', 'data', 'covars'], \
            "data_to_use must be 'all', 'data' or 'covars'"
        self.default_ML_params['data_to_use'] = data_to_use

    elif 'data_to_use' not in self.default_ML_params:
        self.default_ML_params['data_to_use'] = 'all'
        self._print('No default data_to_use passed,',
                    'set to all')

    if compute_train_score != 'default':
        self.default_ML_params['compute_train_score'] = compute_train_score

    elif 'compute_train_score' not in self.default_ML_params:
        self.default_ML_params['compute_train_score'] = False
        self._print('No default compute_train_score passed,',
                    'set to False')

    if random_state != 'default':
        self.default_ML_params['random_state'] = random_state

    elif 'random_state' not in self.default_ML_params:
        self.default_ML_params['random_state'] = self.random_state
        self._print('No default random state passed, using class random',
                    'state value of', self.random_state)

    if calc_base_feature_importances != 'default':
        self.default_ML_params['calc_base_feature_importances'] =\
            calc_base_feature_importances

    elif 'calc_base_feature_importances' not in self.default_ML_params:
        self.default_ML_params['calc_base_feature_importances'] = True

        self._print('No default calc_base_feature_importances passed,',
                    'set to',
                    self.default_ML_params['calc_base_feature_importances'])

    if calc_shap_feature_importances != 'default':
        self.default_ML_params['calc_shap_feature_importances'] =\
            calc_shap_feature_importances

    elif 'calc_shap_feature_importances' not in self.default_ML_params:
        self.default_ML_params['calc_shap_feature_importances'] = False

        self._print('No default calc_shap_feature_importances passed,',
                    'set to',
                    self.default_ML_params['calc_shap_feature_importances'])

    if extra_params != 'default':
        assert isinstance(extra_params, dict), 'extra params must be dict'
        self.default_ML_params['extra_params'] = extra_params

    elif 'extra_params' not in self.default_ML_params:
        self.default_ML_params['extra_params'] = {}
        self._print('No default extra params passed, set to empty dict')

    self._print('Default params set.')
    self._print()


def Evaluate(self, model_type, problem_type='default', metric='default',
             data_scaler='default', sampler='default', feat_selector='default',
             n_splits='default', n_repeats='default', int_cv='default',
             ensemble_type='basic ensemble', ensemble_split=.2,
             search_type='default', model_type_param_ind=0,
             data_scaler_param_ind='default', sampler_param_ind='default',
             feat_selector_param_ind='default', class_weight='default',
             n_jobs='default', n_iter='default', data_to_use='default',
             compute_train_score='default', random_state='default',
             calc_base_feature_importances='default',
             calc_shap_feature_importances='default', extra_params='default'):

    '''Class method to be called during the model selection phase.
    Used to evaluated different combination of models and scaling, ect...

    Parameters
    ----------
    model_type : str or list of str
        Each string refers to a type of model to train.
        If a list of strings is passed then an ensemble model
        will be created over all individual models.

        For a full list of supported options call:
        :func:`Show_Model_Types`

    problem_type : {'regression', 'binary', 'categorical', 'default'}, optional

        - 'regression' : For ML on float or ordinal target data
        - 'binary' : For ML on binary target data
        - 'categorical' : For ML on categorical target data, \
                          as either multilabel or multiclass.
        - 'default' : Use the name problem type within self.default_ML_params.

        (default = 'default')

    metric : str or list, optional
        Indicator for which metric(s) to use for calculating
        score and during model parameter selection.
        Note, some metrics are only avaliable for certain problem types.

        For a full list of supported metrics call:
        :func:`Show_Metrics`

        If 'default', use the saved value within self.default_ML_params.
        (default = 'default')

    data_scaler : str, list or None, optional
        `data_scaler` refers to the type of scaling to apply
        to the saved data (just data, not covars) during model evaluation.
        If a list is passed, then scalers will be applied in that order.
        If None, then no scaling will be applied.

        For a full list of supported options call:
        :func:`Show_Data_Scalers`

        If 'default', use the saved value within self.default_ML_params.
        (default = 'default')

    sampler : str, list or none optional
        `sampler` refers optional to the type of resampling
        to apply within the model pipeline - to correct for
        imbalanced class distributions. These are different
        techniques for over sampling under distributed classed
        and under sampling over distributed ones.
        If a list is passed, then samplers will be fit and applied
        in that order.

        For a full list of supported options call:
        :func:`Show_Samplers`

        If 'default', use the saved value within self.default_ML_params.
        (default = 'default')

    feat_selector : str, list or None, optional
        `feat_selector` should be a str indicator or list of,
        for which feature selection to use, if a list, they will
        be applied in order.
        If None, then no feature selection will be used.

        For a full list of supported options call:
        :func:`Show_Feat_Selectors`

        If 'default', use the saved value within self.default_ML_params.
        (default = 'default')

    n_splits : int or 'default', optional
        ``Evaluate`` performs a repeated k-fold model evaluation,
        `n_splits` refers to the k. E.g., if set to 3, then a 3-fold
        repeated CV will be performed. This parameter is typically
        chosen as a trade off between bias and variance, in addition to
        as a function of sample size.

        If 'default', use the saved value within self.default_ML_params.
        (default = 'default')

    n_repeats : int or 'default', optional
        ``Evaluate`` performs a repeated k-fold model evaluation,
        `n_repeats` refers to the number of times to repeat the
        k-fold CV. This parameter is typical chosen as a balance between
        run time, and accuratly accessing model performance.

        If 'default', use the saved value within self.default_ML_params.
        (default = 2)

    int_cv : int or 'default', optional
        The number of internal folds to use during
        model k-fold parameter selection, if the chosen model requires
        parameter selection. A value greater
        then 2 must be passed.

        If 'default', use the saved value within self.default_ML_params.
        (default = 'default')

    ensemble_type : str or list of str,
        Each string refers to a type of ensemble to train,
        or 'basic ensemble' (default) for base behavior.
        Base ensemble behavior is either to not ensemble,
        if only one model type is passed,
        or when multiple model types are passed,
        to simply train each one independently and
        average the predictions at test time (or max vote).

        The user can optionally pass other ensemble types,
        anything but 'basic ensemble' will require an
        ensemble split though, which is an additional
        train/val split on the training set, where the
        val/ensemble split, is used to fit the ensemble object.

        If a list is passed to ensemble_type, then every
        item in the list must be a valid str indicator for
        a non 'basic ensemble' ensemble type, and each ensemble
        object passed will be fitted independly and then averaged
        using the 'basic ensemble' behvaior... so an ensemble of ensembles.

        For a full list of supported options call:
        :func:`Show_Ensemble_Types`

    ensemble_split : float, int or None
        If a an ensemble_type(s) that requires fitting is passed,
        i.e., not "basic ensemble", then this param is
        the porportion of the train_data within each fold to
        use towards fitting the ensemble objects.
        If multiple ensembles are passed, they are all
        fit with the same fold of data.

    search_type : {'random', 'grid', None, 'default'}
        The type of parameter search to conduct if any.

        - 'random' : Uses sklearn RandomizedSearchCV
        - 'grid' : Uses sklearn GridSearchCV
        - None : No search

        .. WARNING::
            If search type is set to grid, and any of model_type_param_ind,
            data_scaler_param_ind and feat_selector_param_ind are set
            to a random distribution (rather then discrete values),
            this will lead to an error.

        If 'default', use the saved value within self.default_ML_params.
        (default = 'default')

    model_type_param_ind : int, str, or list of
        Each `model_type` has atleast one default parameter distribution
        saved with it. This parameter is used to select between different
        distributions to be used with `search_type` == 'random' or 'grid',
        when `search_type` == None, `model_type_param_ind` is automatically
        set to default 0.
        This parameter can be selected with either an integer index
        (zero based), or the str name for a given `model_type`.
        Likewise with `model_type`, if passed list input, this means
        a list was passed to `model_type` and the indices should correspond.

        The different parameter distributions avaliable for each
        `model_type`, can be shown by calling :func:`Show_Model_Types`

        (default = 0)

    data_scaler_param_ind : int, str, or list of
        Each `data_scaler` has atleast one default parameter distribution
        saved with it. This parameter is used to select between different
        distributions to be used with `search_type` == 'random' or 'grid',
        when `search_type` == None, `data_scaler_param_ind` is automatically
        set to default 0.
        This parameter can be selected with either an integer index
        (zero based), or the str name for a given `data_scaler`.
        Likewise with `data_scaler`, if passed list input, this means
        a list was passed to `data_scaler` and the indices should correspond.

        The different parameter distributions avaliable for each
        `data_scaler`, can be shown by calling :func:`Show_Data_Scalers`

        If 'default', use the saved value within self.default_ML_params.
        (default = 'default')

    sampler_param_ind :  int, str, or list of
        Each `sampler` has atleast one default parameter distribution
        saved with it. This parameter is used to select between different
        distributions to be used with `search_type` == 'random' or 'grid',
        when `search_type` == None, `sampler_param_ind` is automatically
        set to default 0.
        This parameter can be selected with either an integer index
        (zero based), or the str name for a given `sampler`.
        Likewise with `sampler`, if passed list input, this means
        a list was passed to `sampler` and the indices should correspond.

        The different parameter distributions avaliable for each
        `sampler`, can be shown by calling :func:`Show_Samplers`

        If 'default', use the saved value within self.default_ML_params.
        (default = 'default')

    feat_selector_param_ind : int, str, or list of
         Each `feat_selector` has atleast one default parameter distribution
        saved with it. This parameter is used to select between different
        distributions to be used with `search_type` == 'random' or 'grid',
        when `search_type` == None, `feat_selector_param_ind` is automatically
        set to default 0.
        This parameter can be selected with either an integer index
        (zero based), or the str name for a given `feat_selector` param option.
        Likewise with `feat_selector`, if passed list input, this means
        a list was passed to `feat_selector` and the indices should correspond.

        The different parameter distributions avaliable for each
        `feat_selector`, can be shown by calling :func:`Show_Feat_Selectors`

        If 'default', use the saved value within self.default_ML_params.
        (default = 'default')

    class_weight : {dict, 'balanced', None, 'default'}, optional
        Only used for binary and categorical problem types.
        Follows sklearn api class weight behavior. Typically, either use
        'balanced' in the case of class distribution imbalance, or None.
        If 'default', use the saved value within self.default_ML_params.

        (default = 'default')

    n_jobs : int or 'default', optional
        The number of jobs to use (if avaliable) during training ML models.
        This should be the number of procesors avaliable, if wanting to run
        as fast as possible.
        If 'default', use the saved value within self.default_ML_params.

        (default = 'default')

    n_iter : int or 'default', optional
        The number of random search parameters to try, used
        only if using random search.
        If 'default', use the saved value within self.default_ML_params.

        (default = 'default')

    data_to_use : {'all', 'data', 'covars'}, optional
        This setting allows the user to optionally
        run an expiriment with either only the loaded
        data and/or only the loaded covars. Likewise,
        both can be used with the default param of 'all'.

        - 'all' : Uses all data + covars loaded
        - 'data' : Uses only the loaded data, and drops covars if any
        - 'covars' : Uses only the loaded covars, and drops data if any

        If 'default', use the saved value within self.default_ML_params.

        (default = 'default')

    compute_train_score : bool, optional
        If set to True, will compute and print the model pipelines
        training score in addition to validation scores.
        If 'default', use the saved value within self.default_ML_params.

        (default = 'default')

    random_state : int, RandomState instance, None or 'default', optional
        Random state, either as int for a specific seed, or if None then
        the random seed is set by np.random.
        If 'default', use the saved value within self,
        (defined when initing ABCD_ML class).
        Or a different ML params random state is used, if defined when
        calling set default ML params.

        (default = 'default')

    calc_base_feature_importances : bool or 'default, optional
        If set to True, will store the base feature importances
        when running Evaluate or Test. Note, base feature importances
        are only avaliable for tree-based or linear models, specifically
        those with either coefs or feature_importance attributes.

        If 'default', use the saved value within self.default_ML_params.
        (default = 'default')

    calc_shap_feature_importances : bool or 'default, optional
        If set to True, will calculate SHAP (SHapley Additive exPlanations)
        for the model when running Evaluate or Test.
        Note: For any case where the underlytin model is not tree or linear
        based, e.g. an ensemble of different methods, or non-linear svm,
        these values are estimated by a kernel explainer function which is
        very compute intensive.
        Read more about shap on their github:

        https://github.com/slundberg/shap

        If 'default', use the saved value within self.default_ML_params.
        (default = 'default')

    extra_params : dict or 'default', optional
        Any extra params being passed. Typically, extra params are
        added when the user wants to provide a specific model/classifier,
        or data scaler, with updated (or new) parameters.
        These can be supplied by creating another dict within extra_params.

        E.g., ::

            extra_params[model_name] = {'model_param' : new_value}

        Where model param is a valid argument for that model, and model_name in
        this case is the str indicator passed to model_type.
        If 'default', use the saved value within self.default_ML_params.

        (default = 'default')

    Returns
    ----------
    train_scores : array-like of array-like
        numpy array of numpy arrays,
        where each internal array contains the raw scores as computed for
        all passed in metrics, computed for each fold within
        each repeat.
        e.g., array will have a length of `n_repeats` * `n_splits`,
        and each internal array will have the same length as the number of
        metrics.

    validation_scores : array-like of array-like
        numpy array of numpy arrays,
        where each internal array contains the raw scores as computed for
        all passed in metrics, computed for each fold within
        each repeat.
        e.g., array will have a length of `n_repeats` * `n_splits`,
        and each internal array will have the same length as the number of
        metrics.

    Notes
    ----------
    Prints by default the following for each metric,

    float
        The mean macro score (as set by input metric) across each
        repeated K-fold.

    float
        The standard deviation of the macro score (as set by input metric)
        across each repeated K-fold.

    float
        The standard deviation of the micro score (as set by input metric)
        across each fold with the repeated K-fold.

    '''

    # Perform pre-modeling check
    self._premodel_check(problem_type)

    # Create the set of ML_params from passed args + default args
    ML_params = self._make_ML_params(args=locals())

    # Print the params being used
    self._print_model_params(model_type, ML_params, ensemble_type,
                             ensemble_split, model_type_param_ind,
                             test=False)

    # Init the Model object with modeling params
    self._init_model(model_type, ML_params, ensemble_type, ensemble_split,
                     model_type_param_ind)

    # Evaluate the model
    train_scores, scores =\
        self.Model.Evaluate_Model(self.all_data, self.train_subjects)

    # Print out summary stats for all passed metrics
    scorer_strs = self.Model.scorer_strs
    self._print()

    if ML_params['compute_train_score']:
        score_list = [train_scores, scores]
        score_type_list = ['Training', 'Validation']
    else:
        score_list = [scores]
        score_type_list = ['Validation']

    for s, name in zip(score_list, score_type_list):

        self._print(name + ' Scores')
        self._print(''.join('_' for i in range(len(name) + 7)))

        for i in range(len(scorer_strs)):
            self._print('Metric: ', scorer_strs[i])

            score_by_metric = s[:, i]

            if len(score_by_metric[0].shape) > 0:
                by_class = [[score_by_metric[i][j] for i in
                            range(len(score_by_metric))] for j in
                            range(len(score_by_metric[0]))]

                summary_scores_by_class =\
                    [compute_macro_micro(class_scores, ML_params['n_repeats'],
                     ML_params['n_splits']) for class_scores in by_class]

                for summary_scores, class_name in zip(summary_scores_by_class,
                                                      self.targets_key):
                    self._print('for target class: ', class_name)
                    self._print_summary_score(name, summary_scores,
                                              ML_params['n_repeats'])

            else:

                # Compute macro / micro summary of scores
                summary_scores = compute_macro_micro(score_by_metric,
                                                     ML_params['n_repeats'],
                                                     ML_params['n_splits'])

                self._print_summary_score(name, summary_scores,
                                          ML_params['n_repeats'])

    # Return the raw scores from each fold
    return score_list


def Test(self, model_type, problem_type='default', train_subjects=None,
         test_subjects=None, metric='default', data_scaler='default',
         sampler='default', feat_selector='default', int_cv='default',
         ensemble_type='basic ensemble', ensemble_split=.2,
         search_type='default', model_type_param_ind=0,
         data_scaler_param_ind='default', sampler_param_ind='default',
         feat_selector_param_ind='default', class_weight='default',
         n_jobs='default', n_iter='default', data_to_use='default',
         compute_train_score='default', random_state='default',
         return_model=False, calc_base_feature_importances='default',
         calc_shap_feature_importances='default', extra_params='default'):
    '''Class method used to evaluate a specific model / data scaling
    setup on an explicitly defined train and test set.

    Parameters
    ----------
    model_type : str or list of str
        Each string refers to a type of model to train.
        If a list of strings is passed then an ensemble model
        will be created over all individual models.

        For a full list of supported options call:
        :func:`Show_Model_Types`

    problem_type : {'regression', 'binary', 'categorical', 'default'}, optional

        - 'regression' : For ML on float or ordinal target data
        - 'binary' : For ML on binary target data
        - 'categorical' : For ML on categorical target data, \
                          as either multilabel or multiclass.
        - 'default' : Use the name problem type within self.default_ML_params.

        (default = 'default')

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

    metric : str or list, optional
        Indicator for which metric(s) to use for calculating
        score and during model parameter selection.
        If 'default', use the saved value within self.default_ML_params.
        Note, some metrics are only avaliable for certain problem types.

        - 'regression'  : 'r2',
        - 'binary'      : 'roc',
        - 'categorical' : 'weighted roc auc'

        For a full list of supported metrics call:
        :func:`Show_Metrics`

        (default = 'default')

    data_scaler : str, list or None optional
        `data_scaler` refers to the type of scaling to apply
        to the saved data (just data, not covars) during model evaluation.
        If a list is passed, then scalers will be applied in that order.
        If None, then no scaling will be applied.
        If 'default', use the saved value within self.default_ML_params.

        For a full list of supported options call:
        :func:`Show_Data_Scalers`

        (default = 'default')

    sampler : str, list or none optional
        `sampler` refers optional to the type of resampling
        to apply within the model pipeline - to correct for
        imbalanced class distributions. These are different
        techniques for over sampling under distributed classed
        and under sampling over distributed ones.
        If a list is passed, then samplers will be fit and applied
        in that order.

        For a full list of supported options call:
        :func:`Show_Samplers`

        If 'default', use the saved value within self.default_ML_params.
        (default = 'default')

    feat_selector : str, list or None, optional
        `feat_selector` should be a str indicator or list of,
        for which feature selection to use, if a list, they will
        be applied in order.
        If None, then no feature selection will be used.
        If 'default', use the saved value within self.default_ML_params.

        For a full list of supported options call:
        :func:`Show_Feat_Selectors`

        (default = 'default')

    int_cv : int or 'default', optional
        The number of internal folds to use during
        model k-fold parameter selection, if the chosen model requires
        parameter selection. A value greater
        then 2 must be passed.
        If 'default', use the saved value within self.default_ML_params.

        (default = 'default')

    ensemble_type : str or list of str,
        Each string refers to a type of ensemble to train,
        or 'basic ensemble' (default) for base behavior.
        Base ensemble behavior is either to not ensemble,
        if only one model type is passed,
        or when multiple model types are passed,
        to simply train each one independently and
        average the predictions at test time (or max vote).

        The user can optionally pass other ensemble types,
        anything but 'basic ensemble' will require an
        ensemble split though, which is an additional
        train/val split on the training set, where the
        val/ensemble split, is used to fit the ensemble object.

        If a list is passed to ensemble_type, then every
        item in the list must be a valid str indicator for
        a non 'basic ensemble' ensemble type, and each ensemble
        object passed will be fitted independly and then averaged
        using the 'basic ensemble' behvaior... so an ensemble of ensembles.

        For a full list of supported options call:
        :func:`Show_Ensemble_Types`

    ensemble_split : float, int or None
        If a an ensemble_type(s) that requires fitting is passed,
        i.e., not "basic ensemble", then this param is
        the porportion of the train_data within each fold to
        use towards fitting the ensemble objects.
        If multiple ensembles are passed, they are all
        fit with the same fold of data.

    search_type : {'random', 'grid', None, 'default'}
        The type of parameter search to conduct if any.

        - 'random' : Uses sklearn RandomizedSearchCV
        - 'grid' : Uses sklearn GridSearchCV
        - None : No search

        .. WARNING::
            If search type is set to grid, and any of model_type_param_ind,
            data_scaler_param_ind and feat_selector_param_ind are set
            to a random distribution (rather then discrete values),
            this will lead to an error.

        If 'default', use the saved value within self.default_ML_params.
        (default = 'default')

    model_type_param_ind : int, str, or list of
        Each `model_type` has atleast one default parameter distribution
        saved with it. This parameter is used to select between different
        distributions to be used with `search_type` == 'random' or 'grid',
        when `search_type` == None, `model_type_param_ind` is automatically
        set to default 0.
        This parameter can be selected with either an integer index
        (zero based), or the str name for a given `model_type`.
        Likewise with `model_type`, if passed list input, this means
        a list was passed to `model_type` and the indices should correspond.

        The different parameter distributions avaliable for each
        `model_type`, can be shown by calling :func:`Show_Model_Types`

        (default = 0)

    data_scaler_param_ind : int, str, or list of
        Each `data_scaler` has atleast one default parameter distribution
        saved with it. This parameter is used to select between different
        distributions to be used with `search_type` == 'random' or 'grid',
        when `search_type` == None, `data_scaler_param_ind` is automatically
        set to default 0.
        This parameter can be selected with either an integer index
        (zero based), or the str name for a given `data_scaler`.
        Likewise with `data_scaler`, if passed list input, this means
        a list was passed to `data_scaler` and the indices should correspond.

        The different parameter distributions avaliable for each
        `data_scaler`, can be shown by calling :func:`Show_Data_Scalers`

        If 'default', use the saved value within self.default_ML_params.
        (default = 'default')

    sampler_param_ind :  int, str, or list of
        Each `sampler` has atleast one default parameter distribution
        saved with it. This parameter is used to select between different
        distributions to be used with `search_type` == 'random' or 'grid',
        when `search_type` == None, `sampler_param_ind` is automatically
        set to default 0.
        This parameter can be selected with either an integer index
        (zero based), or the str name for a given `sampler`.
        Likewise with `sampler`, if passed list input, this means
        a list was passed to `sampler` and the indices should correspond.

        The different parameter distributions avaliable for each
        `sampler`, can be shown by calling :func:`Show_Samplers`

        If 'default', use the saved value within self.default_ML_params.
        (default = 'default')

    feat_selector_param_ind : int, str, or list of
         Each `feat_selector` has atleast one default parameter distribution
        saved with it. This parameter is used to select between different
        distributions to be used with `search_type` == 'random' or 'grid',
        when `search_type` == None, `feat_selector_param_ind` is automatically
        set to default 0.
        This parameter can be selected with either an integer index
        (zero based), or the str name for a given `feat_selector` param option.
        Likewise with `feat_selector`, if passed list input, this means
        a list was passed to `feat_selector` and the indices should correspond.

        The different parameter distributions avaliable for each
        `feat_selector`, can be shown by calling :func:`Show_Feat_Selectors`

        If 'default', use the saved value within self.default_ML_params.
        (default = 'default')

    class weight : {dict, 'balanced', None, 'default'}, optional
        Only used for binary and categorical problem types.
        Follows sklearn api class weight behavior. Typically, either use
        'balanced' in the case of class distribution imbalance, or None.
        If 'default', use the saved value within self.default_ML_params.

        (default = 'default')

    n_jobs : int or 'default', optional
        The number of jobs to use (if avaliable) during training ML models.
        This should be the number of procesors avaliable, if wanting to run
        as fast as possible.
        If 'default', use the saved value within self.default_ML_params.

        (default = 'default')

    n_iter : int or 'default', optional
        The number of random search parameters to try, used
        only if using random search.
        If 'default', use the saved value within self.default_ML_params.

        (default = 'default')

    data_to_use : {'all', 'data', 'covars'}, optional

        This setting allows the user to optionally
        run an expiriment with either only the loaded
        data and/or only the loaded covars. Likewise,
        both can be used with the default param of 'all'.

        - 'all' : Uses all data + covars loaded
        - 'data' : Uses only the loaded data, and drops covars if any
        - 'covars' : Uses only the loaded covars, and drops data if any

        If 'default', use the saved value within self.default_ML_params.

        (default = 'default')

    compute_train_score : bool, optional
        If set to True, will compute and print the model pipelines
        training score in addition to validation scores.
        If 'default', use the saved value within self.default_ML_params.

        (default = 'default')

    random_state : int, RandomState instance, None or 'default', optional
        Random state, either as int for a specific seed, or if None then
        the random seed is set by np.random.
        If 'default', use the saved value within self,
        (defined when initing ABCD_ML class) ^,
        Or a different ML params random state is used, if defined when
        calling set default ML params.

        (default = 'default')

    calc_base_feature_importances : bool or 'default, optional
        If set to True, will store the base feature importances
        when running Evaluate or Test. Note, base feature importances
        are only avaliable for tree-based or linear models, specifically
        those with either coefs or feature_importance attributes.

        If 'default', use the saved value within self.default_ML_params.
        (default = 'default')

    calc_shap_feature_importances : bool or 'default, optional
        If set to True, will calculate SHAP (SHapley Additive exPlanations)
        for the model when running Evaluate or Test.
        Note: For any case where the underlytin model is not tree or linear
        based, e.g. an ensemble of different methods, or non-linear svm,
        these values are estimated by a kernel explainer function which is
        very compute intensive.
        Read more about shap on their github:

        https://github.com/slundberg/shap

        If 'default', use the saved value within self.default_ML_params.
        (default = 'default')

    return_model : bool, optional
        If `return_model` is True, then model constructed and tested
        will be returned in addition to the score. If False,
        just the score will be returned.

        (default = False)

    extra_params : dict or 'default', optional
        Any extra params being passed. Typically, extra params are
        added when the user wants to provide a specific model/classifier,
        or data scaler, with updated (or new) parameters.
        These can be supplied by creating another dict within extra_params.
        E.g., ::

            extra_params[model_name] = {'model_param' : new_value}

        Where model param is a valid argument for that model, and model_name in
        this case is the str indicator passed to model_type.
        If 'default', use the saved value within self.default_ML_params.

        (default = 'default')

    Returns
    ----------
    train_score : array-like
        A numpy array of scores as determined by the passed
        metric/scorer(s) on the provided testing set.

    test_score : array-like
        A numpy array of scores as determined by the passed
        metric/scorer(s) on the provided testing set.

    model (if return_model == True)
        The sklearn api trained model object.
    '''

    # Perform pre-modeling check
    self._premodel_check()

    # Create the set of ML_params from passed args + default args
    ML_params = self._make_ML_params(args=locals())

    # Print the params being used
    self._print_model_params(model_type, ML_params, ensemble_type,
                             ensemble_split, model_type_param_ind,
                             test=True)

    # Init the Model object with modeling params
    self._init_model(model_type, ML_params, ensemble_type, ensemble_split,
                     model_type_param_ind)

    # If not train subjects or test subjects passed, use class
    if train_subjects is None:
        train_subjects = self.train_subjects
    if test_subjects is None:
        test_subjects = self.test_subjects

    # Train the model w/ selected parameters and test on test subjects
    train_scores, scores = self.Model.Test_Model(self.all_data, train_subjects,
                                                 test_subjects)

    # Print out score for all passed metrics
    scorer_strs = self.Model.scorer_strs
    self._print()

    score_list, score_type_list = [], []
    if ML_params['compute_train_score']:
        score_list.append(train_scores)
        score_type_list.append('Training')

    score_list.append(scores)
    score_type_list.append('Testing')

    for s, name in zip(score_list, score_type_list):

        self._print(name + ' Scores')
        self._print(''.join('_' for i in range(len(name) + 7)))

        for i in range(len(scorer_strs)):

            self._print('Metric: ', scorer_strs[i])

            scr = s[i]
            if len(scr.shape) > 0:

                for score_by_class, class_name in zip(scr, self.targets_key):
                    self._print('for target class: ', class_name)
                    self._print(name + ' Score: ', score_by_class)

            else:
                self._print(name + ' Score: ', scr)

    # Optionally return the model object itself
    if return_model:
        return score_list, self.Model.model

    return score_list


def _premodel_check(self, problem_type='default'):
    '''Internal helper function to ensure that self._prepare_data()
    has been called, and to force a train/test split if not already done.
    Will also call Set_Default_ML_Params if not already called.

    Parameters
    ----------
    problem_type : {'regression', 'binary', 'categorical', 'default'}, optional

        - 'regression' : For ML on float or ordinal target data
        - 'binary' : For ML on binary target data
        - 'categorical' : For ML on categorical target data,
                          as either multilabel or multiclass.
        - 'default' : Use the name problem type within self.default_ML_params.
    '''

    if self.all_data is None:
        self._prepare_data()

    if self.train_subjects is None:

        print('No train-test set defined! \
              Performing one automatically with default test split =.25')
        print('If no test set is intentional, \
              call self.Train_Test_Split(test_size=0)')

        self.Train_Test_Split(test_size=.25)

    if self.default_ML_params == {}:

        self._print('Setting default ML params.')
        self._print('Note, if the following values are not desired,',
                    'call self.Set_Default_ML_Params()')
        self._print('Or just pass values everytime to Evaluate',
                    'or Test, and these default values will be ignored')

        self.Set_Default_ML_Params(problem_type=problem_type)


def _make_ML_params(self, args):

    ML_params = {}

    # If passed param is default use default value.
    # Otherwise use passed value.
    for key in args:
        if args[key] == 'default':
            ML_params[key] = self.default_ML_params[key]
        elif key != 'self':
            ML_params[key] = args[key]

    # Fill in any missing params w/ default value.
    for key in self.default_ML_params:
        if key not in ML_params:
            ML_params[key] = self.default_ML_params[key]

    return ML_params


def _print_model_params(self, model_type, ML_params, ensemble_type,
                        ensemble_split, model_type_param_ind, test=False):

    if test:
        self._print('Running Test with:')
    else:
        self._print('Running Evaluate with:')

    self._print('model_type =', model_type)
    self._print('problem_type =', ML_params['problem_type'])
    self._print('metric =', ML_params['metric'])
    self._print('data_scaler =', ML_params['data_scaler'])
    self._print('sampler =', ML_params['sampler'])
    self._print('feat_selector =', ML_params['feat_selector'])

    if not test:
        self._print('n_splits =', ML_params['n_splits'])
        self._print('n_repeats =', ML_params['n_repeats'])

    self._print('int_cv =', ML_params['int_cv'])

    self._print('ensemble_type =', ensemble_type)
    self._print('ensemble_split =', ensemble_split)

    self._print('search_type =', ML_params['search_type'])
    self._print('model_type_param_ind =', model_type_param_ind)
    self._print('data_scaler_param_ind =', ML_params['data_scaler_param_ind'])
    self._print('sampler_param_ind =', ML_params['sampler_param_ind'])
    self._print('feat_selector_param_ind =',
                ML_params['feat_selector_param_ind'])

    if ML_params['problem_type'] != 'regression':
        self._print('class_weight =', ML_params['class_weight'])

    self._print('n_jobs =', ML_params['n_jobs'])
    self._print('n_iter =', ML_params['n_iter'])
    self._print('data_to_use =', ML_params['data_to_use'])
    self._print('compute_train_score =', ML_params['compute_train_score'])

    self._print('random_state =', ML_params['random_state'])
    self._print('calc_base_feature_importances =',
                ML_params['calc_base_feature_importances'])
    self._print('calc_shap_feature_importances =',
                ML_params['calc_shap_feature_importances'])
    self._print('extra_params =', ML_params['extra_params'])
    self._print()


def _init_model(self, model_type, ML_params, ensemble_type, ensemble_split,
                model_type_param_ind):

    problem_types = {'binary': Binary_Model, 'regression': Regression_Model,
                     'categorical': Categorical_Model}

    assert ML_params['problem_type'] in problem_types, \
        "Invalid problem type!"

    Model = problem_types[ML_params['problem_type']]

    self.Model = Model(model_type, ML_params, model_type_param_ind, self.CV,
                       self.data_keys,
                       self.covars_keys, self.cat_keys, self.targets_key,
                       self.targets_encoder, ensemble_type, ensemble_split,
                       self._print)


def _print_summary_score(self, name, summary_scores, n_repeats):

    self._print('Mean ' + name + ' score: ', summary_scores[0])

    if n_repeats > 1:
        self._print('Macro std in ' + name + ' score: ',
                    summary_scores[1])
        self._print('Micro std in ' + name + ' score: ',
                    summary_scores[2])
    else:
        self._print('std in ' + name + ' score: ',
                    summary_scores[1])

    self._print()


def Get_Base_Feat_Importances(self, top_n=None):
    '''Returns a pandas series with
    the base feature importances as calculated from the
    last run :func:`Evaluate` or :func:`test`.

    .. WARNING::
        `calc_base_feature_importances` must have been set to True,
        during the last call to :func:`Evaluate` or :func:`test`,
        AND the `model_type` must have been a linear or tree-based model
        (with no extra ensembling).

    Parameters
    ----------
    top_n : int or None, optional
        If not None, then will only return the top_n
        number of features, by base feature importance.

    Returns
    ----------
    pandas Series
        A sorted series containing the base feature importances,
        as averaged over folds for Evaluate, and as is for Test.
        Unless top_n is passed, then just a series with the top_n
        features is returnes
    '''

    assert len(self.Model.feature_importances) > 0,\
        "Either calc_base_feat_importances not set to True, or bad model_type!"

    importances = np.mean(np.abs(self.Model.feature_importances))
    importances.sort_values(ascending=False, inplace=True)

    if top_n:
        return importances[:top_n]

    return importances


def Get_Shap_Feat_Importances(self, top_n=None):
    '''Returns a pandas series with
    the shap feature importances, specifically the
    absolute mean shap value per feature, as calculated from the
    last run :func:`Evaluate` or :func:`test`. If running this function
    on categorical data, the average shap values will be computed
    by further averaging over each target variables mean shap values.

    .. WARNING::
        `calc_shap_feature_importances` must have been set to True,
        during the last call to :func:`Evaluate` or :func:`test`

    Parameters
    ----------
    top_n : int or None, optional
        If not None, then will only return the top_n
        number of features, by abs mean shap feature importance.

    Returns
    ----------
    pandas Series
        A sorted series containing the abs mean shap feature importances,
        as averaged over repeated folds for Evaluate, and as is for Test.
        Unless top_n is passed, then just a series with just the top_n
        features is returned.
    '''
    assert len(self.Model.shap_df) > 0, \
        "calc_shap_feature_importances must be set to True!"

    # for categorical
    if 'pandas' not in str(type(self.Model.shap_df)):

        # First grab copy of first ind as base
        shap_df = self.Model.shap_df[0].copy()

        # Get mean across list
        shap_df_arrays = [np.array(df) for df in self.Model.shap_df]
        mean_shap_array = np.mean(shap_df_arrays, axis=0)

        # Set values in df
        shap_df[list(shap_df)] = mean_shap_array

        # Store in avg_shap_df - for plotting
        self.avg_shap_df = shap_df

    else:
        shap_df = self.Model.shap_df

    shap_importances = np.mean(np.abs(shap_df))
    shap_importances.sort_values(ascending=False, inplace=True)

    if top_n:
        return shap_importances[:top_n]

    return shap_importances
