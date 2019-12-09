"""
_ML.py
====================================
Main class extension file for the Machine Learning functionality
"""
import pandas as pd
import numpy as np

from tqdm import tqdm, tqdm_notebook

from ..helpers.Data_Helpers import get_unique_combo_df, reverse_unique_combo_df
from ..helpers.ML_Helpers import compute_macro_micro
from ..pipeline.Model_Pipeline import (Regression_Model_Pipeline,
                                       Binary_Model_Pipeline,
                                       Categorical_Model_Pipeline,
                                       Multilabel_Model_Pipeline)


def Set_Default_ML_Params(self, problem_type='default', target='default',
                          model='default', model_params='default',
                          metric='default', imputer='default',
                          imputer_scope='default', imputer_params='default',
                          scaler='default', scaler_scope='default',
                          scaler_params='default', sampler='default',
                          sample_on='default', sampler_params='default',
                          feat_selector='default',
                          feat_selector_params='default', ensemble='default',
                          ensemble_split='default', ensemble_params='default',
                          splits='default', n_repeats='default',
                          search_type='default', search_splits='default',
                          search_n_iter='default', feats_to_use='default',
                          subjects_to_use='default',
                          feat_importances='default',
                          feat_importances_params='default',
                          n_jobs='default', random_state='default',
                          compute_train_score='default', cache='default',
                          extra_params='default'):
    '''Sets self.default_ML_params dictionary with user passed or default
    values. In general, if any argument is left as 'default' and it has
    not been previously defined, it will be set to a default value,
    sometimes passed on other values. See notes for rationale behind
    default ML params.

    Parameters
    ----------
    problem_type : str, optional

        - 'regression'
            For ML on float target data.

        - 'binary'
            For ML on binary target data.

        - 'categorical'
            For ML on categorical target data, as multiclass.

        - 'multilabel'
            On categorical multilabel data.

        - 'default'
            Use 'regression', if nothing else already defined.

        If 'default', and not already defined, set to 'regression'
        (default = 'default')

    target : int or str, optional
        The loaded target in which to use during modelling.
        This can be the int index, or the name of the target column.
        If only one target is loaded, just leave as default.

        If 'default', and not already defined, set to 0
        (default = 'default')

    model : str or list of str
        Each string refers to a type of model to train.
        If a list of strings is passed then an ensemble model
        will be created over all individual models.

        For a full list of supported options call:
        :func:`Show_Models` or view the docs at :ref:`Models`

        If 'default', and not already defined, set to 'linear'
        (default = 'default')

    model_params : int, str, or list of
        Each `model` has atleast one default parameter distribution
        saved with it.

        This parameter is used to select between different
        distributions to be used with different search types,
        when `search_type` == None, `model_params` is automatically
        set to default 0.

        This parameter can be selected with either an integer index
        (zero based), or the str name for a given `model`.
        Likewise with `model`, if passed list input, this means
        a list was passed to `model` and the indices should correspond.

        The different parameter distributions avaliable for each
        `model`, can be shown by calling :func:`Show_Models`
        or on the docs at :ref:`Models`

        If 'default', and not already defined, set to 0
        (default = 'default')

    metric : str or list, optional
        Indicator for which metric(s) to use for calculating
        score and during model parameter selection.
        If `metric` left as 'default', then the default metric
        for that problem types will be used.
        Note, some metrics are only avaliable for certain problem types.

        For a full list of supported metrics call:
        :func:`Show_Metrics` or view the docs at :ref:`Metrics`

        If 'default', and not already defined, set to default
        metric for the problem type.

        - 'regression'  : 'r2'
        - 'binary'      : 'macro roc auc'
        - 'categorical' : 'macro f1'
        - 'multilabel' : 'macro roc auc'

        (default = 'default')

    imputer : str, list or None, optional
        If there is any missing data (NaN's) that have been kept
        within data or covars, then an imputation strategy must be
        defined! This param controls the imputer to use, along with
        `imputer_scope` to determine what each imputer should cover.
        A single str can be passed, or a list of strs.

        There are a number of pre-defined imputers to select from,
        but the user can also pass a valid model str indicator here.
        This model str refers to the base_estimator to be used in
        an IterativeImputer, see :class:`sklearn.impute.IterativeImputer`

        If a model str is passed, then it must be a valid model
        for whatever scope is passed additional. If the `imputer_scope`
        passed is 'float' or specific set of column names, then a regression
        model type will be selected. If the scope is 'binary' or 'categorical',
        then a binary / multiclass model type will be selected.
        (Note: categorical cols are converted to multiclass first if nec.)

        For a full list of supported options call:
        :func:`Show_Imputers` or view the docs at :ref:`Imputers`

        If 'default', and not already defined, set to ['mean', 'median']
        (default = 'default')

    imputer_scope : str, list or None, optional
        The `imputer_scope` param determines the scope,
        or rather which columns the imputer should fill
        (in data / covars), for each `imputer` passed.
        Options are,

        - 'float' or 'f'
            To select just float data

        - 'categorical' or 'c'
            To select any categorical type
            (including binary) data
            regardless of encoding (e.g. one hot)

        - array-like of strs
            Can pass specific col names in as array-like
            to select only those cols.

        If 'default', and not already defined, set to ['float', 'categorical']
        (default = 'default')

    imputer_params : int, str or list of
        Each `imputer` has atleast one param distribution,
        which can be selected with an int index, or a corresponding
        str name. Likewise, a user can pass in a dictionary with their
        own custom values.

        This parameter is used to select between different
        distributions to be used with different search types,
        when `search_type` == None, `model_params` is automatically
        set to default 0.

        The different parameter distributions avaliable for each
        `imputer`, can be shown by calling :func:`Show_Imputers`
        or on the docs at :ref:`Imputers`

        Note: If a model was passed to the imputer, then
        `imputer_params` will refer to the parameters for that
        base model!

        If 'default', and not already defined, set to 0
        (default = 'default')

    scaler : str, list or None, optional
        `scaler` refers to the type of scaling to apply
        to the saved data (just data, not covars) during model evaluation.
        If a list is passed, then scalers will be applied in that order.
        If None, then no scaling will be applied.

        For a full list of supported options call:
        :func:`Show_Scalers` or view the docs at :ref:`Scalers`

        If 'default', and not already defined, set to 'standard'
        (default = 'default')

    scaler_scope : str, list or None, optional
        `scaler_scope` refers to the "scope" or rather columns in
        which each passed scaler (if multiple), should be applied.
        If a list of scalers is passed, then scopes should also be a
        list with index corresponding to each scaler.
        If less then the number of scalers are passed, then the
        first passed scaler scope will be used for all of the
        remaining scalers. Likewise, if no scaler is passed, this
        parameter will be ignored!

        Each scaler scope can be either,

        - 'all' or 'a'
            To apply to all non-categorical columns.

        - 'data' or 'd'
            To apply to all loaded data columns only.

        - 'covars' or 'c'
            To apply to all non-categorical covars columns only.

        - array-like of strs
            Can pass specific col names in as array-like
            to select only those cols.

        If 'default', and not already defined, set to 'data'
        (default = 'default')

    scaler_params : int, str, or list of
        Each `scaler` has atleast one default parameter distribution
        saved with it.

        This parameter is used to select between different
        distributions to be used with different search types,
        when `search_type` == None, `model_params` is automatically
        set to default 0.

        This parameter can be selected with either an integer index
        (zero based), or the str name for a given `scaler`.
        Likewise with `scaler`, if passed list input, this means
        a list was passed to `scaler` and the indices should correspond.

        The different parameter distributions avaliable for each
        `scaler`, can be shown by calling :func:`Show_Scalers`
        or on the docs at :ref:`Scalers`

        If 'default', and not already defined, set to 0
        (default = 'default')

    sampler : str, list or None, optional
        `sampler` refers optional to the type of resampling
        to apply within the model pipeline - to correct for
        imbalanced class distributions. These are different
        techniques for over sampling under distributed classed
        and under sampling over distributed ones.
        If a list is passed, then samplers will be fit and applied
        in that order.

        For a full list of supported options call:
        :func:`Show_Samplers` or view the docs at :ref:`Samplers`

        If 'default', and not already defined, set to None
        (default = 'default')

    sample_on : str, list or None, optional
        If `sampler` is set, then for each sampler used,
        this `sample_on` parameter must be set. This parameter
        dictates what the sampler should use as its class to
        re-sample. For example, the most typical case is passing
        in 't' which will then resample according to the
        distribution of the target variable.
        (If multipled loaded, then whichever is is selected in target param),
        The user can also pass in
        the names of any loaded strat columns (See: :func:`Load_Strat`),
        or a combination as list of both,
        similar to input accepted by the `stratify` param
        in :func:`Define_Validation_Strategy`.

        Note: The way sample_on is coded, any passed col name which is not
        loaded in Strat, will be interpreted as sample the target variable!
        That is why the default value is just t, but you can pass anything that
        isnt a loaded strat col, to have it sample on strat.

        When a list is passed to one sampler, then the unique combination
        of values fis created, and sampled on.

        In the case where a list of a samplers is passed, then a
        list of `sample_on` params should be passed, where the index
        correspond. `sample_on` would look like a list of lists in the case
        where you want to pass a combination of options, or differing combos
        to different samplers.

        If 'default', and not already defined, set to 'targets'
        (default = 'default')

    sampler_params :  int, str, or list of
        Each `sampler` has atleast one default parameter distribution
        saved with it.

        This parameter is used to select between different
        distributions to be used with different search types,
        when `search_type` == None, `model_params` is automatically
        set to default 0.

        This parameter can be selected with either an integer index
        (zero based), or the str name for a given `sampler`.
        Likewise with `sampler`, if passed list input, this means
        a list was passed to `sampler` and the indices should correspond.

        The different parameter distributions avaliable for each
        `sampler`, can be shown by calling :func:`Show_Samplers`
        or on the docs at :ref:`Samplers`

        If 'default', and not already defined, set to 0
        (default = 'default')

    feat_selector : str, list or None, optional
        `feat_selector` should be a str indicator or list of,
        for which feature selection to use, if a list, they will
        be applied in order.
        If None, then no feature selection will be used.

        For a full list of supported options call:
        :func:`Show_Feat_Selectors` or view the docs at :ref:`Feat Selectors`

        If 'default', and not already defined, set to None
        (default = 'default')

    feat_selector_params : int, str, or list of
         Each `feat_selector` has atleast one default parameter distribution
        saved with it.

        This parameter is used to select between different
        distributions to be used with different search types,
        when `search_type` == None, `model_params` is automatically
        set to default 0.

        This parameter can be selected with either an integer index
        (zero based), or the str name for a given `feat_selector` param option.
        Likewise with `feat_selector`, if passed list input, this means
        a list was passed to `feat_selector` and the indices should correspond.

        The different parameter distributions avaliable for each
        `feat_selector`, can be shown by calling :func:`Show_Feat_Selectors`
        or on the docs at :ref:`Feat Selectors`

        If 'default', and not already defined, set to 0
        (default = 'default')

    ensemble :  str or list of str,
        Each string refers to a type of ensemble to train,
        or 'basic ensemble' (default) for base behavior.
        Base ensemble behavior is either to not ensemble,
        if only one model type is passed,
        or when multiple models are passed,
        to simply train each one independently and
        average the predictions at test time (or max vote).

        The user can optionally pass other ensemble types,
        though with other types of ensembles there are two
        different types to consider. One additional set of
        ensemble types will require a parameter to be set for
        `ensemble_split`, as these ensembles need to be fit
        on a left out portion of the data. This ensemble split
        will importantly always do a stratified split for now,
        and not uphold any defined CV behavior.

        The other possible ensemble type is one based on a single
        estimator, for example Bagging. In this case, if a list of models
        is passed, a Basic Ensemble will be fit over the models, and
        the Bagging Classifier or Regressor built on that ensemble of
        models.

        If a list is passed to ensemble, then every
        item in the list must be a valid str indicator for
        a non 'basic ensemble' ensemble type, and each ensemble
        object passed will be fitted independly and then averaged
        using the 'basic ensemble' behvaior... so an ensemble of ensembles.

        For a full list of supported options call:
        :func:`Show_Ensembles` or view the docs at :ref:`Ensemble Types`

        If 'default', and not already defined, set to 'basic ensemble'
        (default = 'default')

    ensemble_split : float, int or None, optional
        If an ensemble(s) that requires fitting is passed,
        i.e., not "basic ensemble", then this param is
        the porportion of the train_data within each fold to
        use towards fitting the ensemble objects.
        If multiple ensembles are passed, they are all
        fit with the same fold of data.

        If 'default', and not already defined, set to .2
        (default = 'default')

    ensemble_params : int, str, or list of, optional
         Each `ensemble` has atleast one default parameter distribution
        saved with it.

        This parameter is used to select between different
        distributions to be used with different search types,
        when `search_type` == None, `model_params` is automatically
        set to default 0.

        This parameter can be selected with either an integer index
        (zero based), or the str name for a given `ensemble` param option.
        Likewise with `ensemble`, if passed list input, this means
        a list was passed to `ensemble` and the indices should correspond.

        If 'default', and not already defined, set to 0
        (default = 'default')

    splits : int, str or list, optional
        If `splits` is an int, then :func:`Evaluate` performs a repeated
        k-fold model evaluation, where `splits` refers to the k, and
        `n_repeats` refers to the number of repeats.
        E.g., if set to 3, then a 3-fold CV will be performed at each repeat.

        Note: the splits will be determined according to the validation
        strategy defined in :func:`Define_Validation_Strategy` if any!

        Alternatively, `splits` can be set to a str or list of
        strings, where each str refers to a column name loaded
        within strat! In this case, a leave-one-out CV will be
        performed on that strat value, or combination of values.
        The number of folds will therefore be equal to the number of unique
        values, and can optionally be repeated with `n_repeats` set to > 1.

        If 'default', and not already defined, set to 3
        (default = 'default')

    n_repeats : int or 'default', optional
        :func:`Evaluate` performs a repeated k-fold model evaluation,
        or a left out CV, based on input. `n_repeats` refers to the number of
        times to repeat this.

        Note: In the case where `splits` is set to a strat col and therefore
        a leave-one-out CV, setting `n_repeats` > 1, with a fixed random seed
        will be redundant.

        If 'default', and not already defined, set to 2
        (default = 'default')

    search_type : {None, str}
        The type of parameter search to conduct if any. If set to None,
        no hyperparameter search will be conducted.

        The option is to pass the name of a nevergrad optimizer.

        If 'default', and not already defined, set to None
        (default = 'default')

    search_splits : int, str or list, optional
        The number of internal folds to use during
        model k-fold parameter selection if `search_splits` is an int.

        Note: the splits will be determined according to the validation
        strategy defined in :func:`Define_Validation_Strategy` if any!

        Alternatively, `search_splits` can be set to a str or list of
        strings, where each str refers to a column name loaded
        within strat! In this case, a leave-one-out CV will be
        performed on that strat value, or combination of values.
        The number of search CV folds will therefore be equal to the
        number of unique values.

        If 'default', and not already defined, set to 3
        (default = 'default')

    search_n_iter : int or 'default', optional
        The number of random search parameters to try, used
        only if using random search.

        if 'default', and not already defined, set to 10.
        (default = 'default')

    feats_to_use : {'all', 'data', 'covars'} or array, optional
        This parameter allows the user to optionally
        run an expiriment with a subset of the loaded features
        / columns. Typically either only the loaded
        data and/or only the loaded covars. Specific key words
        exist for selecting these, or alternatively, an array-like
        of column keys can be passed in explicitly.

        - 'all'
            Uses all data + covars loaded

        - 'data'
            Uses only the loaded data, and drops covars if any.

        - 'covars'
            Uses only the loaded covars, and drops data if any

        - array-like of strs
            Can pass specific col names in as array-like
            to select only those cols.

        - wild card str or array-like
            If user passed str doesn't match with
            valid col name, will use as wildcard.

        The way the wild card system works is that if for all user
        passed strs that do not match a column name, they will be treated
        as wildcards. For example, if '._desikan' and '._lh' were passed
        as wildcards, any column name with both '._desikan' and '._lh', will
        be added to feats to use.

        You can also pass a list combination of any of the above,
        for example you could pass ['covars', specific_column_name, wildcard]
        to select all of the covariate columns, the specific column name(s)
        and any extra columns which match the wildcard(s).

        if 'default', and not already defined, set to 'all'.
        (default = 'default')

    subjects_to_use : 'all', array-like or str, optional
        This parameter allows the user to optionally run
        an Evaluation run with just a subset of the loaded subjects.
        It is designed to be to be used after a global train test split
        has been defined (see :func:`Train_Test_Split`), for cases such
        as, creating and testing models on just Males, or just Females.

        If set to 'all' (as is by default), all avaliable subjects will be
        used.

        `subjects_to_use` can accept either a specific array of subjects,
        or even a loc of a text file (formatted one subject per line) in
        which to read from. Note: do not pass a tuple of subjects, as that
        is reserved for specifying special behavior.

        Alternatively, `subjects_to_use` will accept a tuple, (Note:
        it must be a tuple!), where the first element is a loaded strat key,
        or a list of, and the second is an int value. In this case,
        `subjects_to_use`, will be set to the subset of subjects associated
        with the specified strat values (or combination) that have that value.

        For example, if sex was loaded within strat, and ('sex', 0) was
        passed to `subjects_to_use`, then :func:`Evaluate` would be run
        on just those subjects with sex == 0.

        if 'default', and not already defined, set to 'all'.
        (default = 'default')

    feat_importances : None, str or list, optional
        This parameter controls which feature importances should
        be calculated, and can be set to None, a single feature
        importance, or a list of different types.

        Different feature importances are restricted by problem_type
        in some cases, as well as will vary based on specific type of
        model used. For example, only linear models and tree based models
        are supported for calculating 'base' feature importances. With
        'shap' feature importance, any underlying model is supported, but
        only tree based and linear models can be computed quickly, with
        other models requiring a great deal of computation.

        Please view :ref:`Feat Importances` to learn more about the different
        options for calculating feature importance, as well as the
        distinction between 'local' and 'global' measures of
        feature importance, and also the tradeoffs and differences
        between computing
        feature importance based of only train data, only test or all
        of the data.

        If 'default', and not already defined, set to 'base'
        (default = 'default')

    feat_importances_params : int, str, dict or list of, optional
        Different feature importances may vary on different
        hyperparameters. If the selected feature importance has
        hyperparameters, this parameter either selects from default
        choices (using either int input, or str for selecting the name
        of the preset). If a list of feat_importances is passed, a
        corresponding list of feat_importances_params should be passed.

        A user-defined dictionary can passed as well, containing user
        specified values. When only changing one
        or two parameters from the default, any non-specified params
        will be replaced with the default value.

        See the docs for which parameters are required by which
        feature importance types, and for what the default values are.

        If 'default', and not already defined, set to 0
        (default = 'default')

    n_jobs : int or 'default', optional
        The number of jobs to use (if avaliable) during training ML models.
        This should be the number of procesors avaliable for fastest run times.

        if 'default', and not already defined, set to 1.
        (default = 'default')

    random_state : int, RandomState instance, None or 'default', optional
        Random state, either as int for a specific seed, or if None then
        the random seed is set by np.random.

        If 'default', use the saved value within self
        ( Defined in :func:`__init__` )
        Or the user can define a different random state for use in ML.
        (default = 'default')

    compute_train_score : bool, optional
        If set to True, then :func:`Evaluate` and :func:`Test`
        will compute, and print, training scores along with
        validation / test scores.

        if 'default', and not already defined, set to False.
        (default = 'default')

    cache : None or str, optional
        sklearn pipeline's allow the caching of fitted transformers.
        If this behavior is desired (in the cases where a non-model step take
        a long time to fit), then a str indicating the directory where
        the cache should be stored should be passed.

        Note: cache dr's are not automatically removed, as different Evaluate
        calls may benefit from overlapping cached steps. That said, throughout
        a longer expiriment, the size of the cache will grow fairly quickly!
        Therefore be careful to delete the cache when you are done, and to
        only use this option if you have the free storage.

    extra_params : dict or 'default', optional
        Any extra params being passed. Typically, extra params are
        added when the user wants to provide a specific model/classifier,
        or data scaler, with updated (or new) parameters.
        These can be supplied by creating another dict within extra_params.
        E.g., ::

            extra_params[model_name] = {'model_param' : new_value}

        Where model param is a valid argument for that model, and model_name in
        this case is the str indicator passed to model.
        If 'default', and not already defined, set to empty dict.

        (default = 'default')
    '''

    default_metrics = {'binary': 'macro roc auc', 'regression': 'r2',
                       'categorical': 'macro f1',
                       'multilabel': 'macro roc auc'}

    if problem_type != 'default':
        problem_type = problem_type.lower()
        assert problem_type in default_metrics, 'Invalid problem type passed!'
        self.default_ML_params['problem_type'] = problem_type
    elif 'problem_type' not in self.default_ML_params:
        self.default_ML_params['problem_type'] = 'regression'

    if target != 'default':
        self.default_ML_params['target'] = target
    elif 'target' not in self.default_ML_params:
        self.default_ML_params['target'] = 0

    if model != 'default':
        self.default_ML_params['model'] = model
    elif 'model' not in self.default_ML_params:
        self.default_ML_params['model'] = 'linear'

    if metric != 'default':
        self.default_ML_params['metric'] = metric
    elif 'metric' not in self.default_ML_params:

        self.default_ML_params['metric'] = \
            default_metrics[self.default_ML_params['problem_type']]

    if imputer != 'default':
        self.default_ML_params['imputer'] = imputer
    elif 'imputer' not in self.default_ML_params:
        self.default_ML_params['imputer'] = ['mean', 'median']

    if imputer_scope != 'default':
        self.default_ML_params['imputer_scope'] = imputer_scope
    elif 'imputer_scope' not in self.default_ML_params:
        self.default_ML_params['imputer_scope'] =\
            ['float', 'categorical']

    if scaler != 'default':
        self.default_ML_params['scaler'] = scaler
    elif 'scaler' not in self.default_ML_params:
        self.default_ML_params['scaler'] = 'standard'

    if scaler_scope != 'default':
        self.default_ML_params['scaler_scope'] = scaler_scope
    elif 'scaler_scope' not in self.default_ML_params:
        self.default_ML_params['scaler_scope'] = 'data'

    if sampler != 'default':
        self.default_ML_params['sampler'] = sampler
    elif 'sampler' not in self.default_ML_params:
        self.default_ML_params['sampler'] = None

    if sample_on != 'default':
        self.default_ML_params['sample_on'] = sample_on
    elif 'sample_on' not in self.default_ML_params:
        self.default_ML_params['sample_on'] = 'targets'

    if feat_selector != 'default':
        self.default_ML_params['feat_selector'] = feat_selector
    elif 'feat_selector' not in self.default_ML_params:
        self.default_ML_params['feat_selector'] = None

    if splits != 'default':
        self.default_ML_params['splits'] = splits
    elif 'splits' not in self.default_ML_params:
        self.default_ML_params['splits'] = 3

    if n_repeats != 'default':
        assert isinstance(n_repeats, int), 'n_repeats must be int'
        assert n_repeats > 0, 'n_repeats must be greater than 0'
        self.default_ML_params['n_repeats'] = n_repeats
    elif 'n_repeats' not in self.default_ML_params:
        self.default_ML_params['n_repeats'] = 2

    if search_splits != 'default':
        self.default_ML_params['search_splits'] = search_splits
    elif 'search_splits' not in self.default_ML_params:
        self.default_ML_params['search_splits'] = 3

    if ensemble != 'default':
        self.default_ML_params['ensemble'] = ensemble
    elif 'ensemble' not in self.default_ML_params:
        self.default_ML_params['ensemble'] = 'basic ensemble'

    if ensemble_split != 'default':
        self.default_ML_params['ensemble_split'] = ensemble_split
    elif 'ensemble_split' not in self.default_ML_params:
        self.default_ML_params['ensemble_split'] = .2

    if search_type != 'default':
        self.default_ML_params['search_type'] = search_type
    elif 'search_type' not in self.default_ML_params:
        self.default_ML_params['search_type'] = None

    if model_params != 'default':
        self.default_ML_params['model_params'] = model_params
    elif 'model_params' not in self.default_ML_params:
        self.default_ML_params['model_params'] = 0

    if imputer_params != 'default':
        self.default_ML_params['imputer_params'] = imputer_params
    elif 'imputer_params' not in self.default_ML_params:
        self.default_ML_params['imputer_params'] = 0

    if scaler_params != 'default':
        self.default_ML_params['scaler_params'] = scaler_params
    elif 'scaler_params' not in self.default_ML_params:
        self.default_ML_params['scaler_params'] = 0

    if sampler_params != 'default':
        self.default_ML_params['sampler_params'] = sampler_params
    elif 'sampler_params' not in self.default_ML_params:
        self.default_ML_params['sampler_params'] = 0

    if feat_selector_params != 'default':
        self.default_ML_params['feat_selector_params'] =\
            feat_selector_params
    elif 'feat_selector_params' not in self.default_ML_params:
        self.default_ML_params['feat_selector_params'] = 0

    if ensemble_params != 'default':
        self.default_ML_params['ensemble_params'] =\
            ensemble_params
    elif 'ensemble_params' not in self.default_ML_params:
        self.default_ML_params['ensemble_params'] = 0

    if n_jobs != 'default':
        assert isinstance(n_jobs, int), 'n_jobs must be int'
        self.default_ML_params['n_jobs'] = n_jobs
    elif 'n_jobs' not in self.default_ML_params:
        self.default_ML_params['n_jobs'] = 1

    if search_n_iter != 'default':
        assert isinstance(search_n_iter, int), 'search_n_iter must be int'
        assert search_n_iter > 0, 'search_n_iter must be greater than 0'
        self.default_ML_params['search_n_iter'] = search_n_iter
    elif 'search_n_iter' not in self.default_ML_params:
        self.default_ML_params['search_n_iter'] = 10

    if feats_to_use != 'default':
        self.default_ML_params['feats_to_use'] = feats_to_use
    elif 'feats_to_use' not in self.default_ML_params:
        self.default_ML_params['feats_to_use'] = 'all'
        self._print('No default feats_to_use passed,',
                    'set to all')

    if subjects_to_use != 'default':
        self.default_ML_params['subjects_to_use'] = subjects_to_use
    elif 'subjects_to_use' not in self.default_ML_params:
        self.default_ML_params['subjects_to_use'] = 'all'

    if compute_train_score != 'default':
        self.default_ML_params['compute_train_score'] = compute_train_score
    elif 'compute_train_score' not in self.default_ML_params:
        self.default_ML_params['compute_train_score'] = False

    if random_state != 'default':
        self.default_ML_params['random_state'] = random_state
    elif 'random_state' not in self.default_ML_params:
        self.default_ML_params['random_state'] = self.random_state

    if feat_importances != 'default':
        self.default_ML_params['feat_importances'] =\
            feat_importances
    elif 'feat_importances' not in self.default_ML_params:
        self.default_ML_params['feat_importances'] = 'base'

    if feat_importances_params != 'default':
        self.default_ML_params['feat_importances_params'] =\
            feat_importances_params
    elif 'feat_importances_params' not in self.default_ML_params:
        self.default_ML_params['feat_importances_params'] = 0

    if cache != 'default':
        self.default_ML_params['cache'] = cache
    elif 'cache' not in self.default_ML_params:
        self.default_ML_params['cache'] = None

    if extra_params != 'default':
        assert isinstance(extra_params, dict), 'extra params must be dict'
        self.default_ML_params['extra_params'] = extra_params
    elif 'extra_params' not in self.default_ML_params:
        self.default_ML_params['extra_params'] = {}

    self._print('Default ML params set within self.default_ML_params.')
    self._print('----------------------')
    for param in self.default_ML_params:
        self._print(param + ':', self.default_ML_params[param])

    self._print()


def Set_Default_ML_Verbosity(
 self, progress_bar='default', show_init_params='default', fold_name='default',
 time_per_fold='default', score_per_fold='default', fold_sizes='default',
 save_to_logs='default'):
    '''This function allows setting various verbosity options that effect
    output during :func:`Evaluate` and :func:`Test`.

    Parameters
    ----------
    progress_bar : bool, optional
        If True, a progress bar, implemented in the python
        library tqdm, is used to show progress during use of
        :func:`Evaluate` , If False, then no progress bar is shown.
        This bar should work both in a notebook env and outside one,
        assuming self.notebook has been set correctly.

        if 'default', and not already defined, set to True.
        (default = 'default')

    show_init_params : bool, optional
        If True, then print/show the parameters used before running
        Evaluate / Test. If False, then don't print the params used.

        if 'default', and not already defined, set to True.
        (default = 'default')

    fold_name : bool, optional
        If True, prints a rough measure of progress via
        printing out the current fold (somewhat redundant with the
        progress bar if used, except if used with other params, e.g.
        time per fold, then it is helpful to have the time printed
        with each fold). If False, nothing is shown.

        if 'default', and not already defined, set to False.
        (default = 'default')

    time_per_fold : bool, optional
        If True, prints the full time that a fold took to complete.

        if 'default', and not already defined, set to False.
        (default = 'default')

    score_per_fold : bool, optional
        If True, displays the score for each fold, though slightly less
        formatted then in the final display.

        if 'default', and not already defined, set to False.
        (default = 'default')

    fold_sizes : bool, optional
        If True, will show the number of subjects within each train
        and val/test fold.

        if 'default', and not already defined, set to False.
        (default = 'default')

    save_to_logs : bool, optional
        If True, then when possible, and with the selected model
        verbosity options, verbosity ouput will be saved to the
        log file.

        if 'default', and not already defined, set to False.
        (default = 'default')
    '''

    if progress_bar != 'default':
        if progress_bar is True:
            if self.notebook:
                self.ML_verbosity['progress_bar'] = tqdm_notebook
            else:
                self.ML_verbosity['progress_bar'] = tqdm
        else:
            self.ML_verbosity['progress_bar'] = None
    elif 'progress_bar' not in self.ML_verbosity:
        if self.notebook:
            self.ML_verbosity['progress_bar'] = tqdm_notebook
        else:
            self.ML_verbosity['progress_bar'] = tqdm

    if show_init_params != 'default':
        self.ML_verbosity['show_init_params'] = show_init_params
    elif 'show_init_params' not in self.ML_verbosity:
        self.ML_verbosity['show_init_params'] = True

    if fold_name != 'default':
        self.ML_verbosity['fold_name'] = fold_name
    elif 'fold_name' not in self.ML_verbosity:
        self.ML_verbosity['fold_name'] = False

    if time_per_fold != 'default':
        self.ML_verbosity['time_per_fold'] = time_per_fold
    elif 'time_per_fold' not in self.ML_verbosity:
        self.ML_verbosity['time_per_fold'] = False

    if score_per_fold != 'default':
        self.ML_verbosity['score_per_fold'] = score_per_fold
    elif 'score_per_fold' not in self.ML_verbosity:
        self.ML_verbosity['score_per_fold'] = False

    if fold_sizes != 'default':
        self.ML_verbosity['fold_sizes'] = fold_sizes
    elif 'fold_sizes' not in self.ML_verbosity:
        self.ML_verbosity['fold_sizes'] = False

    if save_to_logs != 'default':
        self.ML_verbosity['save_to_logs'] = save_to_logs
    elif 'save_to_logs' not in self.ML_verbosity:
        self.ML_verbosity['save_to_logs'] = False

    self._print('Default ML verbosity set within self.ML_verbosity.')
    self._print('----------------------')
    for param in self.ML_verbosity:

        if param == 'progress_bar':
            if self.ML_verbosity[param] is None:
                self._print(param + ':', False)
            else:
                self._print(param + ':', True)

        else:
            self._print(param + ':', self.ML_verbosity[param])

    self._print()


def _ML_print(self, *args, **kwargs):
    '''Overriding the print function to allow for
    customizable verbosity. This print is setup with specific
    settings for the Model_Pipeline class, for using Evaluate and Test.

    Parameters
    ----------
    args
        Anything that would be passed to default python print
    '''

    if self.ML_verbosity['save_to_logs']:
        _print = self._print
    else:
        _print = print

    level = kwargs.pop('level', None)

    # If no level passed, always print
    if level is None:
        _print(*args, **kwargs)

    elif level == 'name' and self.ML_verbosity['fold_name']:
        _print(*args, **kwargs)

    elif level == 'time' and self.ML_verbosity['time_per_fold']:
        _print(*args, **kwargs)

    elif level == 'score' and self.ML_verbosity['score_per_fold']:
        _print(*args, **kwargs)

    elif level == 'size' and self.ML_verbosity['fold_sizes']:
        _print(*args, **kwargs)


def Evaluate(self, run_name=None, problem_type='default', target='default',
             model='default', model_params='default', metric='default',
             imputer='default', imputer_scope='default',
             imputer_params='default', scaler='default',
             scaler_scope='default', scaler_params='default',
             sampler='default', sample_on='default', sampler_params='default',
             feat_selector='default', feat_selector_params='default',
             ensemble='default', ensemble_split='default',
             ensemble_params='default', splits='default',
             n_repeats='default', search_type='default',
             search_splits='default', search_n_iter='default',
             feats_to_use='default', subjects_to_use='default',
             feat_importances='default',
             feat_importances_params='default',
             n_jobs='default', random_state='default',
             compute_train_score='default', cache='default',
             extra_params='default'):
    '''Class method to be called during the model selection phase.
    Used to evaluated different combination of models and scaling, ect...

    Parameters
    ----------
    run_name : str or None, optional
        All results from `Evaluate`, or rather the metrics are
        saved within self.eval_scores by default (in addition to in
        the logs, though this way saves them in a more programatically
        avaliable way). This name is also used to
        store the exact params used in self.eval_settings.
        `run_name` refers to the specific name under which to
        store this Evaluate's run on results. If left as None, then will just
        use a default name.
    problem_type :
    target :
    model :
    model_params :
    metric :
    imputer :
    imputer_scope :
    imputer_params :
    scaler :
    scaler_scope :
    scaler_params :
    sampler :
    sample_on :
    sampler_params :
    feat_selector :
    feat_selector_params :
    ensemble :
    ensemble_split :
    ensemble_params :
    splits :
    n_repeats :
    search_type :
    search_splits :
    search_n_iter :
    feats_to_use :
    subjects_to_use :
    feat_importances :
    feat_importances_params :
    n_jobs :
    random_state :
    compute_train_score :
    cache :
    extra_params :

    Returns
    ----------
    raw_scores : array-like of array-like
        numpy array of numpy arrays,
        where each internal array contains the raw scores as computed for
        all passed in metrics, computed for each fold within
        each repeat.
        e.g., array will have a length of `n_repeats` * number of folds,
        and each internal array will have the same length as the number of
        metrics.
        Optionally, this could instead return a list containing as the first
        element the raw training score in this same format,
        and then the raw testing scores.

    raw_preds : pandas DataFrame
        A pandas dataframe containing the raw prediction for each subject,
        with both prob. and prediction. Will show the predictions per repeat by
        subject, as well as the internal fold the prediction was made in.

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
    self._premodel_check()

    # Create the set of ML_params from passed args + default args
    ML_params = self._make_ML_params(args=locals())

    # If no NaN, no imputer
    if not pd.isnull(self.all_data).any().any():
        ML_params['imputer'] = None

    # Print the params being used
    if self.ML_verbosity['show_init_params']:
        self._print_model_params(ML_params, test=False)

    run_name = self._get_avaliable_eval_scores_name(run_name,
                                                    ML_params['model'])
    self._print('Saving scores and settings with unique name:', run_name)
    self._print()

    # Save this specific set of settings
    run_settings = ML_params.copy()
    run_settings.update({'run_name': run_name})
    self.eval_settings[run_name] = run_settings

    # Init. the Model_Pipeline object with modeling params
    self._init_model(ML_params)

    # Proc. splits
    split_names, split_vals, sv_le = self._get_split_vals(ML_params['splits'])

    # Evaluate the model
    train_scores, scores, raw_preds, FIs =\
        self.Model_Pipeline.Evaluate(self.all_data, self.train_subjects,
                                     split_vals)

    # Set target and run name
    for fi in FIs:
        fi.set_target(ML_params['target'])
        fi.set_run_name(run_name)

    self._print()

    # Print out summary stats for all passed metrics
    if ML_params['compute_train_score']:
        score_list = [train_scores, scores]
        score_type_list = ['Training', 'Validation']
    else:
        score_list = [scores]
        score_type_list = ['Validation']

    for scrs, name in zip(score_list, score_type_list):
        self._handle_scores(scrs, name, ML_params, run_name,
                            self.Model_Pipeline.n_splits)

    # Return the raw scores from each fold
    return score_list, raw_preds, FIs


def Test(self, run_name=None, train_subjects=None, test_subjects=None,
         problem_type='default', target='default', model='default',
         model_params='default', metric='default', imputer='default',
         imputer_scope='default', imputer_params='default',
         scaler='default', scaler_scope='default', scaler_params='default',
         sampler='default', sample_on='default', sampler_params='default',
         feat_selector='default', feat_selector_params='default',
         ensemble='default', ensemble_split='default',
         ensemble_params='default', search_type='default',
         search_splits='default', search_n_iter='default',
         feats_to_use='default', subjects_to_use='default',
         feat_importances='default',
         feat_importances_params='default',
         n_jobs='default', random_state='default',
         compute_train_score='default', cache='default',
         extra_params='default', **kwargs):
    '''Class method used to evaluate a specific model / data scaling
    setup on an explicitly defined train and test set.

    Parameters
    ----------
    run_name : str or None, optional
        A run_name within Test if set to a previous Evaluate run name
        will load the settings used from that Evaluate call instead of
        additional params passed! This is useful when testing a best
        model on the test set, but be wary that it will ignore any
        new params passed. If left as None, will act as a normal, and
        used the passed params.

        (default = None)

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

    problem_type :
    target :
    model :
    model_params :
    metric :
    imputer :
    imputer_scope :
    imputer_params :
    scaler :
    scaler_scope :
    scaler_params :
    sampler :
    sample_on :
    sampler_params :
    feat_selector :
    feat_selector_params :
    ensemble :
    ensemble_split :
    ensemble_params :
    search_type :
    search_splits :
    search_n_iter :
    feats_to_use :
    subjects_to_use :
    feat_importances :
    feat_importances_params :
    n_jobs :
    random_state :
    compute_train_score :
    cache :
    extra_params :

    Returns
    ----------
    raw_scores : array-like
        A numpy array of scores as determined by the passed
        metric(s) on the provided testing set. Optionally,
        this could instead return a list containing as the first
        element the raw training score in this same format,
        and then the raw testing scores.

    raw_preds : pandas DataFrame
        A pandas dataframe containing the raw predictions for each subject,
        in the test set.
    '''

    # Perform pre-modeling check
    self._premodel_check()

    # Create the set of ML_params from passed args + default args
    # or from a passed run name.
    if run_name is None:
        args = locals()
    else:
        self._print('Loading existing settings from run:', run_name)
        self._print()
        args = self.eval_settings[run_name]

    ML_params = self._make_ML_params(args=args)

    # If no NaN, no imputer
    if not pd.isnull(self.all_data).any().any():
        ML_params['imputer'] = None

    # Print the params being used
    if self.ML_verbosity['show_init_params']:
        self._print_model_params(ML_params, test=True, kwargs=kwargs)

    # Init the Model_Pipeline object with modeling params
    self._init_model(ML_params)

    # If not train subjects or test subjects passed, use class
    if train_subjects is None:
        train_subjects = self.train_subjects
    if test_subjects is None:
        test_subjects = self.test_subjects

    # Train the model w/ selected parameters and test on test subjects
    train_scores, scores, raw_preds, FIs =\
        self.Model_Pipeline.Test(self.all_data, train_subjects,
                                 test_subjects)

    # Set run name
    for fi in FIs:
        fi.set_target(ML_params['target'])

    # Print out score for all passed metrics
    metric_strs = self.Model_Pipeline.metric_strs
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

        for i in range(len(metric_strs)):

            self._print('Metric: ', metric_strs[i])

            scr = s[i]
            if len(scr.shape) > 0:

                targets_key = self.Model_Pipeline.targets_key

                for score_by_class, class_name in zip(scr, targets_key):
                    self._print('for target class: ', class_name)
                    self._print(name + ' Score: ', score_by_class)
                    self._print()

            else:
                self._print(name + ' Score: ', scr)
                self._print()

    return score_list, raw_preds, FIs


def _premodel_check(self):
    '''Internal helper function to ensure that self._prepare_data()
    has been called, and to force a train/test split if not already done.
    Will also call Set_Default_ML_Params if not already called.
    '''

    if self.all_data is None:
        self._prepare_data()

    if self.train_subjects is None:

        raise RuntimeError('No train-test set defined!',
                           'If this is intentional, call Train_Test_Split',
                           'with test_size = 0')

    if self.default_ML_params == {}:

        self._print('Setting default ML params!')
        self._print('Note, if the following values are not desired,',
                    'call self.Set_Default_ML_Params()')
        self._print('Or just pass values everytime to Evaluate',
                    'or Test, and these default values will be ignored')
        self._print('')

        self.Set_Default_ML_Params()

    if self.ML_verbosity == {}:

        self._print('Setting default ML verbosity settings!')
        self._print('Note, if the following values are not desired,',
                    'call self.Set_Default_ML_Verbosity()')

        self.Set_Default_ML_Verbosity()


def _make_ML_params(self, args):

    ML_params = {}

    # If passed param is default use default value.
    # Otherwise use passed value.
    for key in args:

        try:
            if args[key] == 'default':
                ML_params[key] = self.default_ML_params[key]
            elif key != 'self':
                ML_params[key] = args[key]

        # If value error, set to key
        except ValueError:
            ML_params[key] = args[key]

    # Fill in any missing params w/ default value.
    for key in self.default_ML_params:
        if key not in ML_params:
            ML_params[key] = self.default_ML_params[key]

    return ML_params


def _print_model_params(self, ML_params, test=False, kwargs={}):

    if test:
        self._print('Running Test with:')
    else:
        self._print('Running Evaluate with:')

    self._print('target =', ML_params['target'])
    self._print('problem_type =', ML_params['problem_type'])

    self._print('model =', ML_params['model'])
    self._print('model_params =', ML_params['model_params'])

    ensmb_flag = ML_params['ensemble'] != 'basic ensemble'
    if isinstance(ML_params['model'], list) or ensmb_flag:
        self._print('ensemble =', ML_params['ensemble'])

        if ensmb_flag:
            self._print('ensemble_split =', ML_params['ensemble_split'])
            self._print('ensemble_params =',
                        ML_params['ensemble_params'])

    self._print('metric =', ML_params['metric'])

    if pd.isnull(self.all_data).any().any():
        self._print('imputer =', ML_params['imputer'])

        if ML_params['imputer'] is not None:
            self._print('imputer_scope =', ML_params['imputer_scope'])
            self._print('imputer_params =', ML_params['imputer_params'])

    self._print('scaler =', ML_params['scaler'])
    if ML_params['scaler'] is not None:
        self._print('scaler_scope =', ML_params['scaler_scope'])
        self._print('scaler_params =',
                    ML_params['scaler_params'])

    self._print('sampler =', ML_params['sampler'])
    if ML_params['sampler'] is not None:
        self._print('sample_on =', ML_params['sample_on'])
        self._print('sampler_params =', ML_params['sampler_params'])

    self._print('feat_selector =', ML_params['feat_selector'])
    if ML_params['feat_selector'] is not None:
        self._print('feat_selector_params =',
                    ML_params['feat_selector_params'])

    if not test:
        self._print('splits =', ML_params['splits'])
        self._print('n_repeats =', ML_params['n_repeats'])

    self._print('search_type =', ML_params['search_type'])
    if ML_params['search_type'] is not None:
        self._print('search_splits =', ML_params['search_splits'])
        self._print('search_n_iter =', ML_params['search_n_iter'])

    self._print('n_jobs =', ML_params['n_jobs'])

    if len(ML_params['feats_to_use']) > 50:
        self._print('feats_to_use = custom passed keys with len',
                    len(ML_params['feats_to_use']))
    else:
        self._print('feats_to_use =', ML_params['feats_to_use'])

    if len(ML_params['subjects_to_use']) > 20:
        self._print('subjects_to_use = custom passed keys with len',
                    len(ML_params['subjects_to_use']))
    else:
        self._print('subjects_to_use =', ML_params['subjects_to_use'])

    self._print('compute_train_score =', ML_params['compute_train_score'])
    self._print('random_state =', ML_params['random_state'])

    self._print('feat_importances =',
                ML_params['feat_importances'])
    self._print('feat_importances_params =',
                ML_params['feat_importances_params'])

    self._print('cache =', ML_params['cache'])
    self._print('extra_params =', ML_params['extra_params'])

    if len(kwargs) > 0:
        self._print('Ignoring the following invalid passed params:')
        for k in kwargs:
            self._print(k, '=', kwargs[k])
    self._print()


def _get_split_vals(self, splits):

    if isinstance(splits, int):
        split_names, split_vals, sv_le = None, None, None

    else:
        split_names = self._add_strat_u_name(splits)

        if isinstance(split_names, str):
            split_names = [split_names]

        split_vals, sv_le =\
            get_unique_combo_df(self.strat, split_names)

    return split_names, split_vals, sv_le


def _proc_feats_to_use(self, feats_to_use):

    if isinstance(feats_to_use, str):
        valid = ['data', 'd', 'covars', 'c', 'all', 'a']

        if feats_to_use in valid:
            return feats_to_use
        else:
            feats_to_use = [feats_to_use]

    final_feats_to_use = []
    restrict_keys = []

    all_feats =\
        self.all_data_keys['data_keys'] + self.all_data_keys['covars_keys']
    all_feats = np.array(all_feats)

    for feat in feats_to_use:
        if feat == 'data' or feat == 'd':
            final_feats_to_use += self.all_data_keys['data_keys']
        elif feat == 'covars' or feat == 'c':
            final_feats_to_use += self.all_data_keys['covars_keys']
        elif feat in all_feats:
            final_feats_to_use.append(feat)
        else:
            restrict_keys.append(feat)

    if len(restrict_keys) > 0:

        rest = list(all_feats[[all([r in a for r in restrict_keys]) for
                               a in all_feats]])
        final_feats_to_use += rest

    # If any repeats
    final_feats_to_use = list(set(final_feats_to_use))
    return final_feats_to_use


def _get_final_subjects_to_use(self, subjects_to_use):

    if subjects_to_use == 'all':
        subjects = self.all_data.index

    elif isinstance(subjects_to_use, tuple):
        split_names, split_vals, sv_le =\
            self._get_split_vals(subjects_to_use[0])

        selected = split_vals[split_vals == subjects_to_use[1]]
        subjects = set(selected.index)

        rev_values = reverse_unique_combo_df(selected, sv_le)[0]
        self._print('subjects_to_use set to: ', end='')

        for strat_name, value in zip(split_names, rev_values):
            if self.strat_u_name in strat_name:
                strat_name = strat_name.replace(self.strat_u_name, '')
            self._print(strat_name, '=', value, ',', end='', sep='')

        self._print()
        self._print()

    else:
        if isinstance(subjects_to_use, str):
            loc = subjects_to_use
            subjs = None

        else:
            loc = None
            subjs = subjects_to_use

        subjects = self._load_set_of_subjects(loc=loc, subjects=subjs)

    return subjects


def _init_model(self, ML_params):

    problem_types = {'binary': Binary_Model_Pipeline,
                     'regression': Regression_Model_Pipeline,
                     'categorical': Categorical_Model_Pipeline,
                     'multilabel': Multilabel_Model_Pipeline}

    assert ML_params['problem_type'] in problem_types, \
        "Invalid problem type!"

    Model_Pipeline = problem_types[ML_params['problem_type']]

    targets_key = self._get_targets_key(ML_params['target'])

    # Grab index info
    covar_scopes, cat_encoders = self._get_covar_scopes()

    # Conv sample_on params w/ added unique key here, if needed
    ML_params['sample_on'] = self._add_strat_u_name(ML_params['sample_on'])

    # Proc feats_to_use
    ML_params['feats_to_use'] =\
        self._proc_feats_to_use(ML_params['feats_to_use'])

    # Proc subjects to use
    ML_params['subjects_to_use'] =\
        self._get_final_subjects_to_use(ML_params['subjects_to_use'])

    # Grab search split_vals_here
    _, search_split_vals, _ = self._get_split_vals(ML_params['search_splits'])

    # Make model
    self.Model_Pipeline =\
        Model_Pipeline(ML_params, self.CV, search_split_vals,
                       self.all_data_keys, targets_key,
                       covar_scopes, cat_encoders,
                       self.ML_verbosity['progress_bar'],
                       self._ML_print)


def _get_avaliable_eval_scores_name(self, name, model):

    if name is None:
        if isinstance(model, list):
            name = 'ensemble'

        elif not isinstance(model, str):
            name = 'user passed'

        else:
            name = model

    if name in self.eval_scores:

        n = 0
        while name + str(n) in self.eval_scores:
            n += 1

        name = name + str(n)

    return name


def _handle_scores(self, scores, name, ML_params, run_name, n_splits):

    metric_strs = self.Model_Pipeline.metric_strs

    self._print(name + ' Scores')
    self._print(''.join('_' for i in range(len(name) + 7)))

    for i in range(len(metric_strs)):

        metric_name = metric_strs[i]
        self._print('Metric: ', metric_name)
        score_by_metric = scores[:, i]

        if len(score_by_metric[0].shape) > 0:
            by_class = [[score_by_metric[i][j] for i in
                        range(len(score_by_metric))] for j in
                        range(len(score_by_metric[0]))]

            summary_scores_by_class =\
                [compute_macro_micro(class_scores, ML_params['n_repeats'],
                 n_splits) for class_scores in by_class]

            targets_key = self.Model_Pipeline.targets_key
            classes = self.Model_Pipeline.classes

            class_names =\
                self.targets_encoders[targets_key].inverse_transform(
                    classes.astype(int))

            for summary_scores, class_name in zip(summary_scores_by_class,
                                                  class_names):

                self._print('Target class: ', class_name)
                self._print_summary_score(name, summary_scores,
                                          ML_params['n_repeats'], run_name,
                                          metric_name, class_name)

        else:

            # Compute macro / micro summary of scores
            summary_scores = compute_macro_micro(score_by_metric,
                                                 ML_params['n_repeats'],
                                                 n_splits)

            self._print_summary_score(name, summary_scores,
                                      ML_params['n_repeats'], run_name,
                                      metric_name)


def _print_summary_score(self, name, summary_scores, n_repeats, run_name,
                         metric_name, class_name=None):
    '''Besides printing, also adds scores to self.eval_scores dict
    under run name.'''

    self._print('Mean ' + name + ' score: ', summary_scores[0])
    self._add_to_eval_scores(run_name, name, metric_name, 'Mean',
                             summary_scores[0], class_name)

    if n_repeats > 1:
        self._print('Macro Std in ' + name + ' score: ',
                    summary_scores[1])
        self._print('Micro Std in ' + name + ' score: ',
                    summary_scores[2])
        self._add_to_eval_scores(run_name, name, metric_name, 'Macro Std',
                                 summary_scores[1], class_name)
        self._add_to_eval_scores(run_name, name, metric_name, 'Micro Std',
                                 summary_scores[2], class_name)
    else:
        self._print('Std in ' + name + ' score: ',
                    summary_scores[2])
        self._add_to_eval_scores(run_name, name, metric_name, 'Std',
                                 summary_scores[2], class_name)

    self._print()


def _add_to_eval_scores(self, run_name, name, metric_name, val_type, val,
                        class_name=None):

    if run_name not in self.eval_scores:
        self.eval_scores[run_name] = {}

    if name not in self.eval_scores[run_name]:
        self.eval_scores[run_name][name] = {}

    if metric_name not in self.eval_scores[run_name][name]:
        self.eval_scores[run_name][name][metric_name] = {}

    if class_name is None:
        self.eval_scores[run_name][name][metric_name][val_type] = val

    else:
        if class_name not in self.eval_scores[run_name][name][metric_name]:
            self.eval_scores[run_name][name][metric_name][class_name] = {}

        self.eval_scores[run_name][name][metric_name][class_name][val_type] =\
            val
