"""
_ML.py
====================================
Main class extension file for the Machine Learning functionality
"""
import pandas as pd
import numpy as np
import os
import pickle as pkl

from tqdm import tqdm, tqdm_notebook

from .Input_Tools import is_value_subset
from ..helpers.Data_Helpers import get_unique_combo_df, reverse_unique_combo_df
from ..helpers.ML_Helpers import compute_macro_micro, conv_to_list, get_avaliable_run_name
from ..pipeline.Model_Pipeline import Model_Pipeline


def Set_Default_ML_Params(self, problem_type='default', target='default',
                          model='default', model_params='default',
                          metric='default', weight_metric='default',
                          loader='default',
                          loader_scope='default', loader_params='default',
                          imputer='default',
                          imputer_scope='default', imputer_params='default',
                          scaler='default', scaler_scope='default',
                          scaler_params='default', 
                          transformer='default',
                          transformer_scope='default',
                          transformer_params='default',
                          sampler='default',
                          sample_on='default', sampler_params='default',
                          feat_selector='default',
                          feat_selector_params='default', ensemble='default',
                          ensemble_split='default', ensemble_params='default',
                          splits='default', n_repeats='default',
                          search_type='default', search_splits='default',
                          search_n_iter='default',
                          scope='default',
                          subjects='default',
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

        (default = 'default')

    weight_metric : bool, list of, optional
        If True, then the metric of interest will be weighted within
        each repeated fold by the number of subjects in that validation set.
        This parameter only makes sense for custom split behavior where 
        validation folds end up with different sizes. When default CV schemes are
        used there is no point weighting by very simmilar numbers.

        If you are passing mutiple metrics, then you can also pass a 
        list of values for weight_metric, with each value as True or False
        as to if the corresponding metric by index should be weighted.

        As a reminder, the first metric specified will be used as the
        parameter to optimize if a hyperparameter search is conducted.
        Likewise, if it is specified that the first metric be weighted,
        then the weighted metric will be used in the hyper-param search!

        If 'default', and not already defined, set to False
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

        If an imputer str is passed, then it must be a valid imputer
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

        - 'float'
            To apply to all non-categorical columns, in both
            loaded data and covars.

        - 'cat' or 'categorical'
            To apply to just loaded categorical data.

        - array-like of strs
            Can pass specific col names in as array-like
            to select only those cols.

        If 'default', and not already defined, set to ['float', 'cat']
        (default = 'default')

    imputer_params : int, str or list of, optional
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

        - 'float'
            To apply to all non-categorical columns, in both
            loaded data and covars.

        - 'data'
            To apply to all loaded data columns only.

        - 'data files'
            To apply to just columns which were originally loaded as data files.

        - 'float covars' or 'fc'
            To apply to all non-categorical, float covars columns only.

        - 'all'
            To apply to everything, regardless of float/cat or data/covar.

        - 'cat' or 'categorical'
            To apply to just loaded categorical data.

        - 'covars'
            To apply to all loaded covar columns only.

        - array-like of strs
            Can pass specific col names in as array-like
            to select only those cols.

        If 'default', and not already defined, set to 'float'
        (default = 'default')

    scaler_params : int, str, or list of, optional
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

    transformer : str, list or None, optional
        Transformers are any type of transformation to
        the data that changes the number of features in non-deterministic
        or not simply removal (i.e., as in feat_selectors), for example
        applying a PCA, where both the number of features change, but
        also the new features do not 1:1 correspond to the original
        features. 

        For a full list of supported options call:
        :func:`Show_Transformers` or view the docs at :ref:`Transformers`
    
    transformer_scope : str, list, tuple or None, optional
        `transformer_scope` refers to the "scope" or rather columns in
        which each passed scaler (if multiple), should be applied.
        If a list of transformers is passed, then scopes should also be a
        list with index corresponding to each transformer.
        If less then the number of transformers are passed, then the
        first passed transformer_scope will be used for all of the
        remaining transformers. Likewise, if no transformer is passed, this
        parameter will be ignored!

        Each transformer scope can be either,

        - 'float'
            To apply to all non-categorical columns, in both
            loaded data and covars.

        - 'data'
            To apply to all loaded data columns only.

        - 'data files'
            To apply to just columns which were originally loaded as data files.

        - 'float covars' or 'fc'
            To apply to all non-categorical, float covars columns only.

        - 'all'
            To apply to everything, regardless of float/cat or data/covar.

        - 'cat' or 'categorical'
            To apply to just loaded categorical data.

        - 'covars'
            To apply to all loaded covar columns only.

        - array-like of strs (not tuple)
            Can pass specific col names in as array-like
            to select only those cols.

        - tuple of str
            Can pass a tuple of scopes, to replicate the base
            object, run seperately with each entry of the tuple

        If 'default', and not already defined, set to 'float'
        (default = 'default')

    transformer_params : int, str, or list of, optional
        Each `transformer` has atleast one default parameter distribution
        saved with it.

        This parameter is used to select between different
        distributions to be used with different search types,
        when `search_type` == None, `model_params` is automatically
        set to default 0.

        This parameter can be selected with either an integer index
        (zero based), or the str name for a given `transformer`.
        Likewise with `transformer`, if passed list input, this means
        a list was passed to `transformer` and the indices should correspond.

        The different parameter distributions avaliable for each
        `transformer`, can be shown by calling :func:`Show_Transformers`
        or on the docs at :ref:`Transformers`

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

    scope : {'all', 'data', 'covars'} or array, optional
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

    subjects : 'all', array-like or str, optional
        This parameter allows the user to optionally run
        an Evaluation run with just a subset of the loaded subjects.
        It is designed to be to be used after a global train test split
        has been defined (see :func:`Train_Test_Split`), for cases such
        as, creating and testing models on just Males, or just Females.

        If set to 'all' (as is by default), all avaliable subjects will be
        used.

        `subjects` can accept either a specific array of subjects,
        or even a loc of a text file (formatted one subject per line) in
        which to read from. Note: do not pass a tuple of subjects, as that
        is reserved for specifying special behavior.

        Alternatively, `subjects` will accept a tuple, (Note:
        it must be a tuple!), where the first element is a loaded strat key,
        or a list of, and the second is an int value. In this case,
        `subjects`, will be set to the subset of subjects associated
        with the specified strat values (or combination) that have that value.

        For example, if sex was loaded within strat, and ('sex', 0) was
        passed to `subjects`, then :func:`Evaluate` would be run
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
        ( Defined in :class:`ABCD_ML.ABCD_ML`)
        
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

    self.default_ML_params.set_values(locals())


def Set_Default_ML_Verbosity(
 self, save_results='default', progress_bar='default', compute_train_score='default',
 show_init_params='default', fold_name='default',
 time_per_fold='default', score_per_fold='default', fold_sizes='default',
 best_params='default', save_to_logs='default'):
    '''This function allows setting various verbosity options that effect
    output during :func:`Evaluate` and :func:`Test`.

    Parameters
    ----------
    save_results : bool, optional
        If True, all results returned by Evaluate
        will be saved within the log dr (if one exists!),
        under run_name + .eval, and simmilarly for results
        returned by Test, but as run_name + .test.

        if 'default', and not already defined, set to False.
        (default = 'default')

    progress_bar : bool, optional
        If True, a progress bar, implemented in the python
        library tqdm, is used to show progress during use of
        :func:`Evaluate` , If False, then no progress bar is shown.
        This bar should work both in a notebook env and outside one,
        assuming self.notebook has been set correctly.

        if 'default', and not already defined, set to True.
        (default = 'default')

    compute_train_score : bool, optional
        If True, then metrics and raw preds will also be
        computed on the training set in addition to just the 
        eval or testing set.

        if 'default', and not already defined, set to False.
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

    best_params : bool, optional
        If True, print the best search params found after every
        param search.

    save_to_logs : bool, optional
        If True, then when possible, and with the selected model
        verbosity options, verbosity ouput will be saved to the
        log file.

        if 'default', and not already defined, set to False.
        (default = 'default')
    '''

    if save_results != 'default':
        self.default_ML_verbosity['save_results'] = save_results
    elif 'save_results' not in self.default_ML_verbosity:
        self.default_ML_verbosity['save_results'] = False

    if progress_bar != 'default':
        if progress_bar is True:
            if self.notebook:
                self.default_ML_verbosity['progress_bar'] = tqdm_notebook
            else:
                self.default_ML_verbosity['progress_bar'] = tqdm
        else:
            self.default_ML_verbosity['progress_bar'] = None
    elif 'progress_bar' not in self.default_ML_verbosity:
        if self.notebook:
            self.default_ML_verbosity['progress_bar'] = tqdm_notebook
        else:
            self.default_ML_verbosity['progress_bar'] = tqdm

    if compute_train_score != 'default':
        self.default_ML_verbosity['compute_train_score'] = compute_train_score
    elif 'compute_train_score' not in self.default_ML_verbosity:
        self.default_ML_verbosity['compute_train_score'] = False

    if show_init_params != 'default':
        self.default_ML_verbosity['show_init_params'] = show_init_params
    elif 'show_init_params' not in self.default_ML_verbosity:
        self.default_ML_verbosity['show_init_params'] = True

    if fold_name != 'default':
        self.default_ML_verbosity['fold_name'] = fold_name
    elif 'fold_name' not in self.default_ML_verbosity:
        self.default_ML_verbosity['fold_name'] = False

    if time_per_fold != 'default':
        self.default_ML_verbosity['time_per_fold'] = time_per_fold
    elif 'time_per_fold' not in self.default_ML_verbosity:
        self.default_ML_verbosity['time_per_fold'] = False

    if score_per_fold != 'default':
        self.default_ML_verbosity['score_per_fold'] = score_per_fold
    elif 'score_per_fold' not in self.default_ML_verbosity:
        self.default_ML_verbosity['score_per_fold'] = False

    if fold_sizes != 'default':
        self.default_ML_verbosity['fold_sizes'] = fold_sizes
    elif 'fold_sizes' not in self.default_ML_verbosity:
        self.default_ML_verbosity['fold_sizes'] = False

    if best_params != 'default':
        self.default_ML_verbosity['best_params'] = best_params
    elif 'best_params' not in self.default_ML_verbosity:
        self.default_ML_verbosity['best_params'] = False

    if save_to_logs != 'default':
        self.default_ML_verbosity['save_to_logs'] = save_to_logs
    elif 'save_to_logs' not in self.default_ML_verbosity:
        self.default_ML_verbosity['save_to_logs'] = False

    self._print('Default ML verbosity set within self.default_ML_verbosity.')
    self._print('----------------------')
    for param in self.default_ML_verbosity:

        if param == 'progress_bar':
            if self.default_ML_verbosity[param] is None:
                self._print(param + ':', False)
            else:
                self._print(param + ':', True)

        else:
            self._print(param + ':', self.default_ML_verbosity[param])

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

    if self.default_ML_verbosity['save_to_logs']:
        _print = self._print
    else:
        _print = print

    level = kwargs.pop('level', None)

    # If no level passed, always print
    if level is None:
        _print(*args, **kwargs)

    elif level == 'name' and self.default_ML_verbosity['fold_name']:
        _print(*args, **kwargs)

    elif level == 'time' and self.default_ML_verbosity['time_per_fold']:
        _print(*args, **kwargs)

    elif level == 'score' and self.default_ML_verbosity['score_per_fold']:
        _print(*args, **kwargs)

    elif level == 'size' and self.default_ML_verbosity['fold_sizes']:
        _print(*args, **kwargs)

    elif level == 'params' and self.default_ML_verbosity['best_params']:
        _print(*args, **kwargs)


def Evaluate(self,
             model_pipeline,
             problem_spec,
             splits=3,
             n_repeats=2,
             train_subjects='train',
             run_name='default'):
    
    '''
    Returns
    ----------
    results : dict
        Dictionary containing:
        'summary_scores', A list representation of the
        printed summary scores, where the 0 index is the mean,
        1 index is the macro std, then second index is the micro std.
        'train_summary_scores', Same as summary scores, but only exists
        if train scores are computed.
        'raw_scores', a numpy array of numpy arrays,
        where each internal array contains the raw scores as computed for
        all passed in metrics, computed for each fold within
        each repeat. e.g., array will have a length of `n_repeats` * number of
        folds, and each internal array will have the same length as the number of
        metrics. Optionally, this could instead return a list containing as
        the first element the raw training score in this same format,
        and then the raw testing scores.
        'raw_preds', A pandas dataframe containing the raw predictions
        for each subject, in the test set, and
        'FIs' a list where each element corresponds to a passed feature importance.

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

    # Should save the params used here*** before any preproc done
    run_name =\
        get_avaliable_run_name(run_name, model_pipeline.model, self.eval_scores)
    self.last_run_name = run_name

    # Preproc model pipeline & specs
    problem_spec = self._preproc_problem_spec(problem_spec)
    model_pipeline = self._preproc_model_pipeline(model_pipeline,
                                                  problem_spec.n_jobs)

    # Get the the train subjects to use
    _train_subjects = self._get_subjects_to_use(train_subjects)
    
    # Print the params being used
    if self.default_ML_verbosity['show_init_params']:

        model_pipeline.print_all(self._print)
        problem_spec.print_all(self._print)

        self._print('Evaluate Params')
        self._print('---------------')
        self._print('splits =', splits)
        self._print('n_repeats =', n_repeats)
        self._print('train_subjects =', train_subjects)
        self._print('len(train_subjects) =', len(_train_subjects),
                    '(before overlap w/ problem_spec.subjects)')
        self._print('run_name =', run_name)
        self._print()

    # Init. the Model_Pipeline object with modeling params
    self._init_model(model_pipeline, problem_spec)

    # Get the Eval splits
    _, splits_vals, _ = self._get_split_vals(splits)

    # Evaluate the model
    train_scores, scores, raw_preds, FIs =\
        self.Model_Pipeline.Evaluate(self.all_data, _train_subjects,
                                     splits, n_repeats, splits_vals)

    # Set target and run name
    for fi in FIs:
        fi.set_target(problem_spec.target)
        fi.set_run_name(run_name)

    self._print()

    # Print out summary stats for all passed metrics
    if self.default_ML_verbosity['compute_train_score']:
        score_list = [train_scores, scores]
        score_type_list = ['Training', 'Validation']
    else:
        score_list = [scores]
        score_type_list = ['Validation']

    results = {}
    for scrs, name in zip(score_list, score_type_list):

        summary_scores = self._handle_scores(scrs, name,
                                             problem_spec.weight_metric,
                                             n_repeats, run_name,
                                             self.Model_Pipeline.n_splits_)

        if name == 'Validation':
            results['summary_scores'] = summary_scores
        else:
            results['train_summary_scores'] = summary_scores

    results['raw_scores'] = score_list
    results['raw_preds'] = raw_preds
    results['FIs'] = FIs

    self._save_results(results, run_name + '.eval')
    return results


def Test(self,
         model_pipeline,
         problem_spec,
         train_subjects='train',
         test_subjects='test',
         run_name='default'):
    '''Class method used to evaluate a specific model / data scaling
    setup on an explicitly defined train and test set.

    Parameters
    ----------
    run_name : str or None, optional
        Note: This param is seperate from eval_run_name, where
        eval_run_name refers to an optional name to load from,
        run_name refers to the name under which these results
        from Test should be stored. They are stored in self.test_scores,
        and the exact parameters used in self.test_settings.
        If left as None, then will just
        use a default name.

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
    weight_metric :
    loader :
    loader_scope :
    loader_params :
    imputer :
    imputer_scope :
    imputer_params :
    scaler :
    scaler_scope :
    scaler_params :
    transformer : 
    transformer_scope :
    transformer_params :
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
    scope :
    subjects :
    feat_importances :
    feat_importances_params :
    n_jobs :
    random_state :
    compute_train_score :
    cache :
    extra_params :

    Returns
    ----------
    results : dict
        Dictionary containing:
        'scores', the score on the test set by each metric,
        'raw_preds', A pandas dataframe containing the raw predictions
        for each subject, in the test set, and 'FIs' a list where
        each element corresponds to a passed feature importance.

    '''

    # Perform pre-modeling check
    self._premodel_check()

    # Get a free run name
    run_name =\
        get_avaliable_run_name(run_name, model_pipeline.model, self.test_scores)
    self.last_run_name = run_name

    # Preproc model pipeline & specs
    problem_spec = self._preproc_problem_spec(problem_spec)
    model_pipeline = self._preproc_model_pipeline(model_pipeline,
                                                  problem_spec.n_jobs)

    # Get the the train subjects + test subjects to use
    _train_subjects = self._get_subjects_to_use(train_subjects)
    _test_subjects = self._get_subjects_to_use(test_subjects)

    # Print the params being used
    if self.default_ML_verbosity['show_init_params']:

        model_pipeline.print_all(self._print)
        problem_spec.print_all(self._print)

        self._print('Test Params')
        self._print('---------------')
        self._print('train_subjects =', train_subjects)
        self._print('len(train_subjects) =', len(_train_subjects),
                    '(before overlap w/ problem_spec.subjects)')
        self._print('test_subjects =', test_subjects)
        self._print('len(test_subjects) =', len(_test_subjects),
                    '(before overlap w/ problem_spec.subjects)')
        self._print('run_name =', run_name)
        self._print()

    # Init the Model_Pipeline object with modeling params
    self._init_model(problem_spec, model_pipeline)

    # Train the model w/ selected parameters and test on test subjects
    train_scores, scores, raw_preds, FIs =\
        self.Model_Pipeline.Test(self.all_data, train_subjects,
                                 test_subjects)

    # Set run name
    for fi in FIs:
        fi.set_target(problem_spec.target)
        fi.set_run_name(run_name)

    # Print out score for all passed metrics
    metric_strs = self.Model_Pipeline.metric_strs
    self._print()

    score_list, score_type_list = [], []
    if self.default_ML_verbosity['compute_train_score']:
        score_list.append(train_scores)
        score_type_list.append('Training')

    score_list.append(scores)
    score_type_list.append('Testing')

    for s, name in zip(score_list, score_type_list):

        self._print(name + ' Scores')
        self._print(''.join('_' for i in range(len(name) + 7)))

        for i in range(len(metric_strs)):

            metric_name = metric_strs[i]
            self._print('Metric: ', metric_name)

            scr = s[i]
            if len(scr.shape) > 0:

                targets_key = self.Model_Pipeline.targets_key

                for score_by_class, class_name in zip(scr, targets_key):
                    self._print('for target class: ', class_name)
                    self._print(name + ' Score: ', score_by_class)
                    self._print()
                    self._add_to_scores(run_name, name, metric_name,
                                        'score', score_by_class,
                                        self.test_scores, class_name)

            else:
                self._print(name + ' Score: ', scr)
                self._print()
                self._add_to_scores(run_name, name, metric_name,
                                    'score', scr, self.test_scores)

    results = {}
    results['scores'] = score_list
    results['raw_preds'] = raw_preds
    results['FIs'] = FIs

    self._save_results(results, run_name + '.test')
    return results


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

    if self.default_ML_verbosity == {}:

        self._print('Setting default ML verbosity settings!')
        self._print('Note, if the following values are not desired,',
                    'call self.Set_Default_ML_Verbosity()')

        self.Set_Default_ML_Verbosity()


def _preproc_model_pipeline(self, model_pipeline, n_jobs):

    # Set values across each pipeline pieces params
    model_pipeline.preproc(n_jobs)
    
    # Proc sample_on if needed (by adding strat name)
    model_pipeline.check_samplers(self._add_strat_u_name)

    # Set split vals if search
    if model_pipeline.param_search is not None:

        _, split_vals, _ =\
            self._get_split_vals(model_pipeline.param_search.splits)
        model_pipeline.param_search.set_split_vals(split_vals)

    # Early check to see if imputer could even be needed
    model_pipeline.check_imputer(self.all_data)

    return model_pipeline


def _preproc_problem_spec(self, problem_spec):

    # Update target with actual target key
    target_key = self._get_targets_key(problem_spec.target)
    problem_spec.set_params(target=target_key)

    # Proc subjects to use
    final_subjects = self._get_subjects_to_use(problem_spec.subjects)
    problem_spec.set_final_subjects(final_subjects)

    # Set by class defaults
    if problem_spec.n_jobs == 'default':
        problem_spec.n_jobs = self.n_jobs

    if problem_spec.random_state == 'default':
        problem_spec.random_state = self.random_state

    # If any input has changed, manually (i.e., not by problem_spec init)
    problem_spec._proc_checks()
    
    return problem_spec


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


def _get_subjects_to_use(self, subjects_to_use):

    # If str passed, either loc to load, or train, test, all
    if isinstance(subjects_to_use, str):
        if subjects_to_use == 'all':
            subjects = self.all_data.index
        elif subjects_to_use == 'train':
            subjects = self.train_subjects
        elif subjects_to_use == 'test':
            subjects = self.test_subjects
        else:
            subjects = self._load_set_of_subjects(loc=subjects_to_use)

    # Other case is if passed value subset, determine the values to use
    elif is_value_subset(subjects_to_use):

        split_names, split_vals, sv_le =\
            self._get_split_vals(subjects_to_use.name)

        selected = split_vals[split_vals == subjects_to_use.value]
        subjects = set(selected.index)

        rev_values = reverse_unique_combo_df(selected, sv_le)[0]

        self.last_subjects_names = []
        for strat_name, value in zip(split_names, rev_values):
            if self.strat_u_name in strat_name:
                strat_name = strat_name.replace(self.strat_u_name, '')

            self.last_subjects_names.append((strat_name, value))

        self._print('subjects set to: ',
                    self.last_subjects_names)
        self._print()

    # Lastly, if not the above, assume it is an array-like of subjects
    else:
        subjects = self._load_set_of_subjects(subjects=subjects_to_use)

    return subjects


def _init_model(self, model_pipeline, problem_specs):

    # Set Model_Pipeline
    self.Model_Pipeline =\
        Model_Pipeline(model_pipeline, problem_specs, self.CV, self.Data_Scopes,
                       self.default_ML_verbosity['progress_bar'],
                       self.default_ML_verbosity['compute_train_score'],
                       self._ML_print)

def _handle_scores(self, scores, name, weight_metric, n_repeats, run_name, n_splits):

    all_summary_scores = []
    metric_strs = self.Model_Pipeline.metric_strs

    self._print(name + ' Scores')
    self._print(''.join('_' for i in range(len(name) + 7)))

    weight_metrics = conv_to_list(weight_metric, len(metric_strs))

    for i in range(len(metric_strs)):

        # Weight outputed scores if requested
        if weight_metrics[i]:
            weights = self.Model_Pipeline.n_test_per_fold
        else:
            weights = None

        metric_name = metric_strs[i]
        self._print('Metric: ', metric_name)
        score_by_metric = scores[:, i]

        if len(score_by_metric[0].shape) > 0:
            by_class = [[score_by_metric[i][j] for i in
                        range(len(score_by_metric))] for j in
                        range(len(score_by_metric[0]))]

            summary_scores_by_class =\
                [compute_macro_micro(class_scores, n_repeats,
                 n_splits, weights=weights) for class_scores in by_class]

            targets_key = self.Model_Pipeline.targets_key
            classes = self.Model_Pipeline.classes

            class_names =\
                self.targets_encoders[targets_key].inverse_transform(
                    classes.astype(int))

            for summary_scores, class_name in zip(summary_scores_by_class,
                                                  class_names):

                self._print('Target class: ', class_name)
                self._print_summary_score(name, summary_scores,
                                          n_repeats, run_name,
                                          metric_name, class_name, weights=weights)

            all_summary_scores.append(summary_scores_by_class)

        else:

            # Compute macro / micro summary of scores
            summary_scores = compute_macro_micro(score_by_metric,
                                                 n_repeats,
                                                 n_splits,
                                                 weights=weights)

            self._print_summary_score(name, summary_scores,
                                      n_repeats, run_name,
                                      metric_name, weights=weights)

            all_summary_scores.append(summary_scores)

    return all_summary_scores


def _print_summary_score(self, name, summary_scores, n_repeats, run_name,
                         metric_name, class_name=None, weights=None):
    '''Besides printing, also adds scores to self.eval_scores dict
    under run name.'''

    mn = 'Mean'
    if weights is not None:
        mn = 'Weighted ' + mn

    self._print(mn + ' ' + name + ' score: ', summary_scores[0])
    self._add_to_scores(run_name, name, metric_name, mn,
                        summary_scores[0], self.eval_scores,  class_name)

    if n_repeats > 1:
        self._print('Macro Std in ' + name + ' score: ',
                    summary_scores[1])
        self._print('Micro Std in ' + name + ' score: ',
                    summary_scores[2])
        self._add_to_scores(run_name, name, metric_name, 'Macro Std',
                            summary_scores[1], self.eval_scores, class_name)
        self._add_to_scores(run_name, name, metric_name, 'Micro Std',
                            summary_scores[2], self.eval_scores, class_name)
    else:
        self._print('Std in ' + name + ' score: ',
                    summary_scores[2])
        self._add_to_scores(run_name, name, metric_name, 'Std',
                            summary_scores[2], self.eval_scores, class_name)

    self._print()


def _add_to_scores(self, run_name, name, metric_name, val_type, val, scores,
                   class_name=None):

    if run_name not in scores:
        scores[run_name] = {}

    if name not in scores[run_name]:
        scores[run_name][name] = {}

    if metric_name not in scores[run_name][name]:
        scores[run_name][name][metric_name] = {}

    if class_name is None:
        scores[run_name][name][metric_name][val_type] = val

    else:
        if class_name not in scores[run_name][name][metric_name]:
            scores[run_name][name][metric_name][class_name] = {}

        scores[run_name][name][metric_name][class_name][val_type] =\
            val


def _save_results(self, results, save_name):

    if self.default_ML_verbosity['save_results'] and self.log_dr is not None:

        save_dr = os.path.join(self.exp_log_dr, 'results')
        os.makedirs(save_dr, exist_ok=True)

        save_spot = os.path.join(save_dr, save_name)
        with open(save_spot, 'wb') as f:
            pkl.dump(results, f)
