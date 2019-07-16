"""
_ML.py
====================================
Main class extension file for the Machine Learning functionality
"""
import numpy as np
from ABCD_ML.ML_Helpers import compute_macro_micro
from ABCD_ML.Model import Regression_Model, Binary_Model, Categorical_Model


def set_default_ML_params(self, problem_type='default', metric='default',
                          data_scaler='default', n_splits='default',
                          n_repeats='default', int_cv='default',
                          class_weight='default', n_jobs='default',
                          n_iter='default', random_state='default',
                          extra_params='default'):
    '''Sets the self.default_ML_params dictionary with user passed or default
    values. In general, if any argument is left as 'default' and it has
    not been previously defined, it will be set to a default value,
    sometimes passed on other values.
    See notes for rationale behind default ML params.

    Parameters
    ----------
    problem_type : {'regression', 'binary', 'categorical', 'default'}, optional

        - 'regression' : For ML on float or ordinal target data
        - 'binary' : For ML on binary target data
        - 'categorical' : For ML on categorical target data,
                          as either multilabel or multiclass.
        - 'default' : Use 'regression' if nothing else already defined

        (default = 'default')

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
        If 'default', and not already defined, set to default
        metric for the problem type.
        (default = 'default')

    data_scaler : str, optional
        `data_scaler` refers to the type of scaling to apply
        to the saved data during model evaluation.
        If 'default', and not already defined, set to 'standard'
        For a full list of supported options call:
        self.show_data_scalers()
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

    class weight : {dict, 'balanced', None, 'default'}, optional
        Only used for binary and categorical problem types.
        Follows sklearn api class weight behavior. Typically, either use
        'balanced' in the case of class distribution imbalance, or None.
        If 'default', and not already defined, set to 'balanced'
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

    random_state : int, RandomState instance, None or 'default', optional
        Random state, either as int for a specific seed, or if None then
        the random seed is set by np.random.
        If 'default', use the saved value within self,
        (defined when initing ABCD_ML class) ^,
        Or can define a different random state for use in ML.
        (default = 'default')

    extra_params : dict or 'default', optional
        Any extra params being passed. Typically, extra params are
        added when the user wants to provide a specific model/classifier,
        or data scaler, with updated (or new) parameters.
        These can be supplied by creating another dict within extra_params.
        E.g., extra_params[model_name] = {'model_param' : new_value}
        Where model param is a valid argument for that model, and model_name in
        this case is the str indicator passed to model_type.
        If 'default', and not already defined, set to empty dict.
        (default = 'default')

    Notes
    ----------
    `default_ML_params` are used in the case where the same settings are being set over
    and over. For example, if only exploring a binary problem_type, the default type should
    be set to 'binary', and then the user won't have to pass it as an argument everytime they call
    Evaluate().
    '''

    self.default_ML_params = {}

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

    if class_weight != 'default':
        self.default_ML_params['class_weight'] = class_weight

    elif 'class_weight' not in self.default_ML_params:
        self.default_ML_params['class_weight'] = 'balanced'
        self._print('No default class weight setting passed, set to balanced')

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

    if random_state != 'default':
        self.default_ML_params['random_state'] = random_state

    elif extra_params not in self.default_ML_params:
        self.default_ML_params['random_state'] = self.random_state
        self._print('No default random state passed, using class random',
                    'state value of', self.random_state)

    if extra_params != 'default':
        assert isinstance(extra_params, dict), 'extra params must be dict'
        self.default_ML_params['extra_params'] = extra_params

    elif extra_params not in self.default_ML_params:
        self.default_ML_params['extra_params'] = {}
        self._print('No default extra params passed, set to empty dict')

    self._print('Default params set.')
    self._print()


def Evaluate(self, model_type, problem_type='default', data_scaler='default',
             n_splits='default', n_repeats='default', int_cv='default',
             metric='default', class_weight='default', n_jobs='default',
             n_iter='default', random_state='default', extra_params='default'):

    '''Class method to be called during the model selection phase.
    Used to evaluated different combination of models and scaling, ect...

    Parameters
    ----------
    model_type : str or list of str,
        Each string refers to a type of model to train.
        If a list of strings is passed then an ensemble model
        will be created over all individual models.
        For a full list of supported options call:
        self.show_model_types(), with optional problem type parameter.

    problem_type : {'regression', 'binary', 'categorical', 'default'}, optional

        - 'regression' : For ML on float or ordinal target data
        - 'binary' : For ML on binary target data
        - 'categorical' : For ML on categorical target data,
                          as either multilabel or multiclass.
        - 'default' : Use the name problem type within self.default_ML_params.

        (default = 'default')

    data_scaler : str, optional
        `data_scaler` refers to the type of scaling to apply
        to the saved data during model evaluation.
        If 'default', use the saved value within self.default_ML_params.
        For a full list of supported options call:
        self.show_data_scalers()
        (default = 'default')

    n_splits : int or 'default', optional
        evaluate_model performs a repeated k-fold model evaluation,
        `n_splits` refers to the k. E.g., if set to 3, then a 3-fold
        repeated CV will be performed. This parameter is typically
        chosen as a trade off between bias and variance, in addition to
        as a function of sample size.
        If 'default', use the saved value within self.default_ML_params.
        (default = 'default')

    n_repeats : int or 'default', optional
        evaluate_model performs a repeated k-fold model evaluation,
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

    random_state : int, RandomState instance, None or 'default', optional
        Random state, either as int for a specific seed, or if None then
        the random seed is set by np.random.
        If 'default', use the saved value within self,
        (defined when initing ABCD_ML class) ^,
        Or a different ML params random state is used, if defined when
        calling set default ML params.
        (default = 'default')

    extra_params : dict or 'default', optional
        Any extra params being passed. Typically, extra params are
        added when the user wants to provide a specific model/classifier,
        or data scaler, with updated (or new) parameters.
        These can be supplied by creating another dict within extra_params.
        E.g., extra_params[model_name] = {'model_param' : new_value}
        Where model param is a valid argument for that model, and model_name in
        this case is the str indicator passed to model_type.
        If 'default', use the saved value within self.default_ML_params.
        (default = 'default')

    Returns
    ----------
    list of floats
        The raw score as computed for each fold within each repeat,
        e.g., list will have a length of `n_repeats` * `n_splits`

    Notes
    ----------
    Prints by default the following,

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

    # Perform pre-modeling check
    self._premodel_check(problem_type)

    # Create the set of ML_params from passed args + default args
    ML_params = self._make_ML_params(args=locals())

    # Print the params being used
    self._print_model_params(model_type, ML_params, test=False)

    # Init the Model object with modeling params
    self._init_model(model_type, ML_params)

    # Evaluate the model
    scores = self.Model.Evaluate_Model(self.all_data, self.train_subjects)

    # Compute macro / micro summary of scores
    summary_scores = compute_macro_micro(scores, ML_params['n_repeats'],
                                         ML_params['n_splits'])

    self._print('Macro mean score: ', summary_scores[0])
    self._print('Macro std in score: ', summary_scores[1])
    self._print('Micro mean score: ', summary_scores[2])
    self._print('Micro std in score: ', summary_scores[3])

    # Return the raw scores from each fold
    return scores


def Test(self, model_type, problem_type='default', train_subjects=None,
         test_subjects=None, data_scaler='default', int_cv='default',
         metric='default', class_weight='default', n_jobs='default',
         n_iter='default', random_state='default', return_model=False,
         extra_params='default'):
    '''Class method used to evaluate a specific model / data scaling
    setup on an explicitly defined train and test set.

    model_type : str or list of str,
        Each string refers to a type of model to train.
        If a list of strings is passed then an ensemble model
        will be created over all individual models.
        For a full list of supported options call:
        self.show_model_types(), with optional problem type parameter.

    problem_type : {'regression', 'binary', 'categorical', 'default'}, optional

        - 'regression' : For ML on float or ordinal target data
        - 'binary' : For ML on binary target data
        - 'categorical' : For ML on categorical target data,
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

    data_scaler : str, optional
        `data_scaler` refers to the type of scaling to apply
        to the saved data during model evaluation.
        If 'default', use the saved value within self.default_ML_params.
        For a full list of supported options call:
        self.show_data_scalers()
        (default = 'default')

    int_cv : int or 'default', optional
        The number of internal folds to use during
        model k-fold parameter selection, if the chosen model requires
        parameter selection. A value greater
        then 2 must be passed.
        If 'default', use the saved value within self.default_ML_params.
        (default = 'default')

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

    random_state : int, RandomState instance, None or 'default', optional
        Random state, either as int for a specific seed, or if None then
        the random seed is set by np.random.
        If 'default', use the saved value within self,
        (defined when initing ABCD_ML class) ^,
        Or a different ML params random state is used, if defined when
        calling set default ML params.
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
        E.g., extra_params[model_name] = {'model_param' : new_value}
        Where model param is a valid argument for that model, and model_name in
        this case is the str indicator passed to model_type.
        If 'default', use the saved value within self.default_ML_params.
        (default = 'default')

    Returns
    ----------
    float
        The score as determined by the passed metric/scorer on the
        provided testing set.

    model (if return_model == True)
        The sklearn api trained model object.
    '''

    # Perform pre-modeling check
    self._premodel_check()

    # Create the set of ML_params from passed args + default args
    ML_params = self._make_ML_params(args=locals())

    # Print the params being used
    self._print_model_params(model_type, ML_params, test=True)

    # Init the Model object with modeling params
    self._init_model(model_type, ML_params)

    # If not train subjects or test subjects passed, use class
    if train_subjects is None:
        train_subjects = self.train_subjects
    if test_subjects is None:
        test_subjects = self.test_subjects

    # Train the model w/ selected parameters and test on test subjects
    score = self.Model.Test_Model(self.all_data, train_subjects, test_subjects)

    # Optionally return the model object itself
    if return_model:
        return score, self.Model.model

    return score


def _premodel_check(self, problem_type='default'):
    '''Internal helper function to ensure that self._prepare_data()
    has been called, and to force a train/test split if not already done.
    Will also call set_default_ML_params if not already called.

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
              call self.train_test_split(test_size=0)')

        self.train_test_split(test_size=.25)

    if self.default_ML_params is None:

        self._print('Setting default ML params.')
        self._print('Note, if the following values are not desired,',
                    'call self.set_default_ML_params()')
        self._print('Or just pass values everytime to Evaluate',
                    'or Test, and these default values will be ignored')

        self.set_default_ML_params(problem_type=problem_type)


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


def _print_model_params(self, model_type, ML_params, test=False):

    if test:
        self._print('Running Test with:')
    else:
        self._print('Running Evaluate with:')

    self._print('model_type =', model_type)
    self._print('problem_type =', ML_params['problem_type'])
    self._print('metric =', ML_params['metric'])
    self._print('data_scaler =', ML_params['data_scaler'])

    if not test:
        self._print('n_splits =', ML_params['n_splits'])
        self._print('n_repeats =', ML_params['n_repeats'])

    self._print('int_cv =', ML_params['int_cv'])

    if ML_params['problem_type'] != 'regression':
        self._print('class_weight =', ML_params['class_weight'])

    self._print('n_jobs =', ML_params['n_jobs'])
    self._print('n_iter =', ML_params['n_iter'])
    self._print('random_state =', ML_params['random_state'])
    self._print('extra_params =', ML_params['extra_params'])
    self._print()


def _init_model(self, model_type, ML_params):

    problem_types = {'binary': Binary_Model, 'regression': Regression_Model,
                     'categorical': Categorical_Model}

    assert ML_params['problem_type'] in problem_types, \
        "Invalid problem type!"

    Model = problem_types[ML_params['problem_type']]

    self.Model = Model(model_type, ML_params, self.CV, self.data_keys,
                       self.targets_key, self.targets_encoder, self.verbose)
