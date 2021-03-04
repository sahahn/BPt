from .input import Model, Pipeline, ProblemSpec, CV
from copy import deepcopy
import numpy as np
import pandas as pd
from ..pipeline.BPtPipelineConstructor import get_pipe
from ..default.options.scorers import process_scorers
from .BPtEvaluator import BPtEvaluator
from sklearn.model_selection import check_cv
from .input_operations import Intersection
from pandas.util._decorators import doc
from ..shared_docs import _shared_docs


_base_docs = {}

_base_docs[
    "pipeline"
] = """pipeline : :class:`Pipeline`
        A BPt input class Pipeline to be intialized according
        to the passed dataset and problem_spec.

        This parameter can be either an instance of :class:`Pipeline`,
        :class:`ModelPipeline` or one of the below cases.

        In the case that a single str is passed, it will assumed
        to be a model indicator str and the pipeline used will be:

        ::

            pipeline = Pipeline(Model(pipeline))

        Likewise, if just a Model passed, then the input will be
        cast as:

        ::

            pipeline = Pipeline(pipeline)

    """

_base_docs[
    "dataset"
] = """dataset : :class:`Dataset`
        The Dataset in function should be evaluated in the context of.
        The dataset is as the data source for this operation.

        Arguments within problem_spec can be used to
        select just subsets of data. For example parameter
        scope can be used to select only some columns or
        parameter subjects to select a subset of subjects.
    """

_base_docs["problem_spec"] = _shared_docs['problem_spec']

_base_docs[
    "extra_params"
] = """extra_params : problem_spec or pipeline params, optional
        You may pass as extra arguments to this function any pipeline
        or problem_spec argument as python kwargs style value pairs.

        For example:

        ::

            target=1

        Would override the value of the target parameter
        in the passed problem_spec. Or for example:

        ::

            model=Model('ridge')

    """

_eval_docs = _base_docs.copy()
_eval_docs[
    "cv"
] = """cv : :class:`CV` or :ref:`sklearn CV <cross_validation>`, optional
        This parameter controls what type of cross-validation
        splitting strategy is used. You may pass
        a number of options here.

        - An instance of :class:`CV` representing a
          custom strategy as defined by the BPt style CV.

        - The custom str 'test', which specifies that the
          whole train set should be used to train the pipeline
          and the full test set used to validate it (assuming that
          a train test split has been defined in the underlying dataset)

        - Any valid scikit-learn style option:
          Which include an int to specify the number of folds in a
          (Stratified) KFold, a sklearn CV splitter or an iterable
          yielding (train, test) splits as arrays of indices.

        ::

            default = 5
"""


def pipeline_check(pipeline, **extra_params):

    # Make deep copy
    pipe = deepcopy(pipeline)

    # If passed pipeline is not Pipeline or ModelPipeline
    # then make input as Pipeline around model
    if not isinstance(pipe, Pipeline):

        # Check for if model str first
        if isinstance(pipe, str):
            pipe = Model(obj=pipe)

        # In case of passed valid single model, wrap in Pipeline
        if isinstance(pipe, Model):
            pipe = Pipeline(steps=[pipe])
        else:
            raise RuntimeError('pipeline must be a Pipeline',
                               ' model str or Model-like')

    # Set any overlapping extra params
    possible_params = pipe._get_param_names()
    valid_params = {key: extra_params[key] for key in extra_params
                    if key in possible_params}
    pipe.set_params(**valid_params)

    # Internal class checks
    pipe._proc_checks()

    return pipe


def _problem_spec_target_check(ps, dataset):

    # Get target col from dataset, this also
    # makes sure scopes and roles are updated.
    targets = dataset.get_cols('target')

    # Check for no targets
    if len(targets) == 0:
        raise IndexError('The passed Dataset is not valid. '
                         'It must have atleast one target defined.')

    # Update target, if passed as int, treat as index
    if isinstance(ps.target, int):
        try:
            ps.target = targets[ps.target]
        except IndexError:
            raise IndexError('target index: ' + repr(ps.target) +
                             ' is out of range, only ' + repr(len(targets)) +
                             ' targets are defined.')

    # If not int, then raise error if invalid or keep as is
    if ps.target not in targets:
        raise IndexError('Passed target: ' + repr(ps.target) + ' does ' +
                         'not have role target or is not loaded.')


def problem_spec_check(problem_spec, dataset, **extra_params):

    # If attr checked, then means the passed
    # problem_spec has already been checked and is already
    # a proc'ed and ready copy.
    if hasattr(problem_spec, '_checked') and getattr(problem_spec, '_checked'):
        return problem_spec

    # Check if problem_spec is left as default
    if problem_spec == 'default':
        problem_spec = ProblemSpec()

    # Set ps to copy of problem spec and init
    ps = deepcopy(problem_spec)

    # Apply any passed valid extra params
    possible_params = ProblemSpec._get_param_names()
    valid_params = {key: extra_params[key] for key in extra_params
                    if key in possible_params}
    ps.set_params(**valid_params)

    # Proc params
    ps._proc_checks()

    # Get target col from dataset, sets target
    # in ps in place.
    _problem_spec_target_check(ps, dataset)

    # Replace problem_spec type w/ correct if passed short hands
    pt = ps.problem_type

    if pt == 'default':
        pt = dataset._get_problem_type(ps.target)
    elif pt == 'b':
        pt = 'binary'
    elif pt == 'c' or pt == 'multiclass':
        pt = 'categorical'
    elif pt == 'f' or pt == 'float':
        pt = 'regression'

    ps.problem_type = pt

    # Convert to scorer obj
    ps.scorer =\
        process_scorers(ps.scorer,
                        problem_type=ps.problem_type)

    # Set checked flag in ps
    setattr(ps, '_checked', True)

    return ps


@doc(**_base_docs)
def get_estimator(pipeline, dataset,
                  problem_spec='default',
                  **extra_params):
    '''Get a sklearn compatible :ref:`estimator<develop>` from a
    :class:`Pipeline`, :class:`Dataset` and :class:`ProblemSpec`.

    Parameters
    -----------
    {pipeline}

    {dataset}

    {problem_spec}

    {extra_params}

    Returns
    --------
    estimator : :ref:`sklearn Estimator <develop>`
        The returned object is a sklearn-compatible estimator.
        It will be either of type BPtPipeline or a BPtPipeline
        wrapped in a search CV object.

    Examples
    ----------
    This example shows how this function can be
    used with Dataset method
    :func:`get_Xy <Dataset.get_Xy>`.

    First we will setup a dataset and a pipeline.

    .. ipython:: python

        import BPt as bp

        data = bp.Dataset()
        data['1'] = [1, 2, 3]
        data['2'] = [2, 3, 4]
        data['3'] = [5, 6, 7]
        data.set_role('3', 'target', inplace=True)
        data

        pipeline = bp.Pipeline([bp.Scaler('standard'), bp.Model('linear')])
        pipeline

    Next we can use get_estimator and
    also convert the dataset into a traditional sklearn-style
    X, y input. Note that we are using just the default values
    for problem spec.

    .. ipython:: python

        X, y = data.get_Xy()
        X.shape, y.shape

        estimator = bp.get_estimator(pipeline, data)

    Now we can use this estimator as we would any other sklearn
    style estimator.

    .. ipython:: python

        estimator.fit(X, y)
        estimator.score(X, y)

    '''

    # @TODO add verbose option?
    dataset._check_sr()

    # Check passed input - note: returns deep copies
    pipe = pipeline_check(pipeline, **extra_params)
    ps = problem_spec_check(problem_spec, dataset, **extra_params)

    # Get the actual pipeline
    model, _ = _get_pipeline(pipe, ps, dataset)

    return model


def _get_pipeline(pipeline, problem_spec, dataset,
                  has_search=False):

    # If has search is False, means this is the top level
    # or the top level didn't have a search
    if not has_search:
        has_search = True
        if pipeline.param_search is None:
            has_search = False

    # If either this set of pipeline params or the parent
    # params had search params, then a copy of problem spec with n_jobs set
    # to 1 should be passed to children get pipelines,
    nested_ps = problem_spec
    if has_search:
        nested_ps = deepcopy(problem_spec)
        nested_ps.n_jobs = 1

    # Define a nested check, that iterates through searching for
    # nested model pipeline's
    def nested_check(obj):

        if hasattr(obj, 'obj') and isinstance(obj.obj, Pipeline):

            nested_pipe, nested_pipe_params =\
                _get_pipeline(pipeline=obj.obj,
                              problem_spec=nested_ps,
                              has_search=has_search)

            # Set obj as nested pipeline
            setattr(obj, 'obj', nested_pipe)

            # Set obj's params as the nested_pipe_params
            setattr(obj, 'params', nested_pipe_params)
            return

        if isinstance(obj, list):
            [nested_check(o) for o in obj]
            return

        elif isinstance(obj, tuple):
            (nested_check(o) for o in obj)

        elif isinstance(obj, dict):
            [nested_check(obj[k]) for k in obj]
            return

        if hasattr(obj, 'get_params'):
            for param in obj.get_params(deep=False):
                nested_check(getattr(obj, param))
            return

    # Run nested check on passed pipeline
    nested_check(pipeline)

    # Preproc model
    pipeline =\
        _preproc_pipeline(pipeline, problem_spec, dataset)

    # Return the pipeline
    return get_pipe(pipeline_params=pipeline,
                    problem_spec=problem_spec,
                    dataset=dataset)


def _preproc_pipeline(pipe, ps, dataset):

    # Check imputers default case
    # Check if any NaN in data
    data_cols = dataset._get_cols(scope='data', limit_to=ps.scope)
    is_na = dataset[data_cols].isna().any().any()
    pipe._check_imputers(is_na)

    # Pre-proc param search
    has_param_search = _preproc_param_search(pipe, ps)

    # If there is a param search set for the Model Pipeline,
    # set n_jobs, the value to pass in the nested check, to 1
    nested_ps = deepcopy(ps)
    if has_param_search:
        nested_ps.n_jobs = 1

    def nested_model_check(obj):

        # Check for Model or Ensemble
        if isinstance(obj, Model):
            _preproc_param_search(obj, nested_ps)

        if isinstance(obj, list):
            [nested_model_check(o) for o in obj]
            return

        elif isinstance(obj, tuple):
            (nested_model_check(o) for o in obj)
            return

        elif isinstance(obj, dict):
            [nested_model_check(obj[k]) for k in obj]

        elif hasattr(obj, 'get_params'):
            for param in obj.get_params(deep=False):
                nested_model_check(getattr(obj, param))
            return
        return

    # Run nested check
    nested_model_check(pipe)

    # Run nested check for any CV input param objects
    nested_cv_check(pipe, dataset)

    return pipe


def nested_cv_check(obj, dataset):

    def _nested_cv_check(obj):

        # If has cv object, try applying dataset
        # can be either in object or in dict

        # If Object, check if any of the parameters are CV
        if hasattr(obj, 'get_params'):
            for param in obj.get_params(deep=False):
                val = getattr(obj, param)

                # If CV proc
                if isinstance(val, CV):
                    setattr(obj, param, val._apply_dataset(dataset))

                # If not, nested check
                else:
                    _nested_cv_check(val)

        # If Dict, check all valyes
        elif isinstance(obj, dict):
            for k in obj:
                val = obj[k]

                # If CV proc
                if isinstance(val, CV):
                    obj[k] = val._apply_dataset(dataset)

                # If not, nested check
                else:
                    _nested_cv_check(val)

        elif isinstance(obj, (list, tuple, set)):
            [_nested_cv_check(o) for o in obj]

    # Run nested check for any CV input param objects
    _nested_cv_check(obj)


def _preproc_param_search(object, ps):

    # Get param search
    param_search = getattr(object, 'param_search')

    # If None, return False and do nothing
    if param_search is None:
        return False

    # If a param search, apply ps and dataset + convert to dict
    as_dict = param_search._as_dict(ps=ps)

    # Set new proc'ed
    setattr(object, 'param_search', as_dict)

    return True


def _eval_prep(pipeline, dataset, problem_spec='default',
               cv=5, **extra_params):
    '''Internal helper function return the different pieces
    needed by sklearn functions'''

    # Get proc'ed problem spec, then passed proc'ed version
    ps = problem_spec_check(problem_spec=problem_spec, dataset=dataset,
                            **extra_params)

    # Get estimator
    estimator = get_estimator(pipeline=pipeline, dataset=dataset,
                              problem_spec=ps, **extra_params)

    # Get X and y
    X, y = dataset.get_Xy(problem_spec=ps, **extra_params)

    # Save if has n_repeats
    n_repeats = 1
    if hasattr(cv, 'n_repeats'):
        n_repeats = n_repeats

    # If Passed CV class, then need to convert to sklearn compat
    # otherwise, assume that it is sklearn compat and pass as is.
    # @TODO maybe convert to BPtCV then back?
    # to add random state and what not ?
    sk_cv = cv
    if isinstance(cv, CV):

        # Convert from CV class to BPtCV by applying dataset
        bpt_cv = cv._apply_dataset(dataset)

        # Save n_repeats
        n_repeats = bpt_cv.n_repeats

        # Convert from BPtCV to sklearn compat, i.e., just raw index
        sk_cv = bpt_cv.get_cv(X.index,
                              random_state=ps.random_state,
                              return_index=True)

    # Check for passed test case
    elif cv == 'test' or cv == 'Test':

        # Get train and test subjs as intersection of train and ps
        train = dataset.get_subjects(Intersection(['train', ps.subjects]),
                                     return_as='flat index')
        test = dataset.get_subjects(Intersection(['test', ps.subjects]),
                                    return_as='flat index')

        # Set as single iterable.
        sk_cv = [(train, test)]

    # Cast explicitly to sklearn style cv from either user
    # passed input or inds
    is_classifier = ps.problem_type != 'regression'
    sk_cv = check_cv(cv=sk_cv, y=y, classifier=is_classifier)
    setattr(sk_cv, 'n_repeats', n_repeats)

    return estimator, X, y, ps, sk_cv


def _sk_check_y(y):

    if pd.isnull(y).any():
        raise RuntimeError('NaNs are not supported with '
                           'the sklearn compatable evaluate functions. '
                           'See function evaluate instead, which supports '
                           'missing target values, or drop subjects with '
                           'missing target values from the dataset.')


@doc(**_eval_docs)
def cross_val_score(pipeline, dataset,
                    problem_spec='default',
                    cv=5, sk_n_jobs=1, verbose=0,
                    fit_params=None,
                    error_score=np.nan,
                    **extra_params):
    '''This function is a BPt compatible wrapper around
    :func:`sklearn.model_selection.cross_val_score`

    Parameters
    ----------
    {pipeline}

    {dataset}

    {problem_spec}

    {cv}

    sk_n_jobs : int, optional
        The number of jobs as passed to the base sklearn
        cross_val_score. Typically this value
        should be kept at 0, and n_jobs as defined
        through the passed problem_spec used
        to define the number of jobs.

        For added flexibility though, this
        parameter can be used either with
        the n_jobs parameter in problem_spec
        or instead of.

        ::

            default = 1

    verbose : int, optional
        The verbosity level as passed to the sklearn function.

        ::

            default = 0

    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.

        ::

            default = None


    error_score : 'raise' or numeric, optional
        Base sklearn func parameter.

        ::

            default = np.nan

    {extra_params}

    See Also
    ---------
    cross_validate : To run on multiple metrics and other options.
    evaluate : The BPt style similar function with extra options.

    '''

    # Get sk compat pieces
    estimator, X, y, ps, sk_cv =\
        _eval_prep(pipeline=pipeline, dataset=dataset,
                   problem_spec=problem_spec, cv=cv, **extra_params)

    # Make sure no NaN's in y
    _sk_check_y(y)

    # Just take first scorer if dict
    if isinstance(ps.scorer, dict):
        ps.scorer = ps.scorer[list(ps.scorer)[0]]

    from sklearn.model_selection import cross_val_score
    return cross_val_score(estimator=estimator, X=X, y=y, scoring=ps.scorer,
                           cv=sk_cv, n_jobs=sk_n_jobs, verbose=verbose,
                           fit_params=fit_params, error_score=error_score)


@doc(**_eval_docs)
def cross_validate(pipeline, dataset,
                   problem_spec='default',
                   cv=5, sk_n_jobs=1, verbose=0,
                   fit_params=None,
                   return_train_score=False,
                   return_estimator=False,
                   error_score=np.nan,
                   **extra_params):
    '''This function is a BPt compatible wrapper around
    :func:`sklearn.model_selection.cross_validate`

    Parameters
    ----------
    {pipeline}

    {dataset}

    {problem_spec}

    {cv}

    sk_n_jobs : int, optional
        The number of jobs as passed to the base sklearn
        cross_val_score. Typically this value
        should be kept at 0, and n_jobs as defined
        through the passed problem_spec used
        to define the number of jobs.

        For added flexibility though, this
        parameter can be used either with
        the n_jobs parameter in problem_spec
        or instead of.

        ::

            default = 1

    verbose : int, optional
        The verbosity level as passed to the sklearn function.

        ::

            default = 0

    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.

        ::

            default = None

    return_train_score : bool, optional
        Whether to include train scores.

        ::

            default = False

    return_estimator : bool, optional
        Whether to return the estimators fitted on each split.

        ::

            default = False


    error_score : 'raise' or numeric, optional
        Base sklearn func parameter.

        ::

            default = np.nan

    {extra_params}

    See Also
    ---------
    cross_val_score : Simplified version of this function.
    evaluate : The BPt style similar function with extra options.

    '''

    # Get sk compat pieces
    estimator, X, y, ps, sk_cv =\
        _eval_prep(pipeline=pipeline, dataset=dataset,
                   problem_spec=problem_spec, cv=cv, **extra_params)

    # Make sure no NaN's in y.
    _sk_check_y(y)

    from sklearn.model_selection import cross_validate
    return cross_validate(estimator=estimator, X=X, y=y, scoring=ps.scorer,
                          cv=sk_cv, n_jobs=sk_n_jobs, verbose=verbose,
                          fit_params=fit_params,
                          return_train_score=return_train_score,
                          return_estimator=return_estimator,
                          error_score=error_score)


@doc(**_eval_docs)
def evaluate(pipeline, dataset,
             problem_spec='default',
             cv=5,
             progress_bar=True,
             store_preds=False,
             store_estimators=True,
             store_timing=True,
             decode_feat_names=True,
             progress_loc=None,
             **extra_params):
    ''' This method is used to evaluate a model pipeline
    with cross-validation.

    Parameters
    -----------
    {pipeline}

    {dataset}

    {problem_spec}

    {cv}

    progress_bar : bool, optional
        If True, then an progress
        bar will be displaying showing fit
        progress across
        evaluation splits.

        ::

            default = True

    store_preds : bool, optional
        If set to True, the returned :class:`BPtEvaluator` will store
        the saved predictions under :data:`BPt.BPtEvaluator.preds`.
        This includes a saved copy of the true target values as well.

        If False, the `preds` parameter will be empty and it
        will not be possible to use some related functions.

        ::

            default = False

    store_estimators : bool, optional
        If True, then the returned :class:`BPtEvaluator`
        will store the fitted estimators from evaluation
        under :data:`BPt.BPtEvaluator.estimators`.

        If False, the `estimators` parameter will be empty,
        and it will not be possible to access measures of
        feature importance as well.


        ::

            default = True

    store_timing : bool, optional
        If True, then the returned :class:`BPtEvaluator`
        will store the time it took to fit and score the pipeline
        under :data:`BPt.BPtEvaluator.timing`.

        ::

            default = True

    decode_feat_names : bool, optional
        If True, then the :data:`BPt.BPtEvaluator.feat_names`
        as computed during evaluation will try to use the original
        values as first loaded to inform their naming. Note that
        this is only relevant assuming that :class:`Dataset` was
        used to encode one or more columns in the first place.

        ::

            default = True

    progress_loc : str or None, optional
        This parameter is not currently implemented.

        ::

            default = None

    {extra_params}

    Returns
    ---------
    evaluator : :class:`BPtEvaluator`
        Returns an instance of the :class:`BPtEvaluator`
        class. This object stores a wealth of information,
        including the scores from this evaluation as well
        as other utilities including functions for calculating
        feature importances from trained models.

    See Also
    ---------
    cross_val_score : Similar sklearn style function.
    cross_validate : Similar sklearn style function.

    Notes
    ------
    This function supports predictions on an underlying
    target with missing values. It does this by automatically
    moving any data points with a NaN in the target to the validation
    set (keeping in mind this is done after the fold is computed by CV,
    so final size may vary). While subjects with missing values will
    obviously not contribute to the validation score, as long as `store_preds`
    is set to True, predictions will still be made for these subjects.
    '''

    # Base process each component
    estimator, X, y, ps, sk_cv =\
        _eval_prep(pipeline=pipeline, dataset=dataset,
                   problem_spec=problem_spec, cv=cv, **extra_params)

    # Check decode feat_names arg
    encoders = None
    if decode_feat_names:
        encoders = dataset.encoders

    # Init evaluator
    evaluator = BPtEvaluator(estimator=estimator, ps=ps,
                             encoders=encoders,
                             progress_bar=progress_bar,
                             store_preds=store_preds,
                             store_estimators=store_estimators,
                             store_timing=store_timing,
                             progress_loc=progress_loc)

    # Call evaluate on the evaluator
    evaluator._evaluate(X, y, sk_cv)

    # Return the BPtEvaluator object
    return evaluator
