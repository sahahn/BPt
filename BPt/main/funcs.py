import warnings
from .input import Model, ModelPipeline, Pipeline, ProblemSpec, CV, Custom
from copy import deepcopy
import numpy as np
import pandas as pd
from ..pipeline.BPtPipelineConstructor import get_pipe
from ..default.options.scorers import process_scorers
from .BPtEvaluator import BPtEvaluator
from sklearn.model_selection import check_cv
from .input_operations import Intersection
from pandas.util._decorators import doc
from .CV import inds_from_names
from ..shared_docs import _shared_docs
from .compare import (_compare_check, CompareDict, _merge_compare, Compare)
from ..default.pipelines import pipelines as default_pipelines
from ..dataset.fake_dataset import FakeDataset

_base_docs = {}

_base_docs[
    "pipeline"
] = """pipeline : :class:`Pipeline`
        | A BPt input class Pipeline to be intialized according
          to the passed dataset and problem_spec.
          This parameter can be either an instance of :class:`Pipeline`,
          :class:`ModelPipeline` or one of the below cases.

        | In the case that a single str is passed, it will assumed
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
        | The :class:`Dataset` in which this function should be evaluated
          in the context of. In other words, the dataset is
          used as the data source for this operation.

        | Arguments within problem_spec can be used to
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


def pipeline_check(pipeline, error_if_compare=True, **extra_params):

    # Make deep copy
    pipe = deepcopy(pipeline)

    # First check here for compare
    if isinstance(pipe, Compare):
        if error_if_compare:
            raise RuntimeError("This function can't accept Compare arguments!")

        new_pipeline = CompareDict()
        for option in pipe.options:
            option.key = 'pipeline'

            # Get the checked version of the pipeline
            # passing along all extra args to each
            checked_pipe = _base_pipeline_check(option.value, **extra_params)

            # Save this option in the dict
            new_pipeline[option] = checked_pipe

        # Return as compare dict
        return new_pipeline

    # Otherwise, base check
    return _base_pipeline_check(pipe, **extra_params)


def _base_pipeline_check(pipe, **extra_params):

    # If passed pipeline is not Pipeline or ModelPipeline
    # then make input as Pipeline around model
    if not isinstance(pipe, Pipeline):

        # Check for if model str first
        if isinstance(pipe, str):

            # Check if a default pipeline
            if pipe in default_pipelines:
                pipe = default_pipelines[pipe].copy()
            else:
                pipe = Model(obj=pipe)

        # In case of passed valid single model, wrap in Pipeline
        if isinstance(pipe, Model):
            pipe = Pipeline(steps=[pipe])
        elif isinstance(pipe, Pipeline):
            pass
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


def problem_spec_check(problem_spec, dataset, error_if_compare=True,
                       **extra_params):

    # If problem spec is already a CompareDict.
    # Return as is
    if isinstance(problem_spec, CompareDict):
        return problem_spec

    # Set ps to copy of problem spec
    ps = deepcopy(problem_spec)

    # Check if problem_spec is left as default
    if ps == 'default':
        ps = ProblemSpec()

    # Check for any override params
    possible_params = ProblemSpec._get_param_names()
    valid_params = {key: extra_params[key] for key in extra_params
                    if key in possible_params}

    # If any override params - reset checked
    if len(valid_params) > 0:
        ps._checked = False

    # If attr checked, then means the passed
    # problem_spec has already been checked and is already
    # a proc'ed and ready copy.
    if hasattr(ps, '_checked') and getattr(ps, '_checked'):
        return ps

    # Set any overlap params
    ps.set_params(**valid_params)

    # Check for any Compare
    ps = _compare_check(ps)

    # If ps is now a dict, it means there was atleast one Compare
    if isinstance(ps, CompareDict):

        if error_if_compare:
            raise RuntimeError("This function can't accept Compare arguments!")

        return CompareDict({key: _base_ps_check(ps[key], dataset)
                            for key in ps})

    # Otherwise perform base check as usual
    return _base_ps_check(ps, dataset)


def _base_ps_check(ps, dataset):

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
def get_estimator(pipeline, dataset='default',
                  problem_spec='default',
                  **extra_params):
    '''Get a sklearn compatible :ref:`estimator<develop>` from a
    :class:`Pipeline`, :class:`Dataset` and :class:`ProblemSpec`.

    Parameters
    -----------
    {pipeline}

    dataset : :class:`Dataset` or 'default', optional
        | The :class:`Dataset` in which this function should be evaluated
          in the context of. In other words, the dataset is
          used as the data source for this operation.

        | If left as default will initialize and use
          an instance of a FakeDataset class, which will
          work fine for initializing pipeline objects
          with scope of 'all', but should be used with caution
          when elements of the pipeline use non 'all' scopes.
          In these cases a warning will be issued.

        | It is typically advised to pass the actual :class:`Dataset`
          of interest here.

        ::

            default = 'default'

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

        data = bp.Dataset([[1, 2, 3], [2, 3, 4], [5, 6, 7]],
                           columns=['1', '2', '3'],
                           targets='3')
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

    # Check for default dataset
    if isinstance(dataset, str) and (dataset == 'default'):
        dataset = FakeDataset()

    # Use initial prep
    estimator_ps = _initial_prep(pipeline, dataset, problem_spec,
                                 error_if_compare=(True, False),
                                 **extra_params)

    # Reduce estimator_ps to just estimator related
    if isinstance(estimator_ps, CompareDict):

        # Return compare dict w/ only estimator, not estimator_ps tuple
        return CompareDict({key: estimator_ps[key][0] for key in estimator_ps})

    # Otherwise, can just return first element of the tuple
    return estimator_ps[0]


def get_pipeline(pipeline, problem_spec, dataset, error_if_compare=True):

    # Run compare check
    pipeline = _compare_check(pipeline)
    if isinstance(pipeline, CompareDict):
        if error_if_compare:
            raise RuntimeError("This function can't accept Compare arguments!")

        return CompareDict({key: _get_pipeline(
            pipeline[key], problem_spec, dataset)[0]
             for key in pipeline})

    # Otherwise, base
    return _get_pipeline(pipeline, problem_spec, dataset)[0]


def _get_pipeline(pipeline, problem_spec, dataset,
                  has_search=False):
    '''Both pipeline and problem_spec should not be CompareDicts.'''

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

        # Case where pipeline is nested as an obj
        if hasattr(obj, 'obj') and isinstance(obj.obj, Pipeline):

            nested_pipe, nested_pipe_params =\
                _get_pipeline(pipeline=obj.obj,
                              problem_spec=nested_ps,
                              dataset=dataset,
                              has_search=has_search)

            # Set obj as nested pipeline
            setattr(obj, 'obj', nested_pipe)

            # Set obj's params as the nested_pipe_params
            setattr(obj, 'params', nested_pipe_params)

        # Case where pipeline is passed as a step
        elif hasattr(obj, 'steps'):
            for i, step in enumerate(obj.steps):
                if isinstance(step, Pipeline):

                    # Call nested get pipeline
                    nested_pipe, nested_pipe_params =\
                        _get_pipeline(pipeline=step,
                                      problem_spec=nested_ps,
                                      dataset=dataset,
                                      has_search=has_search)

                    # If params to pass along, needs to be Model
                    if len(nested_pipe_params) > 0:
                        new_step = Model(nested_pipe,
                                         params=nested_pipe_params)

                    # Otherwise pass as custom
                    else:
                        new_step = Custom(nested_pipe)

                    # Replace step
                    obj.steps[i] = new_step

        # Recursive cases
        if isinstance(obj, list):
            [nested_check(o) for o in obj]
            return

        elif isinstance(obj, tuple):
            (nested_check(o) for o in obj)
            return

        elif isinstance(obj, dict):
            [nested_check(obj[k]) for k in obj]
            return

        elif hasattr(obj, 'get_params'):
            for param in obj.get_params(deep=False):
                try:
                    nested_check(getattr(obj, param))
                except AttributeError:
                    return
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
                try:
                    nested_model_check(getattr(obj, param))
                except AttributeError:
                    return
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

    # If already processed, do nothing, but return True
    if isinstance(param_search, dict):
        return True

    # If a param search, apply ps and dataset + convert to dict
    as_dict = param_search._as_dict(ps=ps)

    # Set new proc'ed
    setattr(object, 'param_search', as_dict)

    return True


def _initial_prep(pipeline, dataset, problem_spec,
                  error_if_compare=True, **extra_params):

    # Get set of all possible params that extra params could refer to
    possible_params = set(ProblemSpec._get_param_names())
    possible_params.update(set(Pipeline._get_param_names()))
    possible_params.update(set(ModelPipeline._get_param_names()))

    # Warn in extra param doesn't map to a possible param
    for key in extra_params:
        if key not in possible_params:
            warnings.warn(f'Passed extra_params key {key} does not appear '
                          'to be valid, and will be skipped.')

    # error if compare can be bool or tuple of bool's
    if isinstance(error_if_compare, tuple):
        eic_ps, eic_pipe = error_if_compare
    else:
        eic_ps, eic_pipe = error_if_compare, error_if_compare

    # Get proc'ed problem spec, w/ possibility that it is
    # returned as a CompareDict.
    ps = problem_spec_check(problem_spec=problem_spec, dataset=dataset,
                            error_if_compare=eic_ps, **extra_params)

    # Want to get pipe, with possibility that returned pipe is CompareDict
    pipe = pipeline_check(pipeline, error_if_compare=eic_pipe,
                          **extra_params)

    # Get the actual pipeline as an estimator
    # where both pipe and ps can be CompareDicts and
    # CompareDicts can still live inside the pipe params

    # Handle if ps if CompareDict and / or pipe is CompareDict
    merged_compare = _merge_compare(pipe=pipe, ps=ps)

    # Base case first
    if not isinstance(merged_compare, CompareDict):

        # Unpack
        c_pipe, c_ps = merged_compare

        # Get the estimator
        estimator = get_pipeline(c_pipe, c_ps,
                                 dataset, error_if_compare=eic_pipe)

        # If estimator is CompareDict, need to add in problem_spec
        # So compare dict stores tuples of estimator problem spec.
        if isinstance(estimator, CompareDict):
            return CompareDict({key: (estimator[key], deepcopy(c_ps))
                               for key in estimator})

        # Otherwise, return as is
        return (estimator, c_ps)

    # CompareDict case
    estimator_ps = CompareDict()
    for key in merged_compare:

        # Get this sub-model
        c_pipe, c_ps = merged_compare[key]
        estimator = get_pipeline(c_pipe, c_ps, dataset,
                                 error_if_compare=eic_pipe)

        # est is either Pipeline or CompareDict
        # If compare dict, add the combination.
        if isinstance(estimator, CompareDict):
            for e_key in estimator:
                estimator_ps[[key, e_key]] =\
                    (deepcopy(estimator[e_key]), deepcopy(c_ps))

        # Otherwise, add as is.
        else:
            estimator_ps[key] = (estimator, c_ps)

    # Return the compare dict
    return estimator_ps


def _sk_prep(pipeline, dataset, problem_spec='default',
             cv=5, **extra_params):

    estimator, ps = _initial_prep(pipeline, dataset, problem_spec,
                                  error_if_compare=True,
                                  **extra_params)

    return _eval_prep(estimator, ps, dataset, cv)


def _eval_prep(estimator, ps, dataset, cv=5):
    '''Internal helper function.'''

    # Check for subjects == 'default' in problem_spec
    if ps.subjects == 'default':

        # If a test set is defined
        if dataset._get_test_subjects() is not None:
            if cv == 'test' or cv == 'Test':
                ps.subjects = 'all'
            else:
                ps.subjects = 'train'

        # Otherwise, subjects = 'all' if no test set defined
        else:
            ps.subjects = 'all'

    # Get X and y - ps should already be init'ed
    X, y = dataset.get_Xy(problem_spec=ps)

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
        n_repeats = bpt_cv.get_n_repeats()

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

        # Set as actual index
        sk_cv = inds_from_names(X.index, sk_cv)

    # Cast explicitly to sklearn style cv from either user
    # passed input or inds
    is_classifier = ps.problem_type != 'regression'

    # Pass y along only if no NaN's in y.
    sk_cv = check_cv(cv=sk_cv, y=None if pd.isnull(y).any() else y,
                     classifier=is_classifier)

    # Make sure random_state and shuffle are set, if avaliable attributes.
    if hasattr(sk_cv, 'random_state'):
        setattr(sk_cv, 'random_state', ps.random_state)
    if hasattr(sk_cv, 'shuffle'):
        setattr(sk_cv, 'shuffle', True)

    # Store n_repeats in sk_cv
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
        _sk_prep(pipeline=pipeline, dataset=dataset,
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
        _sk_prep(pipeline=pipeline, dataset=dataset,
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
             store_preds=True,
             store_estimators=True,
             store_timing=True,
             decode_feat_names=True,
             eval_verbose=1,
             progress_loc=None,
             mute_warnings=False,
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

            default = True

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

    eval_verbose : int, optional
        The requested verbosity of the evaluator.
        0 or greater means just warnings, 1 or greater
        for some info and 2 and greater for even more.

        Set to negative values to mute warnings.
        Specifically, setting to -1 will mute all
        warnings as generated by the call to evaluate,
        and even further, you can set to -2 or lower, which
        will mute all warnings regardless of where they are generated from.
        Note: You can also optionally set the binary flag mute_warnings
        to accomplish the same thing.

        Note: This parameter is called eval_verbose, as the
        pipeline has an argument called `verbose`, which can be used
        to set verbosity for the pipeline.

        Changed default from 0 to 1 in version 2.0.3

        ::

            default = 1

    progress_loc : str or None, optional
        This parameter is not currently implemented.

        ::

            default = None

    mute_warnings : bool, optional
        Mute any warning regardless of where they are generated from.
        This can also be done by setting eval_verbose to -2 or lower.

        ::

            default = False

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
    Compare : Input class for specifying comparisons between
        parameter options.

    Notes
    ------
    | This function can accept within the `pipeline` and `problem_spec`
        parameters the special input :class:`Compare` class. This
        option is designed for explicitly running evaluate multiple
        times under different configurations.

    | This function supports predictions on an underlying
        target with missing values. It does this by automatically
        moving any data points with a NaN in the target to the validation
        set (keeping in mind this is done after the fold is computed by CV,
        so final size may vary). While subjects with missing values will
        obviously not contribute to the validation score,
        as long as `store_preds`
        is set to True, predictions will still be made for these subjects.
    '''

    params = {'cv': cv,
              'decode_feat_names': decode_feat_names,
              'progress_bar': progress_bar,
              'store_preds': store_preds,
              'store_estimators': store_estimators,
              'store_timing': store_timing,
              'eval_verbose': eval_verbose,
              'progress_loc': progress_loc,
              'mute_warnings': mute_warnings
              }

    # Get the estimator and problem spec,
    # w/ option for returned
    # value as a CompareDict
    estimator_ps =\
        _initial_prep(pipeline, dataset, problem_spec,
                      error_if_compare=False, **extra_params)

    # Base case
    if not isinstance(estimator_ps, CompareDict):
        estimator, ps = estimator_ps
        return _evaluate(estimator=estimator, dataset=dataset, ps=ps, **params)

    # Set at start to number of runs
    if progress_bar:
        compare_bars = len(estimator_ps)
    else:
        compare_bars = None

    # Compare dict case
    evaluators = CompareDict()
    for key in estimator_ps:

        if eval_verbose >= 1:
            print('Running Compare:', key, flush=True)

        # Unpack
        c_estimator, c_ps = estimator_ps[key]

        # Evaluate this option
        evaluator = _evaluate(estimator=c_estimator,
                              dataset=dataset, ps=c_ps,
                              compare_bars=compare_bars,
                              **params)

        # Update compare bars to be the the compare bars
        # set by last run
        compare_bars = evaluator.compare_bars

        # Then reset stored attribute,
        # so the evaluator object can be pickled
        evaluator.compare_bars = None

        # Add to compare dict
        evaluators[key] = evaluator

    # Close compare bars
    if compare_bars is not None:
        for bar in compare_bars:
            bar.n = bar.total
            bar.refresh()
            bar.close()

    return evaluators


def _evaluate(estimator, dataset, ps, cv,
              decode_feat_names, compare_bars=None,
              **verbose_args):

    estimator, X, y, ps, sk_cv =\
        _eval_prep(estimator, ps, dataset, cv)

    # Check decode feat_names arg, if True, pass along encoders
    encoders = None
    if decode_feat_names:
        encoders = dataset._get_encoders()

    # Init evaluator
    evaluator = BPtEvaluator(estimator=estimator, ps=ps,
                             encoders=encoders,
                             compare_bars=compare_bars,
                             **verbose_args)

    # Call eval on the evaluator
    evaluator._eval(X, y, sk_cv)

    # Return the BPtEvaluator object
    return evaluator
