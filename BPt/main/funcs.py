from .input import Model, ModelPipeline, ProblemSpec, Ensemble, CV
from copy import deepcopy
import numpy as np
from ..pipeline.BPtPipelineConstructor import get_pipe
from ..default.options.scorers import process_scorers
from .BPtEvaluator import BPtEvaluator
from sklearn.model_selection import check_cv


def model_pipeline_check(model_pipeline, **extra_params):

    # Make deep copy
    pipe = deepcopy(model_pipeline)

    # Add checks on ModelPipeline
    if not isinstance(pipe, ModelPipeline):

        # Check for if model str first
        if isinstance(pipe, str):
            pipe = Model(obj=pipe)

        # In case of passed valid single model, wrap in ModelPipeline
        # with non-default scalers of None
        if isinstance(pipe, Model):
            pipe = ModelPipeline(model=pipe, scalers=None)
        else:
            raise RuntimeError('model_pipeline must be a Pipeline',
                               ' model str or Model-like')

    # Set any overlapping extra params
    possible_params = ModelPipeline._get_param_names()
    valid_params = {key: extra_params[key] for key in extra_params
                    if key in possible_params}
    pipe.set_params(**valid_params)

    # Internal class checks
    pipe._proc_checks()

    return pipe


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

    # Get target col from dataset
    targets = dataset.get_cols('target')

    # Update target, if passed as int
    # otherwise assume correct
    if isinstance(ps.target, int):
        try:
            ps.target = targets[ps.target]
        except IndexError:
            raise IndexError('target index: ' + repr(ps.target) +
                             ' is out of range, only ' + repr(len(targets)) +
                             ' targets are defined.')
    else:
        if ps.target not in targets:
            raise IndexError('Passed target: ' + repr(ps.target) + ' does ' +
                             'not have role target or is not loaded.')

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
    ps.scorer = process_scorers(ps.scorer, problem_type=ps.problem_type)

    # Set checked flag in ps
    setattr(ps, '_checked', True)

    return ps


def get_estimator(model_pipeline, dataset,
                  problem_spec='default', **extra_params):
    '''Get a sklearn compatible estimator from a :class:`ModelPipeline`,
    :class:`Dataset` and :class:`ProblemSpec`.

    This function can be used together with Dataset method
    :func:`get_Xy <Dataset.get_Xy>` and it's variants.

    Parameters
    -----------
    model_pipeline : :class:`ModelPipeline`
        A BPt input class ModelPipeline to be intialized according
        to the passed Dataset and ProblemSpec.

    dataset : :class:`Dataset`
        The Dataset in which the pipeline should be initialized
        according to. For example, pipeline's can include Scopes,
        these need a reference Dataset.

    problem_spec : :class:`ProblemSpec` or 'default', optional
        `problem_spec` accepts an instance of the
        params class :class:`ProblemSpec`.
        This object is essentially a wrapper around commonly used
        parameters needs to define the context
        the model pipeline should be evaluated in.
        It includes parameters like problem_type, scorer, n_jobs,
        random_state, etc...
        See :class:`ProblemSpec` for more information
        and for how to create an instance of this object.

        If left as 'default', then will initialize a
        ProblemSpec with default params.

        ::

            default = 'default'

    extra_params : problem_spec or model_pipeline params, optional
        You may pass as extra arguments to this function any model_pipeline
        or problem_spec argument value pairs. For example,

        ::

            get_estimator(model_pipeline, dataset, problem_spec,
                          model=Model('ridge'), target='1')

        Would get an estimator according to the passed model_pipeline,
        but with model=Model('ridge') and the passed problem_spec
        except with target='1' instead of the original values.

    Returns
    --------
    estimator : sklearn Estimator
        The returned object is a sklearn-compatible estimator.
        It will be either of type BPtPipeline or a BPtPipeline
        wrapped in a search CV object.
    '''

    # @TODO add verbose option?
    dataset._check_sr()

    # Check passed input - note: returns deep copies
    pipe = model_pipeline_check(model_pipeline, **extra_params)
    ps = problem_spec_check(problem_spec, dataset, **extra_params)

    # Get the actual pipeline
    model, _ = _get_pipeline(pipe, ps, dataset)

    return model


def _get_pipeline(model_pipeline, problem_spec, dataset,
                  has_search=False):

    # If has search is False, means this is the top level
    # or the top level didnt have a search
    if not has_search:
        has_search = True
        if model_pipeline.param_search is None:
            has_search = False

    # If either this set of model_pipeline params or the parent
    # params had search params, then a copy of problem spec with n_jobs set
    # to 1 should be passed to children get pipelines,
    nested_ps = problem_spec
    if has_search:
        nested_ps = deepcopy(problem_spec)
        nested_ps.n_jobs = 1

    # Define a nested check, that iterates through searching for
    # nested model pipeline's
    def nested_check(obj):

        if hasattr(obj, 'obj') and isinstance(obj.obj, ModelPipeline):

            nested_pipe, nested_pipe_params =\
                _get_pipeline(model_pipeline=obj.obj,
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

    # Run nested check on passed model_pipeline
    nested_check(model_pipeline)

    # Preproc model
    model_pipeline =\
        _preproc_model_pipeline(model_pipeline, problem_spec, dataset)

    # Return the pipeline
    return get_pipe(pipeline_params=model_pipeline,
                    problem_spec=problem_spec,
                    dataset=dataset)


def _preproc_model_pipeline(pipe, ps, dataset):

    # Check imputers default case
    # Check if any NaN in data
    data_cols = dataset._get_cols(scope='data', limit_to=ps.scope)
    is_na = dataset[data_cols].isna().any().any()
    pipe.check_imputers(is_na)

    # Pre-proc param search
    has_param_search = _preproc_param_search(pipe, ps)

    # If there is a param search set for the Model Pipeline,
    # set n_jobs, the value to pass in the nested check, to 1
    nested_ps = deepcopy(ps)
    if has_param_search:
        nested_ps.n_jobs = 1

    def nested_model_check(obj):

        # Check for Model or Ensemble
        if isinstance(obj, Model) or isinstance(obj, Ensemble):
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
                    setattr(obj, param, val.apply_dataset(dataset))

                # If not, nested check
                else:
                    _nested_cv_check(val)

        # If Dict, check all valyes
        elif isinstance(obj, dict):
            for k in obj:
                val = obj[k]

                # If CV proc
                if isinstance(val, CV):
                    obj[k] = val.apply_dataset(dataset)

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
    as_dict = param_search.as_dict(ps=ps)

    # Set new proc'ed
    setattr(object, 'param_search', as_dict)

    return True


def _sk_prep(model_pipeline, dataset, problem_spec='default',
             cv=5, **extra_params):
    '''Internal helper function return the different pieces
    needed by sklearn functions'''

    # Get proc'ed problem spec, then passed proced version
    ps = problem_spec_check(problem_spec=problem_spec, dataset=dataset,
                            **extra_params)

    # Get estimator
    estimator = get_estimator(model_pipeline=model_pipeline, dataset=dataset,
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
        bpt_cv = cv.apply_dataset(dataset)

        # Save n_repeats
        n_repeats = bpt_cv.n_repeats

        # Convert from BPtCV to sklearn compat, i.e., just raw index
        sk_cv = bpt_cv.get_cv(X.index,
                              random_state=ps.random_state,
                              return_index=True)

    # Cast explicitly to sklean style cv from either user
    # passed input or inds
    is_classifier = ps.problem_type != 'regression'
    sk_cv = check_cv(cv=sk_cv, y=y, classifier=is_classifier)
    setattr(sk_cv, 'n_repeats', n_repeats)

    return estimator, X, y, ps, sk_cv


def cross_val_score(model_pipeline, dataset,
                    problem_spec='default',
                    cv=5, sk_n_jobs=1, verbose=0,
                    fit_params=None,
                    error_score=np.nan,
                    **extra_params):
    '''This function is a BPt compatible wrapper around the scikit-learn
    function cross_val_score,
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html


    Parameters
    ----------
    model_pipeline : :class:`ModelPipeline`
        A BPt input class ModelPipeline to be intialized according
        to the passed Dataset and ProblemSpec, and then evaluated.

    dataset : :class:`Dataset`
        The Dataset in which the pipeline should be initialized
        according to, and data drawn from. See parameter
        `subjects` to use only a subset of the columns or subjects
        in this call to cross_val_score.

    problem_spec : :class:`ProblemSpec` or 'default', optional
        `problem_spec` accepts an instance of the
        params class :class:`ProblemSpec`.
        This object is essentially a wrapper around commonly used
        parameters needs to define the context
        the model pipeline should be evaluated in.
        It includes parameters like problem_type, scorer, n_jobs,
        random_state, etc...
        See :class:`ProblemSpec` for more information
        and for how to create an instance of this object.

        If left as 'default', then will initialize a
        ProblemSpec with default params.

        Warning: the parameter weight_scorer in problem_spec
        is ignored when used with cross_val_score.

        ::

            default = 'default'

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

    extra_params : problem_spec or model_pipeline params, optional
        You may pass as extra arguments to this function any model_pipeline
        or problem_spec argument value pairs. For example,

        ::

            cross_val_score(model_pipeline, dataset,
                            model=Model('ridge'), target='1')

        Would test an estimator according to the passed model_pipeline,
        but with model=Model('ridge') and the passed problem_spec
        except with target='1' instead of the original values.
    '''

    # Get sk compat pieces
    estimator, X, y, ps, sk_cv =\
        _sk_prep(model_pipeline=model_pipeline, dataset=dataset,
                 problem_spec=problem_spec, cv=cv, **extra_params)

    # Just take first scorer if dict
    if isinstance(ps.scorer, dict):
        ps.scorer = ps.scorer[list(ps.scorer)[0]]

    from sklearn.model_selection import cross_val_score
    return cross_val_score(estimator=estimator, X=X, y=y, scoring=ps.scorer,
                           cv=sk_cv, n_jobs=sk_n_jobs, verbose=verbose,
                           fit_params=fit_params, error_score=error_score)


def cross_validate(model_pipeline, dataset,
                   problem_spec='default',
                   cv=5, sk_n_jobs=1, verbose=0,
                   fit_params=None,
                   return_train_score=False,
                   return_estimator=False,
                   error_score=np.nan,
                   **extra_params):
    '''This function is a BPt compatible wrapper around the scikit-learn
    function cross_validate,
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate


    Parameters
    ----------
    model_pipeline : :class:`ModelPipeline`
        A BPt input class ModelPipeline to be intialized according
        to the passed Dataset and ProblemSpec, and then evaluated.

    dataset : :class:`Dataset`
        The Dataset in which the pipeline should be initialized
        according to, and data drawn from. See parameter
        `subjects` to use only a subset of the columns or subjects
        in this call to cross_val_score.

    problem_spec : :class:`ProblemSpec` or 'default', optional
        `problem_spec` accepts an instance of the
        params class :class:`ProblemSpec`.
        This object is essentially a wrapper around commonly used
        parameters needs to define the context
        the model pipeline should be evaluated in.
        It includes parameters like problem_type, scorer, n_jobs,
        random_state, etc...
        See :class:`ProblemSpec` for more information
        and for how to create an instance of this object.

        If left as 'default', then will initialize a
        ProblemSpec with default params.

        Warning: the parameter weight_scorer in problem_spec
        is ignored when used with cross_val_score.

        ::

            default = 'default'

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

    extra_params : problem_spec or model_pipeline params, optional
        You may pass as extra arguments to this function any model_pipeline
        or problem_spec argument value pairs. For example,

        ::

            cross_val_score(model_pipeline, dataset,
                            model=Model('ridge'), target='1')

        Would test an estimator according to the passed model_pipeline,
        but with model=Model('ridge') and the passed problem_spec
        except with target='1' instead of the original values.
    '''

    # Get sk compat pieces
    estimator, X, y, ps, sk_cv =\
        _sk_prep(model_pipeline=model_pipeline, dataset=dataset,
                 problem_spec=problem_spec, cv=cv, **extra_params)

    from sklearn.model_selection import cross_validate
    return cross_validate(estimator=estimator, X=X, y=y, scoring=ps.scorer,
                          cv=sk_cv, n_jobs=sk_n_jobs, verbose=verbose,
                          fit_params=fit_params,
                          return_train_score=return_train_score,
                          return_estimator=return_estimator,
                          error_score=error_score)


def evaluate(model_pipeline, dataset,
             problem_spec='default',
             cv=5,
             progress_bar=True,
             store_preds=False,
             store_estimators=True,
             store_timing=True,
             decode_feat_names=True,
             progress_loc=None,
             **extra_params):
    '''

    Parameters
    -----------
    store_preds : bool, optional
        If set to True, store raw predictions
        in the 'preds' parameter of the returned
        object. This will be a dictionary where
        each dictionary key corresponds to
        a valid predict function for the base model,
        e.g., preds = {'predict': [fold0_preds, fold1_preds, ...]}
        and the value in the dict is a list
        (each element corresponding to each fold)
        of numpy arrays with the raw predictions.

        Note: if store_preds is set to True, then
        also will stored the corresponding ground
        truth labels under dictionary key 'y_true' in
        preds. Or to see corresponding subjects / index,
        check 'val_subjs' in the evaluator.

    '''

    # Base process each component
    estimator, X, y, ps, sk_cv =\
        _sk_prep(model_pipeline=model_pipeline, dataset=dataset,
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
