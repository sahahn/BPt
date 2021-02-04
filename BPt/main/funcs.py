from .Params_Classes import Model, Model_Pipeline, Problem_Spec, Ensemble, CV
from copy import deepcopy
import numpy as np
from ..pipeline.BPtPipelineConstructor import get_pipe
from ..pipeline.Scorers import process_scorers


def model_pipeline_check(model_pipeline, **extra_params):

    # Make deep copy
    pipe = deepcopy(model_pipeline)

    # Add checks on Model_Pipeline
    if not isinstance(pipe, Model_Pipeline):

        # Check for if model str first
        if isinstance(pipe, str):
            pipe = Model(obj=pipe)

        # In case of passed valid single model, wrap in Model_Pipeline
        if hasattr(pipe, '_is_model'):
            pipe = Model_Pipeline(model=pipe)
        else:
            raise RuntimeError('model_pipeline must be a Model_Pipeline',
                               ' model str or Model-like')

    # Set any overlapping extra params
    possible_params = Model_Pipeline._get_param_names()
    valid_params = {key: extra_params[key] for key in extra_params
                    if key in possible_params}
    pipe.set_params(**valid_params)

    # Internal class checks
    pipe._proc_checks()

    return pipe


def problem_spec_check(problem_spec, dataset, **extra_params):

    print(extra_params)

    # If attr checked, then means the passed
    # problem_spec has already been checked and is already
    # a proc'ed and ready copy.
    if hasattr(problem_spec, '_checked') and getattr(problem_spec, '_checked'):
        return problem_spec

    # Check if problem_spec is left as default
    if problem_spec == 'default':
        problem_spec = Problem_Spec()

    # Set ps to copy of problem spec and init
    ps = deepcopy(problem_spec)

    # Apply any passed valid extra params
    possible_params = Problem_Spec._get_param_names()
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

    # Process scorer strs if default
    if ps.scorer == 'default':
        default_scorers = {'regression': ['explained_variance',
                                          'neg_mean_squared_error'],
                           'binary': ['matthews', 'roc_auc',
                                      'balanced_accuracy'],
                           'categorical': ['matthews', 'roc_auc_ovr',
                                           'balanced_accuracy']}
        ps.scorer = default_scorers[pt]

    # Convert to scorer obj
    ps.scorer = process_scorers(ps.scorer, problem_type=ps.problem_type)[1]

    # Set checked flag in ps
    setattr(ps, '_checked', True)

    return ps


def get_estimator(model_pipeline, dataset,
                  problem_spec='default', **extra_params):
    '''Get a sklearn compatible estimator from a :class:`Model_Pipeline`,
    :class:`Dataset` and :class:`Problem_Spec`.

    This function can be used together with Dataset method
    :func:`get_Xy <Dataset.get_Xy>` and it's variants.

    Parameters
    -----------
    model_pipeline : :class:`Model_Pipeline`
        A BPt input class Model_Pipeline to be intialized according
        to the passed Dataset and Problem_Spec.

    dataset : :class:`Dataset`
        The Dataset in which the pipeline should be initialized
        according to. For example, pipeline's can include Scopes,
        these need a reference Dataset.

    problem_spec : :class:`Problem_Spec` or 'default', optional
        `problem_spec` accepts an instance of the
        params class :class:`Problem_Spec`.
        This object is essentially a wrapper around commonly used
        parameters needs to define the context
        the model pipeline should be evaluated in.
        It includes parameters like problem_type, scorer, n_jobs,
        random_state, etc...
        See :class:`Problem_Spec` for more information
        and for how to create an instance of this object.

        If left as 'default', then will initialize a
        Problem_Spec with default params.

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

        if hasattr(obj, 'obj') and isinstance(obj.obj, Model_Pipeline):

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

    def nested_cv_check(obj):

        # If has cv object, try applying dataset
        # can be either in object or in dict
        if hasattr(obj, 'cv') and isinstance(obj.cv, CV):
            setattr(obj, 'cv', obj.cv.apply_dataset(dataset))

        elif isinstance(obj, dict):
            for k in obj:
                if k == 'cv' and isinstance(obj[k], CV):
                    obj[k] = obj[k].apply_dataset(dataset)
                else:
                    nested_cv_check(obj[k])

        elif isinstance(obj, list):
            [nested_cv_check(o) for o in obj]
        elif isinstance(obj, tuple):
            (nested_cv_check(o) for o in obj)
        elif hasattr(obj, 'get_params'):
            for param in obj.get_params(deep=False):
                nested_cv_check(getattr(obj, param))

    # Run nested check for any CV input param objects
    nested_cv_check(pipe)

    return pipe


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


def _sk_prep(model_pipeline, dataset,
             problem_spec='default', **extra_params):
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

    # Convert cv to sklearn compatible
    cv = ps.cv.get_cv(X.index, random_state=ps.random_state, return_index=True)

    return estimator, X, y, ps, cv


def cross_val_score(model_pipeline, dataset,
                    problem_spec='default',
                    sk_n_jobs=1, verbose=0,
                    fit_params=None,
                    error_score=np.nan,
                    **extra_params):
    '''This function is a BPt compatible wrapper around the scikit-learn
    function cross_val_score,
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html


    Parameters
    ----------
    model_pipeline : :class:`Model_Pipeline`
        A BPt input class Model_Pipeline to be intialized according
        to the passed Dataset and Problem_Spec, and then evaluated.

    dataset : :class:`Dataset`
        The Dataset in which the pipeline should be initialized
        according to, and data drawn from. See parameter
        `subjects` to use only a subset of the columns or subjects
        in this call to cross_val_score.

    problem_spec : :class:`Problem_Spec` or 'default', optional
        `problem_spec` accepts an instance of the
        params class :class:`Problem_Spec`.
        This object is essentially a wrapper around commonly used
        parameters needs to define the context
        the model pipeline should be evaluated in.
        It includes parameters like problem_type, scorer, n_jobs,
        random_state, etc...
        See :class:`Problem_Spec` for more information
        and for how to create an instance of this object.

        If left as 'default', then will initialize a
        Problem_Spec with default params.

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
    estimator, X, y, ps, cv =\
        _sk_prep(model_pipeline=model_pipeline, dataset=dataset,
                 problem_spec=problem_spec, **extra_params)

    # Just take first scorer if list
    if isinstance(ps.scorer, list):
        ps.scorer = ps.scorer[0]

    from sklearn.model_selection import cross_val_score
    return cross_val_score(estimator=estimator, X=X, y=y, scoring=ps.scorer,
                           cv=cv, n_jobs=sk_n_jobs, verbose=verbose,
                           fit_params=fit_params, error_score=error_score)


def cross_validate(self, model_pipeline, dataset,
                   problem_spec='default',
                   sk_n_jobs=1, verbose=0,
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
    model_pipeline : :class:`Model_Pipeline`
        A BPt input class Model_Pipeline to be intialized according
        to the passed Dataset and Problem_Spec, and then evaluated.

    dataset : :class:`Dataset`
        The Dataset in which the pipeline should be initialized
        according to, and data drawn from. See parameter
        `subjects` to use only a subset of the columns or subjects
        in this call to cross_val_score.

    problem_spec : :class:`Problem_Spec` or 'default', optional
        `problem_spec` accepts an instance of the
        params class :class:`Problem_Spec`.
        This object is essentially a wrapper around commonly used
        parameters needs to define the context
        the model pipeline should be evaluated in.
        It includes parameters like problem_type, scorer, n_jobs,
        random_state, etc...
        See :class:`Problem_Spec` for more information
        and for how to create an instance of this object.

        If left as 'default', then will initialize a
        Problem_Spec with default params.

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
    estimator, X, y, ps, cv =\
        _sk_prep(model_pipeline=model_pipeline, dataset=dataset,
                 problem_spec=problem_spec, **extra_params)

    from sklearn.model_selection import cross_validate
    return cross_validate(estimator=estimator, X=X, y=y, scoring=ps.scorer,
                          cv=cv, n_jobs=sk_n_jobs, verbose=verbose,
                          fit_params=fit_params,
                          return_train_score=return_train_score,
                          return_estimator=return_estimator,
                          error_score=error_score)
