from .Params_Classes import Model, Model_Pipeline, Problem_Spec, Ensemble, CV
from copy import deepcopy
from ..pipeline.BPtPipelineConstructor import get_pipe


def model_pipeline_check(model_pipeline):

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

    # Internal class checks
    pipe._proc_checks()

    return pipe


def problem_spec_check(problem_spec, dataset):

    # Check if problem_spec is left as default
    if problem_spec == 'default':
        problem_spec = Problem_Spec()

    # Set ps to copy of problem spec and init
    ps = deepcopy(problem_spec)
    ps._proc_checks()

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

    return ps


def get_estimator(model_pipeline, dataset, problem_spec):
    '''Get from input parameter style model_pipeline, a sklearn compatible
    estimator. This also requires a Dataset, and Problem_Spec.
    '''

    # @TODO add verbose option?

    # Check passed input - note: returns deep copies
    pipe = model_pipeline_check(model_pipeline)
    ps = problem_spec_check(problem_spec, dataset)

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
    data_cols = dataset.get_cols(scope='data', columns=ps.scope)
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
