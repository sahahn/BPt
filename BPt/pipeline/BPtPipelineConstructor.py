from .BPtPipeline import BPtPipeline
from ..main.input import Custom
from ..util import is_array_like
from .BPtSearchCV import get_search_cv


def _check_for_user_passed(objs, cnt, up):

    if objs is not None:

        # If list / array like passed
        if is_array_like(objs):

            # Call recursively on each entry
            for o in range(len(objs)):
                objs[o], cnt = _check_for_user_passed(objs[o], cnt, up)

        # If a single obj
        else:

            if hasattr(objs, 'base_model'):
                if objs.base_model is not None:
                    objs.base_model, cnt =\
                        _check_for_user_passed(objs.base_model, cnt, up)
            if hasattr(objs, 'models'):
                if objs.models is not None:
                    objs.models, cnt =\
                        _check_for_user_passed(objs.models, cnt, up)
            if hasattr(objs, 'target_scaler'):
                if objs.target_scaler is not None:
                    objs.target_scaler, cnt =\
                        _check_for_user_passed(objs.target_scaler, cnt, up)

            # If a Param obj - call recursively to set the value of the
            # base obj
            if hasattr(objs, 'obj'):
                objs.obj, cnt = _check_for_user_passed(objs.obj, cnt, up)

            # Skip custom
            elif isinstance(objs, Custom):
                pass

            # Now, we assume any single obj that gets here, if not
            # a str is user passed custom obj
            elif not isinstance(objs, str):
                save_name = 'Custom ' + str(cnt)
                cnt += 1
                up[save_name] = objs
                objs = save_name

    return objs, cnt


def check_for_user_passed(objs):

    user_passed = {}
    objs, _ = _check_for_user_passed(objs, 0, up=user_passed)
    return objs, user_passed


class BPtPipelineConstructor():

    def __init__(self, pipeline_params, problem_spec, dataset):

        # Save param search here - should be None or dict
        self.param_search = pipeline_params.param_search

        # Extract params for BPtPipeline
        self.cache_loc = pipeline_params.cache_loc
        self.verbose = pipeline_params.verbose

        # Save some params to pass around when building the steps
        spec = problem_spec._get_spec()

        # Get params as ordered list of steps
        pipe_params = pipeline_params._get_steps()

        # Create the pipeline pieces
        self._create_pipeline_pieces(
            pipe_params=pipe_params,
            dataset=dataset, spec=spec)

    def _create_pipeline_pieces(self, pipe_params,
                                dataset, spec):

        # First check for user passed - update in place
        # Check for user passed out here to avoid possible name collisions.
        pipe_params, user_passed_objs = check_for_user_passed(pipe_params)

        # Want to process all pieces with the same constructor
        # at the same time, then put back in the right order

        # Create empty list of objects to fill
        self.objs = [None for _ in range(len(pipe_params))]

        # Params is dict, so unordered
        self.params = {}

        # Get unique constructors
        constructors = [param._constructor for param in pipe_params]
        unique = set(constructors)

        # For each unique piece_type
        for constr_type in unique:

            # Get a list of the pieces of this type
            # and their original index
            in_scope, inds = [], []
            for i in range(len(pipe_params)):
                if constructors[i] is constr_type:
                    in_scope.append(pipe_params[i])
                    inds.append(i)

            # Init the correct constructor
            constructor = constr_type(spec=spec, dataset=dataset,
                                      user_passed_objs=user_passed_objs)

            # Process the pipe params of this type
            objs, params = constructor.process(in_scope)

            # Put in correct spots
            for i, ind in enumerate(inds):
                self.objs[ind] = objs[i]

            # Update params
            self.params.update(params)

        # Now self.objs contains the initialized steps, and self.params
        # the corresponding param dists.

    def get_pipeline(self):
        '''Make the model pipeline object'''

        pipeline = BPtPipeline(self.objs, verbose=self.verbose,
                               cache_loc=self.cache_loc)

        return pipeline

    def is_search(self):

        if self.param_search is None:
            return False
        return True

    def get_search_wrapped_pipeline(self):

        # Grab the base pipeline
        base_pipeline = self.get_pipeline()

        # If no search, just return copy of pipeline
        # and the params seperately
        if not self.is_search():
            return base_pipeline, self.params

        # Create the search object
        search_model = get_search_cv(estimator=base_pipeline,
                                     param_search=self.param_search,
                                     param_distributions=self.params)

        # Return search model, and empty dict, since params are used
        return search_model, {}


def get_pipe(pipeline_params, problem_spec, dataset):

    # Init the Pipeline, which creates the pipeline pieces
    pipeline_constructor =\
        BPtPipelineConstructor(pipeline_params=pipeline_params,
                               problem_spec=problem_spec,
                               dataset=dataset)
    # Set the final model // search wrap
    Model, pipeline_params =\
        pipeline_constructor.get_search_wrapped_pipeline()

    return Model, pipeline_params
