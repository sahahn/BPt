from .BPt_Pipeline import BPt_Pipeline
from ..helpers.ML_Helpers import is_array_like
import os

from ..helpers.VARS import ORDERED_NAMES

from .Pipeline_Pieces import (Models, Loaders, Imputers, Scalers,
                              Transformers, Feat_Selectors)

from .BPtSearchCV import get_search_cv


class Model_Pipeline():

    def __init__(self, pipeline_params, spec, Data_Scopes, verbose=False):

        # Save param search here
        self.param_search = pipeline_params.param_search

        if self.param_search is None:
            spec['search_type'] = None
        else:
            spec['search_type'] = self.param_search.search_type

        # Set n_jobs in model spec
        spec['n_jobs'] = pipeline_params.n_jobs

        # Save cache param
        self.cache = pipeline_params.cache
        self.verbose = verbose

        # Extract ordered
        ordered_pipeline_params = pipeline_params.get_ordered_pipeline_params()

        # Create the pipeline pieces
        self._create_pipeline_pieces(
            ordered_pipeline_params=ordered_pipeline_params,
            Data_Scopes=Data_Scopes, spec=spec)

    def _create_pipeline_pieces(self, ordered_pipeline_params,
                                Data_Scopes, spec):

        # Order is:
        # ['loaders', 'imputers',
        #  'scalers',
        #  'transformers',
        #  'feat_selectors', 'model']

        # First check for user passed
        self.user_passed_objs = {}
        conv_pipeline_params = []
        cnt = 0

        for params in ordered_pipeline_params:
            conv_params, cnt = self._check_for_user_passed(params, cnt)
            conv_pipeline_params.append(conv_params)

        self.named_objs = {}
        self.named_params = {}

        # These are the corresponding pieces classes
        pieces_classes = [Loaders, Imputers, Scalers,
                          Transformers, Feat_Selectors, Models]

        # Generate / process all of the pipeline pieces in order
        for params, piece_class, name in zip(conv_pipeline_params,
                                             pieces_classes, ORDERED_NAMES):

            piece = piece_class(user_passed_objs=self.user_passed_objs,
                                Data_Scopes=Data_Scopes,
                                spec=spec)
            objs, params = piece.process(params)

            self.named_objs[name] = objs
            self.named_params[name] = params

        # Set mapping to map
        self._set_mapping_to_map()
        self._set_needs_index()

    def _check_for_user_passed(self, objs, cnt):

        if objs is not None:

            # If list / array like passed
            if is_array_like(objs):

                # Call recursively on each entry
                for o in range(len(objs)):
                    objs[o], cnt = self._check_for_user_passed(objs[o], cnt)

            # If a single obj
            else:

                if hasattr(objs, 'base_model'):
                    if objs.base_model is not None:
                        objs.base_model, cnt =\
                            self._check_for_user_passed(objs.base_model, cnt)
                if hasattr(objs, 'models'):
                    if objs.models is not None:
                        objs.models, cnt =\
                            self._check_for_user_passed(objs.models, cnt)
                if hasattr(objs, 'target_scaler'):
                    if objs.target_scaler is not None:
                        objs.target_scaler, cnt =\
                            self._check_for_user_passed(objs.target_scaler,
                                                        cnt)

                # If a Param obj - call recursively to set the value of the
                # base obj
                if hasattr(objs, 'obj'):
                    objs.obj, cnt = self._check_for_user_passed(objs.obj, cnt)

                # Now, we assume any single obj that gets here, if not
                # a str is user passed custom obj
                elif not isinstance(objs, str):
                    save_name = 'Custom ' + str(cnt)
                    cnt += 1

                    self.user_passed_objs[save_name] = objs
                    objs = save_name

        return objs, cnt

    def get_all_params(self):
        '''Returns a dict with all the params combined'''

        all_params = {}
        for name in ORDERED_NAMES:
            all_params.update(self.named_params[name])

        return all_params

    def _get_objs(self, names):

        objs = []
        for name in names:
            objs += self.named_objs[name]

        return objs

    def _get_names(self, names):

        all_obj_names = []
        for name in names:

            obj_names = []
            for obj in self.named_objs[name]:
                obj_names.append(obj[0])

            all_obj_names.append(obj_names)

        return all_obj_names

    def _get_sep_objs(self, names):

        objs = []
        for name in names:
            objs.append(self.named_objs[name])

        return objs

    def get_pipeline(self):
        '''Make the model pipeline object'''

        # Get all steps + names
        steps = self._get_objs(ORDERED_NAMES)
        names = self._get_names(ORDERED_NAMES)

        # If caching passed, create directory
        if self.cache is not None and not os.path.isdir(self.cache):
            os.makedirs(self.cache, exist_ok=True)

        model_pipeline = BPt_Pipeline(steps, memory=self.cache,
                                      verbose=self.verbose,
                                      add_mapping=self.add_mapping,
                                      to_map=self.to_map,
                                      needs_index=self.needs_index,
                                      names=names)

        return model_pipeline

    def _set_mapping_to_map(self):

        mapping, to_map = True, []

        # Add every step that needs a mapping
        for valid in self._get_sep_objs(set(ORDERED_NAMES) - set(['model'])):
            for step in valid:
                to_map.append(step[0])

        # Handle model special
        for step in self.named_objs['model']:
            if hasattr(step[1], '_needs_mapping'):
                if step[1]._needs_mapping:
                    to_map.append(step[0])

        self.add_mapping = mapping
        self.to_map = to_map

    def _set_needs_index(self):

        self.needs_index = []
        for step in self.named_objs['model']:
            if hasattr(step[1], '_needs_train_data_index'):
                if step[1]._needs_train_data_index:
                    self.needs_index.append(step[0])

    def is_search(self):

        if self.param_search is None:
            return False
        return True

    def get_search_wrapped_pipeline(self, progress_loc=None):

        # Grab the base pipeline
        base_pipeline = self.get_pipeline()

        # If no search, just return copy of pipeline
        if not self.is_search():
            return base_pipeline, self.get_all_params()

        # Create the search object
        search_model = get_search_cv(
                estimator=base_pipeline,
                param_search=self.param_search,
                param_distributions=self.get_all_params(),
                progress_loc=progress_loc)

        return search_model, {}


def get_pipe(pipeline_params, problem_spec, Data_Scopes, progress_loc,
             verbose=False):

    # Get the model specs from problem_spec
    model_spec = problem_spec.get_model_spec()

    # Init the Model_Pipeline, which creates the pipeline pieces
    base_model_pipeline =\
        Model_Pipeline(pipeline_params=pipeline_params,
                       spec=model_spec,
                       Data_Scopes=Data_Scopes,
                       verbose=verbose)

    # Set the final model // search wrap
    Model, pipeline_params =\
        base_model_pipeline.get_search_wrapped_pipeline(
            progress_loc=progress_loc)

    return Model, pipeline_params
