from .ABCD_Pipeline import ABCD_Pipeline
from ..helpers.ML_Helpers import is_array_like
import os

from ..helpers.VARS import ORDERED_NAMES

from .Pipeline_Pieces import (Models, Loaders, Imputers, Scalers,
                              Transformers, Feat_Selectors,
                              Drop_Strat)

from copy import deepcopy
from .Nevergrad import NevergradSearchCV
from .Scope_Model import Scope_Model


class Base_Model_Pipeline():

    def __init__(self, pipeline_params, model_spec, Data_Scopes, _print=print):

        # Save param search here
        self.param_search = pipeline_params.param_search

        if self.param_search is None:
            model_spec['search_type'] = None
        else:
            model_spec['search_type'] = self.param_search.search_type

        # Set n_jobs in model spec
        model_spec['n_jobs'] = pipeline_params._n_jobs

        # Save cache param + print
        self.cache = pipeline_params.cache
        self._print = _print

        # Extract ordered
        ordered_pipeline_params = pipeline_params.get_ordered_pipeline_params()

        # Create the pipeline pieces
        self._create_pipeline_pieces(
            ordered_pipeline_params=ordered_pipeline_params,
            Data_Scopes=Data_Scopes,
            spec=model_spec)

    def _create_pipeline_pieces(self, ordered_pipeline_params, Data_Scopes,
                                spec):

        # Order is:
        # ['loaders', 'imputers', 'scalers',
        #  'transformers', '_drop_strat',
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
                          Transformers, Drop_Strat,
                          Feat_Selectors, Models]

        # Generate / process all of the pipeline pieces in order
        for params, piece_class, name in zip(conv_pipeline_params,
                                             pieces_classes, ORDERED_NAMES):

            piece = piece_class(user_passed_objs=self.user_passed_objs,
                                Data_Scopes=Data_Scopes,
                                spec=spec,
                                _print=self._print)
            objs, params = piece.process(params)

            self.named_objs[name] = objs
            self.named_params[name] = params

        # Set mapping to map
        self._set_mapping_to_map()

    def _check_for_user_passed(self, objs, cnt):

        if objs is not None:

            # If list / array like passed
            if is_array_like(objs):

                # Call recursively on each entry
                for o in range(len(objs)):
                    objs[o], cnt = self._check_for_user_passed(objs[o], cnt)

            # If a single obj
            else:

                # If a Param obj - call recursively to set the value of the
                # base obj
                if hasattr(objs, 'obj'):
                    objs.obj, cnt = self._check_for_user_passed(objs.obj, cnt)

                # Now, we assume any single obj that gets here, if not
                # a str is user passed obj
                elif not isinstance(objs, str):
                    save_name = 'user passed' + str(cnt)
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

    def get_all_params_with_names(self):

        all_params = []
        for name in ORDERED_NAMES:
            all_params.append(self.named_params[name])

        return all_params, ORDERED_NAMES

    def _get_objs(self, names):

        objs = []
        for name in names:
            objs += self.named_objs[name]

        return objs

    def _get_all_names(self):

        all_obj_names = []
        for name in ORDERED_NAMES:

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

        steps = self._get_objs(ORDERED_NAMES)
        names = self._get_all_names()

        # If caching passed, create directory
        if self.cache is not None and not os.path.isdir(self.cache):
            self._print("Creating pipeline cache dr at:", self.cache)
            os.makedirs(self.cache, exist_ok=True)

        model_pipeline = ABCD_Pipeline(steps, memory=self.cache,
                                       mapping=self.mapping,
                                       to_map=self.to_map,
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
            if isinstance(step[1], Scope_Model):
                to_map.append(step[0])

        self.mapping = mapping
        self.to_map = to_map

    def is_search(self):

        if self.param_search is None:
            return False
        return True

    def get_search_wrapped_pipeline(self, search_scorer=None,
                                    weight_search_scorer=None,
                                    random_state=None,
                                    progress_loc=None):

        # Grab the base pipeline
        base_pipeline = self.get_pipeline()

        # If no search, just return copy of pipeline
        if self.param_search is None:
            return deepcopy(base_pipeline)

        # Create the search object
        search_model =\
            NevergradSearchCV(params=self.param_search,
                              estimator=base_pipeline,
                              param_distributions=self.get_all_params(),
                              scoring=search_scorer,
                              weight_scorer=weight_search_scorer,
                              random_state=random_state,
                              progress_loc=progress_loc)

        return deepcopy(search_model)
