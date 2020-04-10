from .extensions.Pipeline import ABCD_Pipeline
from ..helpers.ML_Helpers import is_array_like
import os

from .Pipeline_Pieces import (Models, Loaders, Imputers, Scalers,
                              Transformers, Samplers, Feat_Selectors,
                              Ensembles, Drop_Strat)


class Base_Model_Pipeline():

    def __init__(self, p, Data_Scopes, extra_params=None, _print=print):

        self.p = p

        if extra_params is None:
            extra_params = {}
        self.extra_params = extra_params

        self._print = _print

        self.user_passed_objs = {}
        self.pieces = {}

        self.ordered_names = ['loaders', 'imputers', 'scalers', 
                              'transformers', 'samplers',
                              'drop_strat', 'feat_selectors', 'models',
                              'ensembles']

        self._create_pipeline_pieces(Data_Scopes)

    def _check_for_user_passed(self, objs, cnt):

        if objs is not None:

            tuple_flag = False
            if isinstance(objs, tuple):
                tuple_flag = True
                objs = list(objs)

            for o in range(len(objs)):

                if is_array_like(objs[o]):
                    objs[o], cnt = self._check_for_user_passed(objs[o], cnt)

                elif not isinstance(objs[o], str):

                    save_name = 'user passed' + str(cnt)
                    cnt += 1

                    self.user_passed_objs[save_name] = objs[o]
                    objs[o] = save_name

            if tuple_flag:
                objs = tuple(objs)

        return objs, cnt

    def _create_pipeline_pieces(self, Data_Scopes):
        
        # These are all the relevant params from p
        all_params = [(self.p.model, self.p.model_params, None),
                      (self.p.loader, self.p.loader_params, self.p.loader_scope),
                      (self.p.imputer, self.p.imputer_params, self.p.imputer_scope),
                      (self.p.scaler, self.p.scaler_params, self.p.scaler_scope),
                      (self.p.transformer, self.p.transformer_params, self.p.transformer_scope),
                      (self.p.sampler, self.p.sampler_params, self.p.sample_on),
                      (self.p.feat_selector, self.p.feat_selector_params, None),
                      (self.p.ensemble, self.p.ensemble_params, None),
                      ([], [], None)]

        # First process for user passed
        cnt = 0

        all_conv_params = []
        for params in all_params:
            obj_strs = params[0]
            
            obj_strs, cnt = self._check_for_user_passed(obj_strs, cnt)
            params = (obj_strs, *(params)[1:])
            
            all_conv_params.append(params)

        # Generate / process all of the pipeline pieces
        pieces_classes = [Models, Loaders, Imputers, Scalers,
                          Transformers, Samplers, Feat_Selectors,
                          Ensembles, Drop_Strat]

        for params, pieces_class in zip(all_conv_params, pieces_classes):
    
            piece = pieces_class(obj_strs=params[0], param_strs=params[1], scopes=params[2],
                                 user_passed_objs=self.user_passed_objs,
                                 problem_type=self.p.problem_type,
                                 search_type=self.p.search_type,
                                 random_state = self.p.random_state,
                                 Data_Scopes = Data_Scopes,
                                 n_jobs = self.p._base_n_jobs,
                                 extra_params = self.extra_params,
                                 _print = self._print)

            self.pieces[piece.name] = piece

        # Need to process Models for passed Ensembles
        self.pieces['models'].apply_ensembles(self.pieces['ensembles'], self.p.ensemble_split)

    def get_all_params(self):
        '''Returns a dict with all the params combined'''

        all_params = {}
        for name in self.ordered_names:
            all_params.update(self.pieces[name].obj_params)

        return all_params

    def get_all_params_with_names(self):

        all_params = []
        for name in self.ordered_names:
            all_params.append(self.pieces[name].obj_params)

        return all_params, self.ordered_names
    
    def _get_objs(self, names):

        objs = []
        for name in names:
            objs += self.pieces[name].objs

        return objs

    def _get_all_names(self):

        all_obj_names = []
        for name in self.ordered_names:
            
            obj_names = []
            for obj in self.pieces[name].objs:
                obj_names.append(obj[0])

            all_obj_names.append(obj_names)

        return all_obj_names

    def _get_sep_objs(self, names):

        objs = []
        for name in names:
            objs.append(self.pieces[name].objs)

        return objs

    def get_pipeline(self):
        '''Make the model pipeline object'''

        steps = self._get_objs(self.ordered_names)
        names = self._get_all_names()

        # If caching passed, create directory
        if self.p.cache is not None:
            os.makedirs(self.p.cache, exist_ok=True)

        mapping, to_map = self._get_mapping_to_map()

        model_pipeline = ABCD_Pipeline(steps, memory=self.p.cache,
                                       mapping=mapping, to_map=to_map,
                                       names=names)

        return model_pipeline

    def _get_mapping_to_map(self):

        mapping, to_map = False, []

        # If any loaders or transformers passed, need mapping
        # otherwise, just don't use it
        if len(self.pieces['transformers'].objs) > 0 or \
         len(self.pieces['loaders'].objs) > 0:
            
            mapping = True

            # Add every step that needs a mapping
            for valid in self._get_sep_objs(['loaders', 'imputers',
                                             'scalers', 'transformers',
                                             'samplers', 'drop_strat']):

                for step in valid:
                    to_map.append(step[0])

            # Special case for feat_selectors, add only if step is selector
            for step in self.pieces['feat_selectors'].objs:

                try:
                    if step[1].name == 'selector':
                        to_map.append(step[0])
                except AttributeError:
                    pass

        return mapping, to_map



