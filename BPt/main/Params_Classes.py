from copy import deepcopy
from sklearn.base import BaseEstimator
import pandas as pd
from ..helpers.ML_Helpers import conv_to_list, proc_input, proc_type_dep_str

from ..helpers.VARS import ORDERED_NAMES
from ..main.Input_Tools import (is_duplicate, is_pipe, is_select,
                                is_special, is_value_subset)
from ..pipeline.Scorers import process_scorers


def proc_all(base_obj):

    if base_obj is None:
        return

    proc_name(base_obj, 'obj')
    proc_name(base_obj, 'base_model')
    proc_name(base_obj, 'models')
    proc_name(base_obj, 'scorer')


def proc_name(base_obj, name):

    if hasattr(base_obj, name):
        obj = getattr(base_obj, name)

        if isinstance(obj, list):
            for o in obj:
                proc_all(o)

        elif hasattr(obj, 'obj'):
            proc_all(obj)

        elif isinstance(obj, str):
            setattr(base_obj, name, proc_input(obj))


class Params(BaseEstimator):

    def set_values(self, args):

        for key in args:
            if key != 'self' and hasattr(self, key):

                try:
                    if isinstance(args[key], str):
                        if args[key] != 'default':
                            setattr(self, key, args[key])
                    else:
                        setattr(self, key, args[key])

                # If value error, set to key
                except ValueError:
                    setattr(self, key, args[key])

    def copy(self):
        return deepcopy(self)


class Check():

    def check_args(self):

        if hasattr(self, 'obj'):
            self._check_obj()

        if hasattr(self, 'params'):
            self._check_params()

        if hasattr(self, 'obj') and hasattr(self, 'params'):
            self._check_obj_and_params()

        if hasattr(self, 'scope'):
            self._check_scope()

        if hasattr(self, 'extra_params'):
            self._check_extra_params()

        if hasattr(self, 'base_model'):
            self._check_base_model()

        try:
            self.check_extra_args()
        except AttributeError:
            pass

    def _check_obj(self):

        obj = getattr(self, 'obj')

        if obj is None:
            raise IOError('passed obj cannot be None, to ignore',
                          'the object itself',
                          'set it within Model_Pipeline to None,',
                          ' not obj here.')

        if isinstance(obj, list) and not is_pipe(obj):
            raise IOError('You may only pass a list of objs with ',
                          'special input',
                          'Pipe()')

    def _check_params(self):

        params = getattr(self, 'params')

        if params is None:
            setattr(self, 'params', 0)

    def _check_obj_and_params(self):

        obj = getattr(self, 'obj')
        params = getattr(self, 'params')

        if is_pipe(obj):

            if not isinstance(params, list):

                if params == 0:
                    new_params = [0 for i in range(len(obj))]
                    setattr(self, 'params', new_params)
                else:
                    raise IOError('obj is passed as Pipe, but params was not.',
                                  'Make sure',
                                  'params is either a list or Pipe ',
                                  'with the same length as',
                                  'self.obj')

            elif len(params) != len(obj):

                if params == [0]:
                    new_params = [0 for i in range(len(obj))]
                    setattr(self, 'params', new_params)
                else:
                    raise IOError('obj is passed as Pipe, Make sure',
                                  'params is either a list or Pipe with the ',
                                  'same length as',
                                  'self.obj')

    def _check_scope(self):

        pass
        # Nothing to check as of right now
        # scope = getattr(self, 'scope')

    def _check_extra_params(self):

        extra_params = getattr(self, 'extra_params')
        if extra_params is None:
            return
        #    setattr(self, 'extra_params', {})

        elif not isinstance(extra_params, dict):
            raise IOError('extra params must be a dict!')

    def _check_base_model(self):

        base_model = getattr(self, 'base_model')

        # None is okay
        if base_model is None:
            return

        # Assume select is okay too
        if is_select(base_model):
            return

        if not hasattr(base_model, '_is_model'):
            raise IOError('base_model must be either None or a valid '
                          'Model / Ensemble ',
                          'set of wrapeper params!')


class Piece(Params, Check):
    pass


class Loader(Piece):

    def __init__(self, obj, params=0, scope='data files',
                 cache_loc=None, extra_params=None,
                 fix_n_wrapper_jobs='default'):
        ''' Loader refers to transformations which operate on loaded Data_Files.
        (See :func:`Load_Data_Files`).
        They in essence take in saved file locations, and after some series
        of transformations pass on compatible features.
        Notably loaders
        define operations which are computed on single files indepedently.

        Parameters
        ----------
        obj : str, custom obj or :class:`Pipe`
            `obj` selects the base loader object to use, this can be either
            a str corresponding to
            one of the preset loaders found at :ref:`Loaders`.
            Beyond pre-defined loaders, users can
            pass in custom objects
            (they just need to have a defined fit_transform function
            which when passed the already loaded file, will return
            a 1D representation of that subjects
            features.

            `obj` can also be passed as a :class:`Pipe`.
            See :class:`Pipe`'s documentation to
            learn more on how this works, and why you might want to use it.

            See :ref:`Pipeline Objects` to read more
            about pipeline objects in general.

            For example, the 'identity' loader will load in saved data at
            the stored file
            location, lets say they are 2d numpy arrays,
            and will return a flattened version
            of the saved arrays, with each data point as a feature.
            A more practical example
            might constitute loading in say 3D neuroimaging data,
            and passing on features as extracted by ROI.

        params : int, str or dict of :ref:`params<Params>`, optional
            `params` determines optionally if the distribution
            of hyper-parameters to
            potentially search over for this loader. Preset param distributions are
            listed for each choice of obj at :ref:`Loaders`, and you can read more on
            how params work more generally at :ref:`Params`.

            If obj is passed as :class:`Pipe`, see :class:`Pipe` for an example on how different
            corresponding params can be passed to each piece individually.

            ::

                default = 0

        scope : :ref:`valid scope<Scopes>`, optional
            `scope` determines on which subset of features the specified loader
            should transform. See :ref:`Scopes` for more information on
            how scopes can
            be specified.

            You will likely want to use either custom key based scopes, or the
            'data files' preset scope, as something like 'covars'
            won't make much sense,
            when atleast for now, you cannot even load Covars data files.

            ::

                default = 'data files'

        cache_loc : str, Path or None, optional
            Optional location in which to cache loader transformations.

        extra_params : :ref`extra params dict<Extra Params>`, optional

            See :ref:`Extra Params`

            ::

                default = None

        fix_n_wrapper_jobs : int or 'default', optional
            Typically this parameter is left as default, but
            in special cases you may want to set this. It controls
            the number of jobs fixed for the Loading Wrapper.

            This parameter can be used to set that value.

            ::

                default = 'default'
        '''

        self.obj = obj
        self.params = params
        self.scope = scope
        self.cache_loc = cache_loc
        self.extra_params = extra_params
        self.fix_n_wrapper_jobs = fix_n_wrapper_jobs

        self.check_args()


class Imputer(Piece):

    def __init__(self, obj, params=0, scope='all',
                 base_model=None, base_model_type='default',
                 extra_params=None):
        ''' If there is any missing data (NaN's) that have been kept
        within data or covars, then an imputation strategy must be
        defined! This object allows you to define an imputation strategy.
        In general, you should need at most two Imputers, one for all
        `float` type data and one for all categorical data, assuming you
        have been present, and they both have missing values.

        Parameters
        ----------
        obj : str
            `obj` selects the base imputation strategy to use.
            See :ref:`Imputers` for all avaliable options.
            Notably, if 'iterative' is passed,
            then a base model must also be passed!
            Also note that the `sample_posterior`
            argument within `iterative` imputer is not currently supported.

            See :ref:`Pipeline Objects` to read more about
            pipeline objects in general.

        params : int, str or dict of :ref:`params<Params>`, optional
            `params` set an associated distribution of hyper-parameters to
            potentially search over with the Imputer.
            Preset param distributions are
            listed for each choice of params with the corresponding
            obj at :ref:`Imputers`,
            and you can read more on how params
            work more generally at :ref:`Params`.

            ::

                default = 0

        scope : :ref:`valid scope<Scopes>`, optional
            `scope` determines on which subset of features the specified
            imputer will have access to.

            The main options that make sense for imputer are
            one for `float` data and one for `categorical` / 'cat' datatypes.
            Though you can also pass a custom set of keys.

            Note: If using iterative imputation you may want to carefully
            consider the scope passed. For example, while it may be beneficial
            to impute categorical and float features seperately, i.e., with
            different base_model_type's
            (categorical for categorical and regression for float), you must
            also consider that in predicting the missing values under
            this setup, the categorical imputer would not have access to
            to the float features and vice versa.

            In this way, you
            may want to either just treat all features as float, or
            instead of imputing categorical features, load missing
            values as a seperate category - and then set the scope
            here to be 'all', such that the iterative imputer has
            access to all features. Essently why this is neccisary
            is the iterative imputer
            will try to replace any NaN value present in its input
            features.

            See :ref:`Scopes` for more information on how scopes can
            be specified.

            ::

                default = 'all'

        scope : {'float', 'cat', custom}, optional
            `scope` determines on which subset of features
            the imputer should act on.

            :ref:`Scopes`.

            ::

                default = 'float'

        base_model : :class:`Model`, :class:`Ensemble` or None, optional
            If 'iterative' is passed to obj, then a base_model is required in
            order to perform iterative imputation! The base model can be
            any valid Model_Pipeline Model.

            ::

                default = None

        base_model_type : 'default' or Problem Type, optional
            In setting a base imputer model, it may be desirable to
            have this model have a different 'problem type', then your
            over-arching problem. For example, if performing iterative
            imputation on categorical features only, you will likely
            want to use a categorical predictor - but for imputing on
            float-type features, you will want to use a 'regression' type
            base model.

            Choices are {'binary', 'regression', 'categorical'} or 'default'.
            If 'default', then the following behavior will be applied:
            If the scope of the imputer is set to 'cat' or 'categorical',
            then the 'categorical' problem type will be used for the base
            model. If anything else, then the 'regression' type will be used.

            ::

                default = 'default'


        extra_params : :ref`extra params dict<Extra Params>`, optional

            See :ref:`Extra Params`

            ::

                default = None

        '''

        self.obj = obj
        self.params = params
        self.scope = scope
        self.base_model = deepcopy(base_model)
        self.base_model_type = base_model_type
        self.extra_params = extra_params

        self.check_args()


class Scaler(Piece):

    def __init__(self, obj, params=0, scope='float', extra_params=None):
        ''' Scaler refers to a piece in the :class:`Model_Pipeline`,
        which is responsible
        for performing any sort of scaling or transformation on the data
        which doesn't require the
        target variable, and doesn't change the number of data
        points or features.

        Parameters
        ----------
        obj : str or custom obj
            `obj` if passed a str selects a scaler from the preset defined
            scalers, See :ref:`Scalers`
            for all avaliable options. If passing a custom object, it must
            be a sklearn compatible
            transformer, and further must not require the target variable,
            not change the number of data
            points or features.

            See :ref:`Pipeline Objects` to read more about
            pipeline objects in general.

        params : int, str or dict of :ref:`params<Params>`, optional
            `params` set an associated distribution of hyper-parameters to
            potentially search over with this Scaler.
            Preset param distributions are
            listed for each choice of params with the
            corresponding obj at :ref:`Scalers`,
            and you can read more on how params work more
            generally at :ref:`Params`.

            ::

                default = 0

        scope : :ref:`valid scope<Scopes>`, optional
            `scope` determines on which subset of features the specified scaler
            should transform. See :ref:`Scopes` for more
            information on how scopes can
            be specified.

            ::

                default = 'float'

        extra_params : :ref`extra params dict<Extra Params>`, optional

            See :ref:`Extra Params`

            ::

                default = None

        '''

        self.obj = obj
        self.params = params
        self.scope = scope
        self.extra_params = extra_params

        self.check_args()


class Transformer(Piece):

    def __init__(self, obj, params=0, scope='float', cache_loc=None,
                 extra_params=None, fix_n_wrapper_jobs='default'):
        ''' The Transformer is base optional component of the
        :class:`Model_Pipeline` class.
        Transformers define any type of transformation to the loaded
        data which may change the number
        of features in a non-simple way (i.e., conceptually distinct
        from :class:`Feat_Selector`, where you
        know in advance the transformation is just selecting a subset
        of existing features). These are
        transformations like applying Principle Component Analysis,
        or on the fly One Hot Encoding.

        Parameters
        ----------
        obj : str or custom_obj
            `obj` if passed a str selects from the avaliable class
            defined options for
            transformer as found at :ref:`Transformers`.

            If a custom object is passed as `obj`, it must be a
            sklearn api compatible
            transformer (i.e., have fit, transform, get_params
            and set_params methods, and further be
            cloneable via sklearn's clone function).
            See :ref:`Custom Input Objects` for more info.

            See :ref:`Pipeline Objects` to read more about
            pipeline objects in general.

        params : int, str or dict of :ref:`params<Params>`, optional
            `params` determines optionally if the distribution of
            hyper-parameters to potentially search over for this transformer.
            Preset param distributions are listed for each choice of obj
            at :ref:`Transformers`, and you can read more on
            how params work more generally at :ref:`Params`.

            ::

                default = 0

        scope : :ref:`valid scope<Scopes>`, optional
            `scope` determines on which subset of features
            the specified transformer
            should transform. See :ref:`Scopes` for more information
            on how scopes can
            be specified.

            Specifically, it may be useful to consider the use of
            :class:`Duplicate` here.

            ::

                default = 'float'

        extra_params : :ref`extra params dict<Extra Params>`, optional

            See :ref:`Extra Params`

            ::

                default = None

        fix_n_wrapper_jobs : int or 'default', optional
            This parameter is ignored right now for Transformers

            ::

                default = 'default'
        '''

        self.obj = obj
        self.params = params
        self.scope = scope
        self.cache_loc = cache_loc
        self.extra_params = extra_params
        self.fix_n_wrapper_jobs = fix_n_wrapper_jobs

        self.check_args()


class Feat_Selector(Piece):

    def __init__(self, obj, params=0, scope='all',
                 base_model=None, extra_params=None):
        ''' Feat_Selector is a base piece of :class:`Model_Pipeline`, which is designed
        to preform feature selection.

        Parameters
        ----------
        obj : str
            `obj` selects the feature selection strategy to use.
            See :ref:`Feat Selectors`
            for all avaliable options. Notably, if 'rfe' is passed, then a
            base model must also be passed!

            See :ref:`Pipeline Objects` to read more about pipeline objects
            in general.

        params : int, str or dict of :ref:`params<Params>`, optional
            `params` set an associated distribution of hyper-parameters to
            potentially search over with this Feat_Selector.
            Preset param distributions are
            listed for each choice of params with the corresponding
            obj at :ref:`Feat Selectors`,
            and you can read more on how params
            work more generally at :ref:`Params`.

            ::

                default = 0

        scope : :ref:`valid scope<Scopes>`, optional
            `scope` determines on which subset of features the specified
            feature selector will have access to.
            See :ref:`Scopes` for more information on how scopes can
            be specified.

            ::

                default = 'all'

        base_model : :class:`Model`, :class:`Ensemble` or None, optional
            If 'rfe' is passed to obj, then a base_model is required in
            order to perform recursive feature elimination.
            The base model can be
            any valid argument accepts by
            param `model` in :class:`Model_Pipeline`.

            ::

                default = None

        extra_params : :ref`extra params dict<Extra Params>`, optional

            See :ref:`Extra Params`

            ::

                default = None

        '''

        self.obj = obj
        self.params = params
        self.scope = scope
        self.base_model = deepcopy(base_model)
        self.base_model_type = None
        self.extra_params = extra_params
        self.check_args()


class Model(Piece):

    _is_model = True

    def __init__(self, obj, params=0, scope='all', param_search=None,
                 target_scaler=None, extra_params=None):
        ''' Model represents a base components of the :class:`Model_Pipeline`,
        specifically a single Model / estimator.
        Model can also be used as a component
        in building other pieces of the model pipeline,
        e.g., :class:`Ensemble`.

        Parameters
        ----------

        obj : str, or custom obj
            `obj` selects the base model object to use from either
            a preset str indicator found at :ref:`Models`,
            or from a custom passed user model (compatible w/ sklearn api).

            See :ref:`Pipeline Objects` to
            read more about pipeline objects in general.

            `obj` should be wither a single str indicator or a
            single custom model object, and not passed a list-like of either.
            If an ensemble of models is requested, then see :class:`Ensemble`.

        params : int, str or dict of :ref:`params<Params>`, optional
            `params` optionally set an
            associated distribution of hyper-parameters to
            this model object. Preset param distributions are
            listed for each choice of obj at :ref:`Models`,
            and you can read more on
            how params work more generally at :ref:`Params`.

            ::

                default = 0

        scope : :ref:`valid scope<Scopes>`, optional
            `scope` determines on which subset of features the specified model
            should work on. See :ref:`Scopes` for more
            information on how scopes can
            be specified.

            ::

                default = 'all'

        param_search : :class:`Param_Search`, None, optional
            If None, by default, this will be a base model.
            Alternatively, by passing a :class:`Param_Search` instance here,
            it specifies that this model should be wrapped in a
            Nevergrad hyper-parameter search object.

            This can be useful to create Model's which have
            a nested hyper-parameter tuning independent from the
            other pipeline steps.

            ::

                default = None

        target_scaler : Scaler, None, optional

            Still somewhat experimental, can pass
            a Scaler object here and have this model
            perform target scaling + reverse scaling.

            Note: Has not been fully tested in
            complicated nesting cases, e.g., if Model is
            wrapping a nested Model_Pipeline, this param will
            likely break.

            ::

                default = None

        extra_params : :ref`extra params dict<Extra Params>`, optional

            See :ref:`Extra Params`

            ::

                default = None

        '''

        self.obj = obj
        self.params = params
        self.scope = scope
        self.param_search = param_search
        self.target_scaler = target_scaler
        self.extra_params = extra_params

        self.check_args()

    def view(self, problem_type='regression'):

        from ..pipeline.Models import get_base_model_and_params, AVALIABLE

        proc_str = proc_type_dep_str(self.obj, AVALIABLE, problem_type)
        model, model_params =\
            get_base_model_and_params(proc_str, self.extra_params,
                                      self.params, True)
        model_params = {'__'.join(m.split('__')[1:]): model_params[m]
                        for m in model_params}

        return model, model_params


class Ensemble(Piece):

    _is_model = True

    def __init__(self, obj, models,
                 params=0, scope='all',
                 param_search=None,
                 target_scaler=None,
                 base_model=None,
                 cv_splits=None,
                 is_des=False,
                 single_estimator=False,
                 des_split=.2,
                 n_jobs_type='ensemble',
                 extra_params=None):
        ''' The Ensemble object is valid base
        :class:`Model_Pipeline` piece, designed
        to be passed as input to the `model` parameter
        of :class:`Model_Pipeline`, or
        to its own models parameters.

        This class is used to create a variety ensembled models,
        typically based on
        :class:`Model` pieces.

        Parameters
        ----------
        obj : str
            Each str passed to ensemble refers to a type of ensemble to train,
            based on also the passed input to the `models` parameter,
            and also the
            additional parameters passed when init'ing Ensemble.

            See :ref:`Ensemble Types` to see all
            avaliable options for ensembles.

            Passing custom objects here, while technically possible,
            is not currently full supported.
            That said, there are just certain assumptions that
            the custom object must meet in order to work, specifially,
            they should have simmilar input params to other simmilar existing
            ensembles, e.g., in the case the `single_estimator` is False
            and `needs_split` is also False, then the passed object needs
            to be able to accept an input parameter `estimators`,
            which accepts a list of (str, estimator) tuples.
            Whereas if needs_split is still False,
            but single_estimator is True, then the passed object needs
            to support an init param of `base_estimator`,
            which accepts a single estimator.

        models : :class:`Model`, :class:`Ensemble` or list of
            The `models` parameter is designed to accept any single model-like
            pipeline parameter object, i.e.,
            :class:`Model` or even another :class:`Ensemble`.
            The passed pieces here will be used along with the
            requested ensemble object to
            create the requested ensemble.

            See :class:`Model` for how to create a valid base model(s)
            to pass as input here.

        params : int, str or dict of :ref:`params<Params>`, optional
            `params` sets as associated distribution of hyper-parameters
            for this ensemble object. These parameters will be used only
            in the context of a hyper-parameter search.
            Notably, these `params` refer to the ensemble obj itself,
            params for base `models` should be passed
            accordingly when creating the base models.
            Preset param distributions are listed at :ref:`Ensemble Types`,
            under each of the options for ensemble obj's.

            You can read more about generally about
            hyper-parameter distributions as associated with
            objects at :ref:`Params`.

            ::

                default = 0

        scope : :ref:`valid scope<Scopes>`, optional
            `scope` determines on which subset of features the specified
            ensemble model
            should work on. See :ref:`Scopes` for more
            information on how scopes can
            be specified.

            ::

                default = 'all'

        param_search : :class:`Param_Search`, None, optional
            If None, by default, this will be a base ensemble model.
            Alternatively, by passing a :class:`Param_Search` instance here,
            it specifies that this model should be wrapped in a
            Nevergrad hyper-parameter search object.

            This can be useful to create Model's which have
            a nested hyper-parameter tuning independent from the
            other pipeline steps.

            ::

                default = None

        target_scaler : Scaler, None, optional

            Still somewhat experimental, can pass
            a Scaler object here and have this model
            perform target scaling + reverse scaling.

            scope in the passed scaler is ignored.

            Note: Has not been fully tested in
            complicated nesting cases, e.g., if Model is
            wrapping a nested Model_Pipeline, this param will
            likely break.

            ::

                default = None

        base_model : :class:`Model`, None, optional
            In the case that an ensemble is passed which has
            the parameter `final_estimator` (not base model!),
            for example in the case of stacking,
            then you may pass a Model type
            object here to be used as that final estimator.

            Otherwise, by default this will be left as None,
            and if the requested ensemble has the final_estimator
            parameter, then it will pass None to the object
            (which is typically for setting the default).

            ::

                default = None

        cv_splits : :class:`CV_Splits` or None, optional
            Used for passing custom CV split behavior to
            ensembles which employ splits, e.g., stacking.

            ::

                default = None

        is_des : bool, optional
            `is_des` refers to if the requested ensemble obj requires
            a further training test split in order to train the base ensemble.
            As of right now, If this parameter is True, it means that the
            base ensemble is from the
            `DESlib library <https://deslib.readthedocs.io/en/latest/>`_ .
            Which means
            the base ensemble obj must have a `pool_classifiers`
            init parameter.

            The following `des_split` parameter determines the
            size of the split if
            is_des is True.

            ::

                default = False

        single_estimator : bool, optional
            The parameter `single_estimator` is used to let the
            Ensemble object know if the `models` must be a single estimator.
            This is used for ensemble types
            that requires an init param `base_estimator`.
            In the case that multiple models
            are passed to `models`, but `single_estimator` is True,
            then the models will automatically
            be wrapped in a voting ensemble,
            thus creating one single estimator.

            ::

                default = False

        des_split : float, optional
            If `is_des` is True, then the passed ensemble must be
            fit on a seperate validation set.
            This parameter determines the size
            of the further train/val split on initial
            training set passed to
            the ensemble. Where the size is comptued as
            the a percentage of the total size.

            ::

                default = .2

        n_jobs_type : 'ensemble' or 'models', optional
            Valid options are either 'ensemble' or 'models'.

            This parameter controls how the total n_jobs are distributed, if
            'ensemble', then the n_jobs will be used all in the ensemble object
            and every instance within the sub-models set to n_jobs = 1.
            Alternatively, if passed 'models', then the ensemble
            object will not be multi-processed, i.e.,
            will be set to n_jobs = 1, and the n_jobs will
            be distributed to each base model.

            If you are training a stacking regressor for example
            with n_jobs = 16, and you have 16+ models, then 'ensemble'
            is likely a good choice here. If instead you have only
            3 base models, and one or more of those 3 could benefit from
            a higher n_jobs, then setting n_jobs_type to 'models' might
            give a speed-up.

            ::

                default = 'ensemble'

        extra_params : :ref`extra params dict<Extra Params>`, optional

            See :ref:`Extra Params`

            ::

                default = None
        '''

        self.obj = obj

        # Force passed models if not a list, into a list
        if not isinstance(models, list):
            models = [models]
        self.models = models

        self.params = params
        self.scope = scope
        self.param_search = param_search
        self.target_scaler = target_scaler
        self.base_model = base_model
        self.cv_splits = cv_splits
        self.is_des = is_des
        self.des_split = des_split
        self.single_estimator = single_estimator
        self.n_jobs_type = n_jobs_type
        self.extra_params = extra_params

        self.check_args()

    def check_extra_args(self):

        if isinstance(self.models, list):
            for model in self.models:
                if not hasattr(model, '_is_model'):
                    raise IOError(
                        'All models must be valid Model/Ensemble !')

        else:
            if not hasattr(self.models, '_is_model'):
                raise IOError(
                    'Passed model in models must be a valid Model/Ensemble.')


class Param_Search(Params):

    def __init__(self, search_type='RandomSearch',
                 splits=3, n_repeats=1,
                 cv='default',
                 n_iter=10,
                 scorer='default',
                 weight_scorer=False,
                 mp_context='default',
                 n_jobs='default',
                 dask_ip=None,
                 memmap_X=False,
                 CV='depreciated',
                 _random_state=None,
                 _splits_vals=None,
                 _cv=None,
                 _scorer=None,
                 _n_jobs=None):
        ''' Param_Search is special input object designed to be
        used with :class:`Model_Pipeline`.
        Param_Search defines a hyperparameter search strategy.
        When passed to :class:`Model_Pipeline`,
        its search strategy is applied in the context of any set :ref:`Params`
        within the base pieces.
        Specifically, there must be atleast one parameter search
        somewhere in the object Param_Search is passed!

        All backend hyper-parameter searches make use of the
        <https://github.com/facebookresearch/nevergrad>`_ library.

        Parameters
        ----------
        search_type : str, optional
            The type of nevergrad hyper-parameter search to conduct. See
            :ref:`Search Types<Search Types>` for all avaliable options.
            Also you may further look into
            nevergrad's experimental varients if you so choose,
            this parameter can accept
            those as well.

            ::

                default = 'RandomSearch'

        splits : int, float, str or list of str, optional
            In order to optimize hyper-parameters, some sort of
            internal cross validation must be specified,
            such that combinations of hyper-parameters can be evaluated
            on different data then they were trained on.
            `splits` allows you to specify the base of what CV strategy
            should be used to evaluate every `n_iter` combination
            of hyper-parameters.

            Specifically, options for split are:

            - int
                The number of k-fold splits to conduct. (E.g., 3 for
                3-fold CV split to be conducted at
                every hyper-param evaluation).

            - float
                Must be 0 < `splits` < 1, and defines a
                single train-test like split,
                with `splits` % of the current training data size used
                as a validation set.

            - str
                If a str is passed, then it must correspond to a
                loaded Strat variable. In
                this case, a leave-out-group CV will be used according
                to the value of the
                indicated Strat variable (E.g., a leave-out-site CV scheme).

            - list of str
                If multiple str passed, first determine the overlapping
                unique values from
                their corresponing loaded Strat variables,
                and then use this overlapped
                value to define the leave-out-group CV as described above.

            Also note that `n_repeats` will work with any of these options,
            but say in the case of
            a leave out group CV, would be awfully redundant,
            versus, with a passed float value, very reasonable.

            ::

                default = 3

        n_repeats : int, optional
            Given the base hyper-param search CV defined /
            described in the `splits` param, this
            parameter further controls if the defined train/val splits
            should be repeated (w/ different random
            splits in all cases but the leave-out-group passed str option).

            For example, if `n_repeats` is set to 2, and `splits` is 3,
            then a twice repeated 3-fold CV
            will be performed to evaluate every choice of `n_iter`
            hyper-params.

            ::

                default = 1

        cv : :class:`CV` or 'default', optional
            If left as default 'default', use the class defined CV behavior
            for the splits, otherwise can pass custom behavior.

            ::

                default = 'default'

        n_iter : int, optional
            The number of hyper-parameters to try / budget
            of the underlying search algorithm.
            How well a hyper-parameter search works and how long
            it takes will be very dependent on this parameter
            and the defined internal CV strategy
            (via `splits` and `n_repeats`). In general, if too few
            choices are provided
            the algorithm will likely not select high
            performing hyper-paramers, and alternatively
            if too high a value/budget is
            set, then you may find overfit/non-generalize
            hyper-parameter choices. Other factors which will influence
            the 'right' number of `n_iter` to specify are:

            - `search_type`
                Depending on the underlying search type, it may
                take a bigger or smaller budget
                on average to find a good set of hyper-parameters

            - The dimension of the underlying search space
                If you are only optimizing a few, say 2,
                underlying parameter distributions,
                this will require a far smaller budget then say a really
                high dimensional search space.

            - The CV strategy
                The CV strategy defined via `splits` and `n_repeats`
                may make it easier or harder to overfit when searching
                for hyper-parameters, thus conceptually
                a good choice of CV strategy can serve to increase
                the number `n_iter` you can use
                before overfitting, or conversely a bad choice may limit it.

            - Number of data points / subjects
                Along with CV strategy, the number of data points/subjects
                will greatly influence
                how quickly you overfit, and therefore a
                good choice of `n_iter`.

            Notably, one can always if they have the resources
            simply expiriment with this parameter.

            ::

                default = 10

        scorer : str or 'default', optional
            In order for a set of hyper-parameters to be evaluated,
            a single scorer must be defined.

            For a full list of supported scorers please view the
            scikit-learn docs at:
            https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules

            If left as 'default', assign a reasonable scorer based on the
            passed problem type.

            - 'regression'  : 'explained_variance'
            - 'binary'      : 'matthews'
            - 'categorical' : 'matthews'

            Be careful to make sure to select an appropriate scorer for
            the problem type.

            Only one value of `scorer` may be passed here.

            ::

                default = 'default'

        weight_scorer : bool or 'default', optional
            `weight_scorer` describes if the scorer of interest
            should be weighted by
            the number of subjects within each validation fold.
            So, for example, if
            a leave-out-group CV scheme is specified to `splits`,
            and the groups have
            drastically different numbers of subjects,
            then you may want to consider
            weighting the final average validation metric
            (as computed across in this case
            all groups used by themselves) by the number
            of subjects in each fold.

            ::

                default = False

        mp_context : str, optional
            When a hyper-parameter search is launched, there are different
            ways through python that the multi-processing can be launched
            (assuming n_jobs > 1). Occassionally some choices can lead to
            unexpected errors.

            Choices are:
            - 'default': If 'default' use the BPt mp_context.

            - 'loky': Create and use the python library loky backend.

            - 'fork': Python default fork mp_context

            - 'forkserver': Python default forkserver mp_context

            - 'spawn': Python default spawn mp_context

            ::

                default = 'default'

        n_jobs : int or 'default', optional
            The number of cores to be used for the
            search. In general, this parameter
            should be left as 'default', which will set it
            based on the n_jobs as set in the problem spec-
            and will attempt to automatically change this
            value if say in the context of nesting.

            ::

                default = 'default'

        dask_ip : str or None, optional
            If None, default, then ignore this parameter..

            For experimental Dask support.
            This should be the ip of a created dask
            cluster. A dask Client object will be created
            and passed this ip in order to connect to the cluster.

            ::

                default = None

        memmap_X : bool, optional
            When passing large memory arrays in each parameter search,
            it can be useful as a memory reduction technique to pass
            numpy memmap'ed arrays. This solves an issue where
            the loky backend will not properly pass too large arrays.

            Warning: This can slow down code, and only reduces the actual
            memory consumption of each job by a little bit.

            Note: If passing a dask_ip, this option will be skipped,
            as if using the dask backend, then large X's will be
            pre-scattered instead.

            ::

                default = False

        CV : 'depreciated'
            Switching to passing cv parameter as cv instead of CV.
            Will raise error if anything is passed here.

            ::

                default = 'depreciated'
        '''

        self.search_type = search_type

        self.splits = splits
        self.n_repeats = n_repeats
        self.cv = cv
        self.n_iter = n_iter
        self.scorer = scorer
        self.weight_scorer = weight_scorer
        self.mp_context = mp_context
        self.n_jobs = n_jobs
        self.dask_ip = dask_ip
        self.memmap_X = memmap_X

        self._random_state = _random_state
        self._splits_vals = _splits_vals
        self._cv = _cv
        self._scorer = _scorer
        self._n_jobs = _n_jobs

        if CV != 'depreciated':
            raise RuntimeError('Pass as cv instead of CV!')

        self.CV = CV

        self.check_args()

    def set_random_state(self, random_state):
        self._random_state = random_state

    def set_n_jobs(self, n_jobs):

        if self.n_jobs == 'default':
            self._n_jobs = n_jobs
        else:
            self._n_jobs = self.n_jobs

    def set_scorer(self, problem_type):
        self._scorer =\
            process_scorers(self.scorer,
                            problem_type)[2]

    def set_cv(self, cv):
        self._cv = cv

    def set_split_vals(self, vals):
        self._splits_vals = vals

    def check_args(self):

        if isinstance(self.scorer, list):
            raise IOError('scorer within Param Search cannot be list-like')
        if isinstance(self.weight_scorer, list):
            raise IOError(
                'weight_scorer within Param Search cannot be list-like')


class Shap_Params(Params):

    def __init__(self,
                 avg_abs=False,
                 linear_feature_perturbation="interventional",
                 linear_nsamples=1000,
                 tree_feature_perturbation='tree_path_dependent',
                 tree_model_output='raw',
                 tree_tree_limit=None,
                 kernel_nkmean=20,
                 kernel_link='default',
                 kernel_nsamples='auto',
                 kernel_l1_reg='auto'):
        '''
        There are a number of parameters associated
        with using shap to determine
        feature importance. The best way to understand Shap is almost certainly through
        their documentation directly, `Shap Docs <https://shap.readthedocs.io/en/latest/>`_
        Just note when using Shap within BPt to pay attention to the version on the shap
        documentation vs. what is currently supported within BPt.

        Broadly, shap feature importance params are split up into if they are used
        to explain linear models, tree based models or an abitrary model (kernel).
        Be warned, kernel shap generally takes a long time to run.

        The type of shap to use will be automatically computed based on the Model to
        explain, but the parameter below are still split up by type, i.e., with the 
        type prended before the parameter name.

        (A number of the params below are copy-pasting from the relevant Shap description)

        Parameters
        ----------
        avg_abs : bool, optional 
            This parameter is considered regardless of the underlying shap model. If
            set to True, then when computing global feature importance from the
            initially computed local shap feature importance the average of the absolute value 
            will be taken! When set to False, the average value will be taken. One might want to
            set this to True if only concerned with the magnitude of feature importance, rather
            than the sign.

            ::

                default = False

        linear_feature_perturbation : {"interventional", "correlation_dependent"}, optional
            Only used with linear base models.
            There are two ways we might want to compute SHAP values, either the full conditional SHAP
            values or the interventional SHAP values. For interventional SHAP values we break any
            dependence structure between features in the model and so uncover how the model would behave if we
            intervened and changed some of the inputs. For the full conditional SHAP values we respect
            the correlations among the input features, so if the model depends on one input but that
            input is correlated with another input, then both get some credit for the model's behavior. The
            interventional option stays "true to the model" meaning it will only give credit to features that are
            actually used by the model, while the correlation option stays "true to the data" in the sense that
            it only considers how the model would behave when respecting the correlations in the input data.
            For sparse case only interventional option is supported.

            ::

                default = "interventional"

        linear_nsamples : int, optional
            Only used with linear base models.
            Number of samples to use when estimating the transformation matrix used to account for
            feature correlations.

            ::

                default = 1000

        tree_feature_perturbation : {"interventional", "tree_path_dependent"}, optional
            Only used with tree based models.
            Since SHAP values rely on conditional expectations
            we need to decide how to handle correlated
            (or otherwise dependent) input features.
            The "interventional" approach breaks the dependencies between
            features according to the rules dictated by casual
            inference (Janzing et al. 2019). Note that the
            "interventional" option requires a background dataset
            and its runtime scales linearly with the size
            of the background dataset you use. Anywhere from 100 to 1000
            random background samples are good
            sizes to use. The "tree_path_dependent" approach is to just follow
            the trees and use the number
            of training examples that went down each leaf to represent
            the background distribution. This approach
            does not require a background dataset and so is used
            by default when no background dataset is provided.

            ::

                default = "tree_path_dependent"

        tree_model_output : {"raw", "probability", "log_loss"} optional
            Only used with tree based models.
            What output of the model should be explained. If "raw" then we explain the raw output of the
            trees, which varies by model. For regression models "raw" is the standard output, for binary
            classification in XGBoost this is the log odds ratio. If model_output is the name of a supported
            prediction method on the model object then we explain the output of that model method name.
            For example model_output="predict_proba" explains the result of calling model.predict_proba.
            If "probability" then we explain the output of the model transformed into probability space
            (note that this means the SHAP values now sum to the probability output of the model). If "logloss"
            then we explain the log base e of the model loss function, so that the SHAP values sum up to the
            log loss of the model for each sample. This is helpful for breaking down model performance by feature.
            Currently the probability and logloss options are only supported when feature_dependence="independent".

            ::

                default = 'raw'

        tree_tree_limit : None or int, optional
            Only used with tree based models.
            Limit the number of trees used by the model. By default None means no use the limit of the
            original model, and -1 means no limit.

            ::

                default = None

        kernel_n_kmeans : int or None, optional
            Used when the underlying model is not linear or tree based.
            This setting offers a speed up to the kernel estimator by replacing
            the background dataset with a kmeans representation of the data.
            Set this option to None in order to use the full dataset directly,
            otherwise the int passed will the determine 'k' in the kmeans algorithm.

            ::

                default = 20

        kernel_link : {"identity", "logit", "default"}
            Used when the underlying model is not linear or tree based.

            A generalized linear model link to connect the feature importance values to the model
            output. Since the feature importance values, phi, sum up to the model output, it often makes
            sense to connect them to the ouput with a link function where link(outout) = sum(phi).
            If the model output is a probability then the LogitLink link function makes the feature
            importance values have log-odds units.

            If 'default', set to using 'logic' for binary + categorical problems, along
            with explaining predict_proba from the model, and 'identity' and explaining
            predict for regression.

            ::

                default = 'default'

        kernel_nsamples : "auto" or int, optional
            Used when the underlying model is not linear or tree based.
            Number of times to re-evaluate the model when explaining each prediction. More samples
            lead to lower variance estimates of the SHAP values. The "auto" setting uses
            `nsamples = 2 * X.shape[1] + 2048`.

            ::

                default = 'auto'

        kernel_l1_reg : "num_features(int)", "auto", "aic", "bic", or float, optional
            Used when the underlying model is not linear or tree based.
            The l1 regularization to use for feature selection (the estimation procedure is based on
            a debiased lasso). The auto option currently uses "aic" when less that 20% of the possible sample
            space is enumerated, otherwise it uses no regularization. THE BEHAVIOR OF "auto" WILL CHANGE
            in a future version to be based on num_features instead of AIC.
            The "aic" and "bic" options use the AIC and BIC rules for regularization.
            Using "num_features(int)" selects a fix number of top features. Passing a float directly sets the
            "alpha" parameter of the sklearn.linear_model.Lasso model used for feature selection.

            ::

                default = 'auto'

        '''

        try:
            import shap
        except ImportError:
            raise ImportError('You must have shap installed to use shap')

        self.avg_abs = avg_abs
        self.linear_feature_perturbation = linear_feature_perturbation
        self.linear_nsamples = linear_nsamples
        self.tree_feature_perturbation = tree_feature_perturbation
        self.tree_model_output = tree_model_output
        self.tree_tree_limit = tree_tree_limit
        self.kernel_nkmean = kernel_nkmean
        self.kernel_link = kernel_link
        self.kernel_nsamples = kernel_nsamples
        self.kernel_l1_reg = kernel_l1_reg


class Feat_Importance(Params):

    def __init__(self, obj, scorer='default',
                 shap_params='default', n_perm=10,
                 inverse_global=False, inverse_local=False):
        '''
        There are a number of options for creating Feature Importances in BPt.
        See :ref:`Feat Importances` to learn more about
        feature importances generally.
        The way this object works, is that you can a type
        of feature importance, and then
        its relevant parameters. This object is designed
        to passed directly to
        :class:`Model_Pipeline`.

        Parameters
        ----------
        obj : str
            `obj` is the str indiciator for which feature importance to use.
            See :ref:`Feat Importances` for what options are avaliable.

        scorer : str or 'default', optional

            If a permutation based feature importance is being used, then a scorer is
            required.

            For a full list of supported scorers please view the scikit-learn docs at:
            https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules

            If left as 'default', assign a reasonable scorer based on the
            passed problem type.

            - 'regression'  : 'explained_variance'
            - 'binary'      : 'matthews'
            - 'categorical' : 'matthews'

            ::

                default = 'default'

        shap_params : :class:`Shap_Params` or 'default', optional
            If a shap based feature importance is used, it is neccicary to define
            a number of relevant parameters for how the importances should be calculated.
            See :class:`Shap_Params` for what these parameters are.

            If 'default' is passed, then shap_params will be set to either the default values of
            :class:`Shap_Params` if shap feature importances are being used, or None if not.

            ::

                default = 'default'

        n_perm : int, optional
            If a permutation based feature importance method is selected, then
            it is neccicary to indicate how many random permutations each feature 
            should be permuted.

            ::

                default = 10

        inverse_global : bool
            Warning: This feature, along with inverse_local, is still
            expirimental.

            If there are any loaders, or transformers specified
            in the Model_Pipeline,
            then feature importance becomes slightly trickier.
            For example, if you have
            a PCA transformer, and what to calculate averaged feature importance
            across 3-folds, there is no gaurentee 'pca feature 1' is the same from one
            fold to the next. In this case, if set to True, global feature
            importances will be inverse_transformed back into their original feature
            space - per fold. Note: this will only work if all transformers / loaders
            have an implemented reverse_transform function, if one does not for transformer, then
            it will just return 0 for that feature. For a loader w/o, then it will return 
            'No inverse_transform'.

            There are also other cases where this might be a bad idea, for example
            if you are using one hot encoders in your transformers then trying to 
            reverse_transform
            feature importances will yield nonsense (NaN's).

            ::

                default = False

        inverse_local : bool
            Same as inverse_global, but for local feature importances.
            By default this is set to False, as it is
            more memory and computationally expensive to
            inverse_transform this case.

            ::

                default = False

        '''

        self.obj = obj
        self.scorer = scorer

        if shap_params == 'default':

            if 'shap' in self.obj:
                shap_params = Shap_Params()
            else:
                shap_params = None

        self.shap_params = shap_params
        self.n_perm = n_perm
        self.inverse_global = inverse_global
        self.inverse_local = inverse_local

        # For compatibility
        self.params = 0


class Model_Pipeline(Params):

    def __init__(self,
                 loaders=None, imputers='default',
                 scalers=None, transformers=None,
                 feat_selectors=None,
                 model='default',
                 param_search=None,
                 cache=None, n_jobs='default',
                 feat_importances='depreciated'):
        ''' Model_Pipeline is defined as essentially a wrapper around
        all of the explicit modelling pipeline parameters. This object is
        used as input to
        :func:`Evaluate <BPt.BPt_ML.Evaluate>`
        and :func:`Test <BPt.BPt_ML.Test>`

        The ordering of the parameters listed below defines the pre-set
        order in which these Pipeline pieces are composed
        (params up to model, param_search is not an ordered pipeline piece).
        For more flexibility, one can always use custom defined objects,
        or even pass custom defined
        pipelines directly to model
        (i.e., in the case where you have a specific pipeline you want to use
        already defined, but say just want to use the loaders from BPt).

        Parameters
        ----------
        loaders : :class:`Loader`, list of or None, optional
            Each :class:`Loader` refers to transformations
            which operate on loaded Data_Files
            (See :func:`Load_Data_Files <BPt.BPt_ML.Load_Data_Files>`).
            See :class:`Loader`
            explcitly for more information on how to create a valid object,
            with relevant params and scope.

            In the case that a list of Loaders is passed to loaders,
            if a native python list, then passed loaders will
            be applied sequentially (likely each
            passed loader given a seperate scope, as the output from
            one loader cannot be input
            to another- note to create actual sequential loader steps,
            look into using the
            :class:`Pipe` wrapper
            argument when creating a single :class:`Loader` obj).

            Passed loaders can also be wrapped in a
            :class:`Select` wrapper, e.g., as either

            .. code-block::

                # Just passing select
                loaders = Select([Loader(...), Loader(...)])

                # Or nested
                loaders = [Loader(...), Select([Loader(...), Loader(...)])]

            In this way, most of the pipeline objects can accept lists,
            or nested lists with
            param wrapped, not just loaders!

            .. code-block::

                default = None

        imputers : :class:`Imputer`, list of or None, optional
            If there is any missing data (NaN's) that have been kept
            within data or covars, then an imputation strategy must be
            defined! This param controls what kind of
            imputation strategy to use.

            Each :class:`Imputer` contains information
            around which imputation
            strategy to use, what scope it is applied
            to (in this case only 'float' vs. 'cat'),
            and other relevant base parameters
            (i.e., a base model if an iterative imputer is selected).

            In the case that a list of :class:`Imputer` are passed,
            they will be applied sequentially,
            though note that unless custom scopes
            are employed, at most passing only an imputer for float data and
            an imputer for categorical data makes sense.
            You may also use input wrapper, like :class:`Select`.

            In the case that no NaN data is passed, but imputers is not None,
            it will simply be set to None.

            ::

                default = [Imputer('mean', scope='float'),
                           Imputer('median', scope='cat')]

        scalers : :class:`Scaler`, list of or None, optional
            Each :class:`Scaler` refers to any potential data scaling where a
            transformation on the data
            (without access to the target variable) is
            computed, and the number of features or
            data points does not change.
            Each :class:`Scaler` object contains information
            about the base object, what
            scope it should be applied to, and saved param
            distributions if relevant.

            As with other pipeline params, scalers can
            accept a list of :class:`Scaler` objects,
            in order to apply sequential transformations
            (or again in the case where each object has a seperate scope,
            these are essentially two different streams of transformations,
            vs. when two Scalers with the same scope are passed,
            the output from one
            is passed as input to the next). Likewise,
            you may also use valid input wrappers,
            e.g., :class:`Select`.

            By default no scaler is used, though it is reccomended.

            ::

                default = None

        transformers : :class:`Transformer`, list of or None, optional
            Each :class:`Transformer` defines a type of transformation to
            the data that changes the number of features in perhaps non-deterministic
            or not simply removal (i.e., different from feat_selectors), for example
            applying a PCA, where both the number of features change, but
            also the new features do not 1:1 correspond to the original
            features. See :class:`Transformer` for more information.

            Transformers can be composed sequentially with list or special
            input type wrappers, the same as other objects.

            ::

                default = None

        feat_selectors : :class:`Feat_Selector`, list of or None, optional
            Each :class:`Feat_Selector` refers to an optional feature selection stage
            of the Pipeline. See :class:`Feat_Selector` for specific options.

            Input can be composed in a list, to apply feature selection sequentially,
            or with special Input Type wrapper, e.g., :class:`Select`.

            ::

                default = None

        model : :class:`Model`, :class:`Ensemble`, optional
            model accepts one input of type
            :class:`Model` or :class:`Ensemble`. Though,
            while it cannot accept a list (i.e., no sequential behavior), you
            may still pass Input Type wrapper like :class:`Select` to perform
            model selection via param search.

            See :class:`Model` for more information on how to specify a single
            model to BPt, and :class:`Ensemble` for information on how
            to build an ensemble of models.

            Note: You must have provide a model, there is
            no option for None. Instead
            default behavior is to use a ridge regression.

            ::

                default = Model('ridge')

        param_search : :class:`Param_Search` or None, optional
            :class:`Param_Search` can be provided in order
            to specify a corresponding
            hyperparameter search for the provided pipeline pieces.
            When defining each
            piece, you may set hyperparameter distributions for that piece.
            If param search
            is None, these distribution will be essentially ignored,
            but if :class:`Param_Search`
            is passed here, then they will be used along with the
            strategy defined in the passed
            :class:`Param_Search` to conduct a nested hyper-param search.

            Note: If using input wrapper types like :class:`Select`,
            then a param search must be passed!

            ::

                default = None

        cache : str or None, optional
            Warning: using cache with a Transformer or Loader
            is currently broken!

            The base scikit-learn Pipeline, upon which the
            BPt Pipeline extends,
            allows for the caching of fitted transformers -
            which in this context means all
            steps except for the model. If this behavior is desired
            (in the cases where a non-model step take a long time to fit),
            then a str indicating a directory where
            the cache should be stored can be passed to cache.
            If this directory
            does not aready exist, it will be created.

            Note: cache_dr's are not automatically removed,
            and while different calls
            to Evaluate or Test may benefit from overlapping cached steps
            - the size of the
            cache can also grow pretty quickly, so you may need
            to manually monitor the size
            of the cache and perform manual deletions when it
            grows too big depending on your
            storage resources.

            ::

                default = None

        n_jobs : int or 'default', optional
            The number of cores to be used with this pipeline.
            In general, this parameter
            should be left as 'default', which will set it
            based on the n_jobs as set in the problem spec-
            and will attempt to automatically change this
            value if say in the context of nesting.

            ::

                default = 'default'

        feat_importances : depreciated
            Feature importances in a past version of BPt were
            specified via this Model Pipeline object.
            Now they should be provided to either
            :func:`Evaluate <BPt.BPt_ML.Evaluate>`
            and :func:`Test <BPt.BPt_ML.Test>`

        ::

            default = 'depreciated'

        '''

        if isinstance(loaders, str):
            loaders = Loader(loaders)
        self.loaders = loaders

        if imputers == 'default':
            imputers = [Imputer('mean', scope='float'),
                        Imputer('median', scope='cat')]
            print('Passed default imputers, setting to:', imputers)
        elif isinstance(imputers, str):
            imputers = Imputer(imputers)
        self.imputers = imputers

        if isinstance(scalers, str):
            scalers = Scaler(scalers)
        self.scalers = scalers

        if isinstance(transformers, str):
            transformers = Transformer(transformers)
        self.transformers = transformers

        if isinstance(feat_selectors, str):
            feat_selectors = Feat_Selector(feat_selectors)
        self.feat_selectors = feat_selectors

        if model == 'default':
            model = Model('ridge')
            print('Passed default model, setting to:', print(model))
        elif isinstance(model, str):
            model = Model(model)
        self.model = model

        self.param_search = param_search
        self.cache = cache
        self.n_jobs = n_jobs

        if feat_importances != 'depreciated':
            print('Warning: Passing feature importances have been moved ',
                  'to the Evaluate and Test functions!')

        # Regardless save the value to avoid sklearn warnings
        self.feat_importances = feat_importances

        # Perform all preproc on input which can be run
        # more then once, these are essentially checks on the input
        self._proc_checks()

    def _proc_all_pieces(self, func):
        '''Helper to proc all pieces with a function
        that accepts a list of pieces'''

        for param_name in ORDERED_NAMES:
            params = getattr(self, param_name)

            if params is None:
                new_params = params

            elif isinstance(params, list):
                new_params = func(params)

            # If not list-
            else:
                new_params = func([params])
                if isinstance(new_params, list) and len(new_params) == 1:
                    new_params = new_params[0]

            setattr(self, param_name, new_params)

    def _proc_duplicates(self, params):

        if isinstance(params, list):
            new_params = deepcopy(params)
            del new_params[:]

            if len(params) == 1:
                return self._proc_duplicates(params[0])

            for p in params:
                new_params.append(self._proc_duplicates(p))

            return new_params

        elif hasattr(params, 'scope'):
            scopes = getattr(params, 'scope')
            new_params = []

            if is_duplicate(scopes):
                for scope in scopes:
                    new = deepcopy(params)
                    new.scope = scope
                    new_params.append(new)

                return new_params

        return params

    def _proc_input(self, params):

        for p in params:
            if isinstance(p, list):
                self._proc_input(p)
            else:
                proc_all(p)

        return params

    def _check_args(self, params):

        for p in params:
            if isinstance(p, list):
                self._check_args(p)
            else:
                p.check_args()

        return params

    def check_imputer(self, data):

        # If no NaN, no imputer
        if not pd.isnull(data).any().any():
            self.imputers = None

    def check_samplers(self, func):
        if self.samplers is not None:
            if isinstance(self.samplers, list):
                for s in self.samplers:
                    s.add_strat_u_name(func)
            else:
                self.samplers.add_strat_u_name(func)

    def preproc(self, n_jobs):

        # Base proc checks
        self._proc_checks()

        # Store n_jobs
        self.set_n_jobs(n_jobs)

    def _proc_checks(self):

        try:
            self.model = self.models
            print('Passed models, set as model')
        except AttributeError:
            pass

        try:
            self.transformers = self.transformer
            print('Passed transformer, set as transformers')
        except AttributeError:
            pass

        try:
            self.imputers = self.imputer
            print('Passed imputer, set as imputers')
        except AttributeError:
            pass

        try:
            self.feat_selectors = self.feat_selector
            print('Passed feat_selector, set as feat_selectors')
        except AttributeError:
            pass

        try:
            self.scalers = self.scaler
            print('Passed scaler, set as scalers')
        except AttributeError:
            pass

        try:
            self.loaders = self.loader
            print('Passed loader, set as loaders')
        except AttributeError:
            pass

        to_check = ['ensemble', 'scorer', 'metric',
                    'feature_selector', 'feature_selectors']
        for p in to_check:
            if hasattr(self, p):
                print('Warning: Model_Pipeline user set param', p,
                      ' was set,',
                      'but will have no effect as it is not',
                      ' a valid parameter!')

        # Check for duplicate scopes
        self._proc_all_pieces(self._proc_duplicates)

        # Proc input
        self._proc_all_pieces(self._proc_input)
        proc_all(self.param_search)

        # Double check input args in case something changed
        self._proc_all_pieces(self._check_args)

        # Proc param search if not None
        if self.param_search is not None:
            self.param_search.check_args()

    def set_n_jobs(self, n_jobs):

        if self.n_jobs == 'default':

            # If no param search, each objs base n_jobs is
            # the passed n_jobs
            if self.param_search is None:
                self.n_jobs = n_jobs

            # Otherwise, base jobs are 1
            else:
                self.n_jobs = 1

    def get_ordered_pipeline_params(self):

        # Conv all to list & return in order as a deep copy
        return deepcopy([conv_to_list(getattr(self, piece_name))
                         for piece_name in ORDERED_NAMES])

    def get_indent(self, indent):

        if indent is None:
            return ''
        else:
            ind = 0
            for s in self._p_stack:
                if s == ']':
                    ind += 1
                elif s == '])':
                    ind += 8

            return ' ' * ind

    def params_print(self, params, indent, _print=print, end='\n'):

        if not isinstance(params, list):
            _print(self.get_indent(indent), params, sep='', end=end)
            return

        elif len(params) == 1:
            self.params_print(params[0], 0, _print=_print)
            return

        elif not is_special(params):

            _print(self.get_indent(indent), '[', sep='', end='')
            self._p_stack.append(']')
            self.params_print(params[0], None, _print=_print, end='')
            _print(',', sep='')

        elif is_select(params):
            _print(self.get_indent(indent), 'Select([', sep='', end='')
            self._p_stack.append('])')
            self.params_print(params[0], None, _print=_print, end='')
            _print(',', sep='')

        for param in params[1:-1]:
            self.params_print(param, 0, _print=_print, end='')
            _print(',', sep='')

        self.params_print(params[-1], 0, _print=_print, end='')
        _print(self._p_stack.pop(), end=end)

        return

    def print_all(self, _print=print):

        self._p_stack = []

        _print('Model_Pipeline')
        _print('--------------')

        pipeline_params = self.get_ordered_pipeline_params()
        for name, params in zip(ORDERED_NAMES, pipeline_params):

            if params is not None:
                _print(name + '=\\')
                self.params_print(params, 0, _print=_print)
                _print()

        _print('param_search=\\')
        _print(self.param_search)
        _print()
        _print()

        if self.cache is not None:
            _print('cache =', self.cache)
        _print()


class Problem_Spec(Params):

    def __init__(self, problem_type='default',
                 target=0, scorer='default', weight_scorer=False,
                 scope='all', subjects='all',
                 n_jobs='default',
                 random_state='default'):
        '''Problem Spec is defined as an object of params encapsulating the set of
        parameters shared by modelling class functions
        :func:`Evaluate <BPt.BPt_ML.Evaluate>` and :func:`Test <BPt.BPt_ML.Test>`

        Parameters
        ----------
        problem_type : str or 'default', optional
            This parameter controls what type of machine learning
            should be conducted. As either a regression, or classification
            where 'categorical' represents a special case of binary classification,
            where typically a binary classifier is trained on each class.

            - 'default'
                Determine the problem type based on how
                the requested target variable is loaded.

            - 'regression', 'f' or 'float'
                For ML on float/continuous target data.

            - 'binary' or 'b'
                For ML on binary target data.

            - 'categorical' or 'c'
                For ML on categorical target data, as multiclass.

            ::

                default = 'default'

        target : int or str, optional
            The loaded target in which to use during modelling.
            This can be the int index (as assigned by order loaded in,
            e.g., first target loaded is 0, then the next is 1),
            or the name of the target column.
            If only one target is loaded, just leave as default of 0.

            ::

                default = 0

        scorer : str or list, optional
            Indicator str for which scorer(s) to use when calculating
            average validation score in Evaluate, or Test set score in Test.

            A list of str's can be passed as well, in this case, scores for
            all of the requested scorers will be calculated and returned.

            Note: If using a Param_Search, the Param_Search object has a
            scorer parameter as well. This scorer describes the scorer optimized
            in a parameter search.

            For a full list of supported scorers please view the scikit-learn docs at:
            https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules

            If left as 'default', assign a reasonable scorer based on the
            passed problem type.

            - 'regression'  : ['explained_variance', 'neg_mean_squared_error']
            - 'binary'      : ['matthews', 'roc_auc', 'balanced_accuracy']
            - 'categorical' : ['matthews', 'roc_auc_ovr', 'balanced_accuracy']

            ::

                default = 'default'

        weight_scorer : bool, list of, optional
            If True, then the scorer of interest will be weighted within
            each repeated fold by the number of subjects in that validation set.
            This parameter only typically makes sense for custom split behavior where 
            validation folds may end up with differing sizes.
            When default CV schemes are employed, there is likely no point in
            applying this weighting, as the validation folds will have simmilar sizes.

            If you are passing mutiple scorers, then you can also pass a 
            list of values for weight_scorer, with each value set as boolean True or False,
            specifying if the corresponding scorer by index should be weighted or not.

            ::

                default = False

        scope : key str or Scope obj, optional
            This parameter allows the user to optionally
            run an expiriment with just a subset of the loaded features
            / columns.

            See :ref:`Scopes` for a more detailed explained / guide on how scopes
            are defined and used within BPt.

            ::

                default = 'all'

        subjects : str, array-like or Value_Subset, optional
            This parameter allows the user to optionally run Evaluate or Test
            with just a subset of the loaded subjects. It is notably distinct
            from the `train_subjects`, and `test_subjects` parameters directly
            avaliable to Evaluate and Test, as those parameters typically refer
            to train/test splits. Specifically, any value specified for this 
            subjects parameter will be applied AFTER selecting the relevant train
            or test subset.

            One use case for this parameter might be specifying subjects of just one
            sex, where you would still want the same training set for example, but just want
            to test sex specific models.

            If set to 'all' (as is by default), all avaliable subjects will be
            used.

            `subjects` can accept either a specific array of subjects,
            or even a loc of a text file (formatted one subject per line) in
            which to read from.

            A special wrapper, Value_Subset, can also be used to specify more specific,
            specifically value specific, subsets of subjects to use.
            See :class:`Value_Subset` for how this input wrapper can be used.

            ::

                default = 'all'

        n_jobs : int, or 'default'
            n_jobs are employed witin the context of a call to Evaluate or Test. 
            If left as default, the class wide BPt value will be used.

            In general, the way n_jobs are propegated to the different pipeline
            pieces on the backend is that, if there is a parameter search, the base
            ML pipeline will all be set to use 1 job, and the n_jobs budget will be used
            to train pipelines in parellel to explore different params. Otherwise, if no
            param search, n_jobs will be used for each piece individually, though some
            might not support it.

            ::

                default = 'default'

        random_state : int, RandomState instance, None or 'default', optional
            Random state, either as int for a specific seed, or if None then
            the random seed is set by np.random.

            This parameter is used to ensure replicability of expirements (wherever possible!).
            In some cases even with a random seed, depending on the pipeline pieces being used,
            if any have a component that occassionally yields different results, even with the same
            random seed, e.g., some model optimizations, then you might still not get
            exact replicicability.

            If 'default', use the saved class value.
            (Defined in :class:`ML <BPt.BPt_ML>`)

            ::

                default = 'default'

        '''

        self.problem_type = problem_type
        self.target = target
        self.scorer = scorer
        self.weight_scorer = weight_scorer
        self.scope = scope
        self.subjects = subjects
        self.n_jobs = n_jobs
        self.random_state = random_state

        self._final_subjects = None

        self._proc_checks()

    def _proc_checks(self):
        proc_all(self)

    def set_final_subjects(self, final_subjects):
        self._final_subjects = final_subjects

    def get_model_spec(self):

        return {'problem_type': self.problem_type,
                'random_state': self.random_state}

    def print_all(self, _print=print):

        _print('Problem_Spec')
        _print('------------')
        _print('problem_type =', self.problem_type)
        _print('target =', self.target)
        _print('scorer =', self.scorer)
        _print('weight_scorer =', self.weight_scorer)
        _print('scope =', self.scope)

        if is_value_subset(self.subjects) or len(self.subjects) < 50:
            _print('subjects =', self.subjects)
    
        if self._final_subjects is not None:
            _print('len(subjects) =', len(self._final_subjects),
                   '(before overlap w/ train/test subjects)')
        _print('n_jobs =', self.n_jobs)
        _print('random_state =', self.random_state)
        _print()


class CV(Params):

    def __init__(self, groups=None, stratify=None,
                 train_only_loc=None, train_only_subjects=None):
        ''' This objects is used to encapsulate a set of parameters
        for a CV strategy.

        Parameters
        ----------
        groups : str, list or None, optional
            In the case of str input, will assume the str to refer
            to a column key within the loaded strat data,
            and will assign it as a value to preserve groups by
            during any train/test or K-fold splits.
            If a list is passed, then each element should be a str,
            and they will be combined into all unique
            combinations of the elements of the list.

        ::

            default = None

        stratify : str, list or None, optional
            In the case of str input, will assume the str to refer
            to a column key within the loaded strat data, or a loaded target col.,
            and will assign it as a value to preserve
            distribution of groups by during any train/test or K-fold splits.
            If a list is passed, then each element should be a str,
            and they will be combined into all unique combinations of
            the elements of the list.

            Any target_cols passed must be categorical or binary, and cannot be
            float. Though you can consider loading in a float target as a strat,
            which will apply a specific k_bins, and then be valid here.

            In the case that you have a loaded strat val with the same name
            as your target, you can distinguish between the two by passing
            either the raw name, e.g., if they are both loaded as 'Sex',
            passing just 'Sex', will try to use the loaded target. If instead
            you want to use your loaded strat val with the same name - you have
            to pass 'Sex' + self.self.strat_u_name (by default this is '_Strat').

            ::

                default = None

        train_only_loc : str, Path or None, optional
            Location of a file to load in train_only subjects,
            where any subject loaded as train_only will be assigned to
            every training fold, and never to a testing fold.
            This file should be formatted as one subject per line.

            You can load from a loc and pass subjects, the subjects
            from each source will be merged.

            This parameter is compatible with groups / stratify.

            ::

                default = None

        train_only_subjects : set, array-like, 'nan', or None, optional
            An explicit list or array-like of train_only subjects, where
            any subject loaded as train_only will be assigned to every training
            fold, and never to a testing fold.

            You can also optionally specify 'nan' as input, which
            will add all subjects with any NaN data to train only.

            If you want to add both all the NaN subjects and custom
            subjects, call :func:`Get_Nan_Subjects` to get all NaN subjects,
            and then merge them yourself with any you want to pass.

            You can load from a loc and pass subjects, the subjects
            from each source will be merged.

            This parameter is compatible with groups / stratify.

            ::

                default = None
        '''

        self.groups = groups
        self.stratify = stratify
        self.train_only_loc = train_only_loc
        self.train_only_subjects = train_only_subjects


class CV_Split(Params):

    def __init__(self, cv='default', split=.2,
                 _cv=None, _random_state=None):

        self.cv = cv
        self.split = split
        self._cv = _cv
        self._random_state = _random_state

    def setup(self, cv, random_state):

        self._cv = cv
        self._random_state = random_state

    def get_split(self, train_data_index):

        return self._cv.train_test_split(subjects=train_data_index,
                                         test_size=self.split,
                                         random_state=self._random_state,
                                         return_index=True)

class CV_Splits(Params):

    def __init__(self, cv='default', splits=3, n_repeats=1,
                 _cv=None, _random_state=None, _splits_vals=None):
        ''' This object is used to wrap around a CV strategy at a higher level.

        Parameters
        ----------
        cv: 'default' or :class:`CV`, optional
            If left as default 'default', use the class defined CV behavior
            for the splits, otherwise can pass custom behavior.

        splits : int, float, str or list of str, optional
            `splits` allows you to specify the base of what CV strategy
            should be used.

            Specifically, options for split are:

            - int
                The number of k-fold splits to conduct. (E.g., 3 for
                3-fold CV split to be conducted at
                every hyper-param evaluation).

            - float
                Must be 0 < `splits` < 1, and defines a
                single train-test like split,
                with `splits` % of the current training data size used
                as a validation set.

            - str
                If a str is passed, then it must correspond to a
                loaded Strat variable. In
                this case, a leave-out-group CV will be used according
                to the value of the
                indicated Strat variable (E.g., a leave-out-site CV scheme).

            - list of str
                If multiple str passed, first determine the overlapping
                unique values from
                their corresponing loaded Strat variables,
                and then use this overlapped
                value to define the leave-out-group CV as described above.

            `n_repeats` is designed to work with any of these choices.

            ::

                default = 3

        n_repeats : int, optional
            The number of times to repeat the defined strategy
            as defined in `splits`.

            For example, if `n_repeats` is set to 2, and `splits` is 3,
            then a twice repeated 3-fold CV will be performed

            ::

                default = 1
        '''

        self.cv = cv
        self.splits = splits
        self.n_repeats = n_repeats
        self._cv = _cv
        self._random_state = _random_state
        self._splits_vals = _splits_vals

    def setup(self, cv, split_vals, random_state):

        self._cv = cv
        self._splits_vals = split_vals
        self._random_state = random_state

    def get_splits(self, train_data_index):

        return self._cv.get_cv(train_data_index,
                               self.splits,
                               self.n_repeats,
                               self._splits_vals,
                               self._random_state,
                               return_index='both')