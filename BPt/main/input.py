from copy import deepcopy
from sklearn.base import BaseEstimator
from ..helpers.ML_Helpers import (conv_to_list, proc_input)

from ..helpers.VARS import ORDERED_NAMES
from ..main.input_operations import (is_duplicate, is_pipe, is_select,
                                     is_special, is_value_subset)
from ..default.options.scorers import process_scorer
from .CV import get_bpt_cv

import warnings


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

        if isinstance(obj, (list, tuple)):
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
                          'set it within ModelPipeline to None,',
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

        # Skip for now
        return

        extra_params = getattr(self, 'extra_params')
        if extra_params is None:
            return

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

    def __init__(self, obj, behav='single', params=0, scope='data file',
                 cache_loc=None,
                 fix_n_wrapper_jobs=False, **extra_params):
        ''' Loader refers to transformations which operate on Data_Files.
        See: :func:`add_data_files <Dataset.add_data_files>`
        in the :ref:`Dataset` class.
        They in essence take in saved file locations, and after some series
        of transformations pass on compatible features.

        Importantly, the Loader object can operate in two ways. Either
        the Loader can define operations which are computed on
        single files independently, or load and pass on data
        to the defined `obj` as a list, where each element of
        the list is a subject's data. See parameter behav.

        Parameters
        ----------
        obj : str, custom obj or :class:`Pipe`
            `obj` selects the base loader object to use, this can be either
            a str corresponding to
            one of the preset loaders found at :ref:`Loaders`.
            Beyond pre-defined loaders, users can
            pass in custom objects as long as they have functions
            corresponding to the correct behavior.

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

        behav : {'single', 'all'}, optional
            The Loader object can operate under two different
            behaviors, corresponding to operations which can
            be done for each subject's Data File independently ('single')
            and operations which must be done using information
            from all train subject's Data Files ('all').

            'single' is the default behavior, if requested
            then the Loader will load each subject's Data File
            seperately and apply the passed `obj` fit_transform.
            The benefit of this method in contrast to 'all' is
            that only one subject's full raw data needs to be
            loaded at once, whereas with all, you must have enough
            avaliable memory to load all of the current training or
            validation subject's raw data at one. Likewise 'single'
            allows for caching fit_transform operations for each
            individual subject (which can then be more flexibily re-used).

            Behavior 'all' is designed to work with base objects
            that accept a list of length n_subjects to their fit_transform
            function, where each element of the list will be that subject's
            loaded Data File. This behavior requires loading all data
            into memory, but allows for using information from the rest
            of the group split. For example we would need to set Loader
            to 'all' if we wanted to use
            https://nilearn.github.io/modules/generated/nilearn.connectome.ConnectivityMeasure.html
            with parameter kind = "tangent" as this transformer requires
            information from the rest of the loaded subjects when training.
            On the otherhand, if we used kind = "correlation",
            then we could use either behavior
            'all' or 'single' since "correlation" can be computed for each
            subject individually.

            ::

                default = 'single'


        params : int, str or dict of :ref:`params<Params>`, optional
            `params` determines optionally if the distribution
            of hyper-parameters to
            potentially search over for this loader. Preset param
            distributions are
            listed for each choice of obj at :ref:`Loaders`,
            and you can read more on
            how params work more generally at :ref:`Params`.

            If obj is passed as :class:`Pipe`, see :class:`Pipe`
            for an example on how different
            corresponding params can be passed to each piece individually.

            ::

                default = 0

        scope : :ref:`valid scope<Scopes>`, optional
            `scope` determines on which subset of
            features the specified loader
            should transform.

            See :ref:`Scope` for more information on
            how scopes can be specified.

            Warning: If using behav = 'all', then the loader
            can only operate on a scope referring to a single fixed column!

            You will likely want to pass either a single custom key
            based column, or the default preset scope of 'data file'.

            ::

                default = 'data file'

        cache_loc : str, Path or None, optional
            Optional location in which to if set, the Loader transform
            function will be cached for each subject. These cached
            transformations can then be loaded for each subject when
            they appear again in later folds.

            Warning: If behav = 'all', then this parameter is currently
            not used.

            Set to None, to ignore

            ::

                default = None

        fix_n_wrapper_jobs : int or False, optional
            Typically this parameter is left as default, but
            in special cases you may want to set this. It controls
            the number of jobs fixed for the Loading Wrapper.

            This parameter can be used to set that value.

            ::

                default = False

        extra_params : :ref:`Extra Params`
            See :ref:`Extra Params`
        '''

        self.obj = obj

        # Make sure valid behav param
        if behav not in ['single', 'all']:
            raise RuntimeError('behav must be either "single" or "all".')

        self.behav = behav
        self.params = params
        self.scope = scope
        self.cache_loc = cache_loc
        self.extra_params = extra_params
        self.fix_n_wrapper_jobs = fix_n_wrapper_jobs

        self.check_args()


class Imputer(Piece):

    def __init__(self, obj, params=0, scope='all',
                 cache_loc=None, base_model=None, base_model_type='default',
                 **extra_params):
        '''If there is any missing data (NaN's), then an imputation strategy
        is likely neccisary (with some expections, i.e., a final model which
        can accept NaN values directly).
        This object allows for defining an imputation strategy.
        In general, you should need at most two Imputers, one for all
        `float` type data and one for all categorical data. If there
        is no missing data, this piece will be skipped.

        Parameters
        ----------
        obj : str
            `obj` selects the base imputation strategy to use.
            See :ref:`Imputers` for all avaliable options.
            Notably, if 'iterative' is passed,
            then a base model must also be passed!

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
            one for `float` data and one for `category` datatypes.
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
            access to all features. This happens because
            the iterative imputer
            will try to replace any NaN value present in its input
            feature.

            See :ref:`Scope` for more information on how scopes can
            be specified.

            ::

                default = 'all'

        cache_loc : str, Path or None, optional
            An optional path in which this Piece should be
            cached after fitting. This is typically useful in
            cases where fitting the base object takes a long time.

            To skip this option, keep at the default argument of None.

            ::

                default = None

        base_model : :class:`Model`, :class:`Ensemble` or None, optional
            If 'iterative' is passed to obj, then a base_model is required in
            order to perform iterative imputation! The base model can be
            any valid ModelPipeline Model.

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
            If all columns within the passed scope of this Imputer object
            have scope / data type 'category', then the problem
            type for the base model will be set to 'categorical'.
            In all other cases, the problem type will be set to 'regression'.

            ::

                default = 'default'


        extra_params : :ref:`Extra Params`
            See :ref:`Extra Params`

        '''

        self.obj = obj
        self.params = params
        self.scope = scope
        self.cache_loc = cache_loc
        self.base_model = deepcopy(base_model)
        self.base_model_type = base_model_type
        self.extra_params = extra_params

        self.check_args()


class Scaler(Piece):

    def __init__(self, obj, params=0, scope='float',
                 cache_loc=None, **extra_params):
        '''The Scaler piece refers to
        a piece in the :class:`ModelPipeline`,
        which is responsible for performing any sort of scaling or
        transformation on the data
        which doesn't require the target variable, and doesn't
        change the number of data points or features.
        These are typically transformations like feature scaling.

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
            should transform. See :ref:`Scope` for more
            information on how scopes can
            be specified.

            ::

                default = 'float'

        cache_loc : str, Path or None, optional
            An optional path in which this Piece should be
            cached after fitting. This is typically useful in
            cases where fitting the base object takes a long time.

            To skip this option, keep at the default argument of None.

            ::

                default = None

        extra_params : :ref:`Extra Params`
            See :ref:`Extra Params`

        '''

        self.obj = obj
        self.params = params
        self.scope = scope
        self.cache_loc = cache_loc
        self.extra_params = extra_params

        self.check_args()


class Transformer(Piece):

    def __init__(self, obj, params=0, scope='float', cache_loc=None,
                 **extra_params):
        ''' The Transformer is base optional component of the
        :class:`ModelPipeline` class.
        Transformers define any type of transformation to the loaded
        data which may change the number
        of features in a non-simple way (i.e., conceptually distinct
        from :class:`FeatSelector`, where you
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
            should transform. See :ref:`Scope` for more information
            on how scopes can
            be specified.

            Specifically, it may be useful to consider the use of
            :class:`Duplicate` here.

            ::

                default = 'float'

        cache_loc : str, Path or None, optional
            An optional path in which this Piece should be
            cached after fitting. This is typically useful in
            cases where fitting the base object takes a long time.

            To skip this option, keep at the default argument of None.

            ::

                default = None

        extra_params : :ref:`Extra Params`
            See :ref:`Extra Params`
        '''

        self.obj = obj
        self.params = params
        self.scope = scope
        self.cache_loc = cache_loc
        self.extra_params = extra_params

        self.check_args()


class FeatSelector(Piece):

    def __init__(self, obj, params=0, scope='all',
                 cache_loc=None, base_model=None, **extra_params):
        ''' FeatSelector is a base piece of
        :class:`ModelPipeline`, which is designed
        to preform feature selection.

        Parameters
        ----------
        obj : str or custom_obj
            `obj` selects the feature selection strategy to use.
            See :ref:`Feat Selectors`
            for all avaliable options. Notably, if 'rfe' is passed, then a
            base model must also be passed!

            See :ref:`Pipeline Objects` to read more about pipeline objects
            in general.

        params : int, str or dict of :ref:`params<Params>`, optional
            `params` set an associated distribution of hyper-parameters to
            potentially search over with this FeatSelector.
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
            See :ref:`Scope` for more information on how scopes can
            be specified.

            ::

                default = 'all'

        cache_loc : str, Path or None, optional
            An optional path in which this Piece should be
            cached after fitting. This is typically useful in
            cases where fitting the base object takes a long time.

            To skip this option, keep at the default argument of None.

            ::

                default = None

        base_model : :class:`Model`, :class:`Ensemble` or None, optional
            If 'rfe' is passed to obj, then a base_model is required in
            order to perform recursive feature elimination.
            The base model can be any valid argument accepts by
            param `model` in :class:`ModelPipeline`.

            ::

                default = None

        extra_params : :ref:`Extra Params`
            See :ref:`Extra Params`
        '''

        self.obj = obj
        self.params = params
        self.scope = scope
        self.cache_loc = cache_loc
        self.base_model = deepcopy(base_model)
        self.base_model_type = None
        self.extra_params = extra_params
        self.check_args()


class Model(Piece):

    _is_model = True

    def __init__(self, obj, params=0, scope='all', cache_loc=None,
                 param_search=None, target_scaler=None, **extra_params):
        ''' Model represents a base components of the :class:`ModelPipeline`,
        specifically a single Model / estimator.

        obj : str or custom_obj
            The pased object should be either
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
            should work on. See :ref:`Scope` for more
            information on how scopes can
            be specified.

            ::

                default = 'all'

        cache_loc : str, Path or None, optional
            An optional path in which this model should be
            cached after fitting. This is typically useful in
            cases where fitting the base object takes a long time.

            To skip this option, keep at the default argument of None.

            ::

                default = None

        param_search : :class:`ParamSearch`, None, optional
            If None, by default, this will be a base model.
            Alternatively, by passing a :class:`ParamSearch` instance here,
            it specifies that this model should be wrapped in a
            Hyper-parameter search object.

            This can be useful to create Model's which have
            a nested hyper-parameter tuning stage independent from the
            other pipeline steps, especially when early
            pipeline steps are long to fit, and don't
            have associated hyper-params. Notably, if set here,
            then all children param distributions of this Model
            will be associated with it's hyper-parameter search
            and this wrapped object will not pass along any param
            distributions to higher level searches.

            ::

                default = None

        target_scaler : Scaler, None, optional

            Still somewhat experimental, can pass
            a Scaler object here and have this model
            perform target scaling + reverse scaling.

            Note: Has not been fully tested in
            complicated nesting cases, e.g., if Model is
            wrapping a nested ModelPipeline, this param will
            likely break.

            ::

                default = None

        extra_params : :ref:`Extra Params`
            See :ref:`Extra Params`
        '''

        self.obj = obj
        self.params = params
        self.scope = scope
        self.cache_loc = cache_loc
        self.param_search = param_search
        self.target_scaler = target_scaler
        self.extra_params = extra_params

        self.check_args()

    def view(self, problem_type='regression'):

        pass

        # Process some way with the associated piece
        #from ..pipeline.Constructors import Models

        # Need to think about user passed objs

        '''
        from ..pipeline.Models import get_base_model_and_params, AVALIABLE

        proc_str = proc_type_dep_str(self.obj, AVALIABLE, problem_type)
        model, model_params =\
            get_base_model_and_params(proc_str, self.extra_params,
                                      self.params, True)
        model_params = {'__'.join(m.split('__')[1:]): model_params[m]
                        for m in model_params}

        return model, model_params
        '''


class Ensemble(Piece):

    _is_model = True

    def __init__(self, obj, models,
                 params=0, scope='all',
                 param_search=None,
                 target_scaler=None,
                 base_model=None,
                 cv=None,
                 single_estimator=False,
                 n_jobs_type='ensemble',
                 **extra_params):
        ''' The Ensemble object is valid base
        :class:`ModelPipeline` piece, designed
        to be passed as input to the `model` parameter
        of :class:`ModelPipeline`, or
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
            should work on. See :ref:`Scope` for more
            information on how scopes can
            be specified.

            ::

                default = 'all'

        param_search : :class:`ParamSearch`, None, optional
            If None, by default, this will be a base ensemble model.
            Alternatively, by passing a :class:`ParamSearch` instance here,
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
            wrapping a nested ModelPipeline, this param will
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

        cv : :class:`CV` or None, optional
            Used for passing custom nested internal CV split behavior to
            ensembles which employ splits, e.g., stacking.

            ::

                default = None

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

        extra_params : :ref:`Extra Params`
            See :ref:`Extra Params`
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
        self.cv = cv
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


class ParamSearch(Params):

    def __init__(self, search_type='RandomSearch', cv='default',
                 n_iter=10, scorer='default', weight_scorer=False,
                 mp_context='loky', n_jobs='default',
                 random_state='default',
                 dask_ip=None, memmap_X=False,
                 search_only_params=None, verbose=0, progress_loc=None):
        ''' ParamSearch is special input object designed to be
        used with :class:`ModelPipeline`.
        ParamSearch defines a hyperparameter search strategy.
        When passed to :class:`ModelPipeline`,
        its search strategy is applied in the context of any set :ref:`Params`
        within the base pieces.
        Specifically, there must be atleast one parameter search
        somewhere in the object ParamSearch is passed!

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

            New: You may pass 'grid' here in addition to the
            supported nevergrad searches. This will use sklearn's
            GridSearch. Note in this case some of the other parameters
            are ignored, these are: weight_scorer, mp_context, dask_ip,
            memmap_X, search_only_params

            ::

                default = 'RandomSearch'

        cv : :class:`CV` or 'default', optional
            This parameter defines the splits to be used
            internally
            @TODO finish writing this doc

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

            - 'loky': Create and use the python library loky backend.

            - 'fork': Python default fork mp_context

            - 'forkserver': Python default forkserver mp_context

            - 'spawn': Python default spawn mp_context

            ::

                default = 'loky'

        n_jobs : int or 'default', optional
            The number of cores to be used for the
            search. In general, this parameter
            should be left as 'default', which will set it
            based on the n_jobs as set in the problem spec-
            and will attempt to automatically change this
            value if say in the context of nesting.

            ::

                default = 'default'

        random_state : int or 'default', optional
            If left as 'default' will set to the value
            set in the ProblemSpec. Otherwise, can
            set a specific value here.

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

        search_only_params : dict or None, optional
            In some rare cases, it may be the case that you want
            to specify that certain parameters be passed only during
            the nested parameter searches. A dict of parameters
            can be passed here to accomplish that. For example,
            if passing:

            search_only_params = {'svm classifier__probability': False}

            And assuming that the default / selecting parameter for this svm
            classifier for probaility is True by default,
            then only when exploring
            nested hyper-parameter options will
            probability be set to False, but
            when fitting the final model with the best
            parameters found from the search,
            it will revert back to the default,
            i.e., in this case probability = True.

            Note: this may be a little bit tricky to use as
            you need to know how to represent the parameters correctly!

            To ignore this parameter / option.
            simply keep the default value of None

            ::

                default = None

        verbose : int, optional
            Controls the verbosity: the higher, the more messages.
            By default, no verbosity, i.e., 0.

            ::

                default = 0

        progress_loc : None or str, optional
            Optional parameter, will append to a text file
            after every completed tested parameter.

            ::

                default = None
        '''

        self.search_type = search_type

        if cv == 'default':
            cv = CV()
        self.cv = cv

        # @TODO allow passing single arguments, e.g. splits
        # as param and have it set within cv

        self.n_iter = n_iter
        self.scorer = scorer
        self.weight_scorer = weight_scorer
        self.mp_context = mp_context
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.dask_ip = dask_ip
        self.memmap_X = memmap_X
        self.verbose = verbose
        self.progress_loc = progress_loc
        self.search_only_params = search_only_params

        self.check_args()

    def as_dict(self, ps):

        params = self.get_params()

        if self.random_state == 'default':
            params['random_state'] = ps.random_state

        if self.n_jobs == 'default':
            params['n_jobs'] = ps.n_jobs

        params['scorer'] = process_scorer(self.scorer,
                                          ps.problem_type)

        if self.search_only_params is None:
            params['search_only_params'] = {}

        return params

    def check_args(self):

        if isinstance(self.scorer, list):
            raise IOError('scorer within Param Search cannot be list-like')
        if isinstance(self.weight_scorer, list):
            raise IOError(
                'weight_scorer within Param Search cannot be list-like')


class ModelPipeline(Params):

    def __init__(self,
                 loaders=None, imputers='default',
                 scalers=None, transformers=None,
                 feat_selectors=None,
                 model='default',
                 param_search=None,
                 cache_fit_dr=None,
                 verbose=0):
        ''' ModelPipeline is defined as essentially a wrapper around
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
            the data that changes the number of
            features in perhaps non-deterministic
            or not simply removal
            (i.e., different from feat_selectors), for example
            applying a PCA, where both the number of features change, but
            also the new features do not 1:1 correspond to the original
            features. See :class:`Transformer` for more information.

            Transformers can be composed sequentially with list or special
            input type wrappers, the same as other objects.

            ::

                default = None

        feat_selectors : :class:`FeatSelector`, list of or None, optional
            Each :class:`FeatSelector` refers to an optional
            feature selection stage
            of the Pipeline. See :class:`FeatSelector` for specific options.

            Input can be composed in a list, to apply
            feature selection sequentially,
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

        param_search : :class:`ParamSearch` or None, optional
            :class:`ParamSearch` can be provided in order
            to specify a corresponding
            hyperparameter search for the provided pipeline pieces.
            When defining each
            piece, you may set hyperparameter distributions for that piece.
            If param search
            is None, these distribution will be essentially ignored,
            but if :class:`ParamSearch`
            is passed here, then they will be used along with the
            strategy defined in the passed
            :class:`ParamSearch` to conduct a nested hyper-param search.

            Note: If using input wrapper types like :class:`Select`,
            then a param search must be passed!

            ::

                default = None

        cache_fit_dr : Path str or None, optional
            Optional parameter specifying a directory
            in which full BPt pipeline's should
            be cached after fitting. This may be useful
            in some contexts. If desired,
            passed the location of the directory in
            which to store this cache.

            ::

                default = None

        verbose : int, optional
            If greater than 0, use pipelin verbosity.

            ::

                default = 0
        '''

        if isinstance(loaders, str):
            loaders = Loader(loaders)
        self.loaders = loaders

        if imputers == 'default':
            imputers = Imputer('default')
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
            feat_selectors = FeatSelector(feat_selectors)
        self.feat_selectors = feat_selectors

        if model == 'default':
            model = Model('ridge')
            print('Passed default model, setting to:', print(model))
        elif isinstance(model, str):
            model = Model(model)
        self.model = model

        self.param_search = param_search
        self.cache_fit_dr = cache_fit_dr
        self.verbose = verbose

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

    def check_imputers(self, is_na):

        # If no NaN set to none
        if not is_na:
            self.imputers = None

        elif isinstance(self.imputers, Imputer):
            if self.imputers.obj == 'default':
                self.imputers = [Imputer('mean', scope='float'),
                                 Imputer('median', scope='cat')]

        return

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
                print('Warning: ModelPipeline user set param', p,
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

        _print('ModelPipeline')
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


class ProblemSpec(Params):

    def __init__(self, problem_type='default',
                 target=0, scorer='default',
                 weight_scorer=False,
                 scope='all', subjects='all',
                 n_jobs=1, random_state=1, base_dtype='float32'):
        '''Problem Spec is defined as an object of params encapsulating the set of
        parameters shared by modelling class functions
        :func:`Evaluate <BPt.BPt_ML.Evaluate>`
        and :func:`Test <BPt.BPt_ML.Test>`

        Parameters
        ----------
        problem_type : str or 'default', optional
            This parameter controls what type of machine learning
            should be conducted. As either a regression, or classification
            where 'categorical' represents a special case of
            binary classification,
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
            This should be passed as the name of the target column.
            This can also be set as the int index
            (in alphabetical order)
            If only one target is loaded, just leave as default of 0.

            ::

                default = 0

        scorer : str or list, optional
            Indicator str for which scorer(s) to use when calculating
            average validation score in Evaluate, or Test set score in Test.

            A list of str's can be passed as well, in this case, scores for
            all of the requested scorers will be calculated and returned.

            Note: If using a ParamSearch, the ParamSearch object has a
            seperate scorer parameter.

            For a full list of supported scorers please view the
            scikit-learn docs at:
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
            each repeated fold by the number of subjects in that
            validation set.
            This parameter only typically makes sense for
            custom split behavior where
            validation folds may end up with differing sizes.
            When default CV schemes are employed,
            there is likely no point in
            applying this weighting, as the validation
            folds will have simmilar sizes.

            If you are passing mutiple scorers, then you can also pass a 
            list of values for weight_scorer, with each value
            set as boolean True or False,
            specifying if the corresponding scorer by index
            should be weighted or not.

            Warning: This parameter is ignored when using sklearn
            compatible functions.

            ::

                default = False

        scope : key str or Scope obj, optional
            This parameter allows the user to optionally
            run an expiriment with just a subset of the loaded features
            / columns.

            See :ref:`Scope` for a more detailed explained / guide
            on how scopes are defined and used within BPt.

            ::

                default = 'all'

        subjects : str, array-like or Value_Subset, optional
            This parameter allows the user to optionally run Evaluate or Test
            with just a subset of the loaded subjects. It is notably distinct
            from the `train_subjects`, and `test_subjects` parameters directly
            avaliable to Evaluate and Test, as those parameters typically refer
            to train/test splits. Specifically, any value specified
            for this  subjects parameter will be applied
            AFTER selecting the relevant train or test subset.

            One use case for this parameter might be specifying
            subjects of just one
            sex, where you would still want the same training set for example,
            but just want to test sex specific models.

            If set to 'all' (as is by default), all avaliable subjects will be
            used.

            `subjects` can accept either a specific array of subjects,
            or even a loc of a text file (formatted one subject per line) in
            which to read from.

            A special wrapper, Value_Subset,
            can also be used to specify more specific,
            specifically value specific, subsets of subjects to use.
            See :class:`Value_Subset` for how this input wrapper can be used.

            ::

                default = 'all'

        n_jobs : int
            n_jobs are employed witin the context
            of a call to Evaluate or Test.

            In general, the way n_jobs are propegated to the different pipeline
            pieces on the backend is that,
            if there is a parameter search, the base
            ML pipeline will all be set to use 1 job,
            and the n_jobs budget will be used
            to train pipelines in parellel to explore different params.
            Otherwise, if no param search,
            n_jobs will be used for each piece individually,
            though some might not support it.

            ::

                default = 1

        random_state : int, RandomState instance or None, optional
            Random state, either as int for a specific seed, or if None then
            the random seed is set by np.random.

            This parameter is used to ensure replicability
            of expirements (wherever possible!).
            In some cases even with a random seed, depending on
            the pipeline pieces being used,
            if any have a component that occassionally yields
            different results, even with the same
            random seed, e.g., some model optimizations,
            then you might still not get
            exact replicicability.

            ::

                default = 1

        base_dtype : numpy dtype
            The dataset is cast to a numpy array of float.
            This parameter can be used to change the default
            behavior, e.g., if more resolution or less is needed.

            ::

                default = 'float32'

        '''

        self.problem_type = problem_type
        self.target = target
        self.scorer = scorer
        self.weight_scorer = weight_scorer
        self.scope = scope
        self.subjects = subjects
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.base_dtype = base_dtype

        self._final_subjects = None
        self._checked = False

        self._proc_checks()

    def _proc_checks(self):
        proc_all(self)

    def set_final_subjects(self, final_subjects):
        self._final_subjects = final_subjects

    def get_model_spec(self):

        return {'problem_type': self.problem_type,
                'random_state': self.random_state}

    def print_all(self, _print=print):

        _print('ProblemSpec')
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


class CVStrategy(Params):

    # @TODO add support for test_only ?
    def __init__(self, groups=None, stratify=None,
                 train_only_subjects=None):
        ''' This objects is used to encapsulate a set of parameters
        for a CV strategy.

        Parameters
        ----------
        groups : str or None, optional
            The str should refer to the column key in which
            to preserve groups by during any CV splits.
            To create a combination of unique values, see the
            add_unique_overlap function in the Dataset Class.

            Note: the passed column must be of type category as well.

        ::

            default = None

        stratify : str or None, optional
            The str input should refer
            to a loaded column key, either non input / strat or target.
            It will assign it as a value to preserve
            distribution of groups by during any during any CV splits.

            Note: the passed column must be of type category as well.

            Any target_cols passed must be categorical
            or binary, and cannot be
            float. Though you may consider creating a binary / k bin copy
            of a float / cont. type target variable.

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
        self.train_only_subjects = train_only_subjects

        if groups is not None and stratify is not None:
            raise RuntimeError('Warning: BPt does not currently '
                               'support groups and'
                               'stratify together!'
                               'Please use only one.')

        if not isinstance(groups, str) and groups is not None:
            raise RuntimeError('Passing non str for groups is depreciated. '
                               ' Instead, first use the function'
                               ' add_unique_overlap(cols=..., new_cols=...) to'
                               ' create explicitly the specified new values.')

        if not isinstance(stratify, str) and stratify is not None:
            raise RuntimeError('Passing non str for stratify is depreciated. '
                               ' Instead, first use the function'
                               ' add_unique_overlap(cols=..., new_cols=...) to'
                               ' create explicitly the specified new values.')


class CV(Params):

    def __init__(self, splits=3, n_repeats=1,
                 cv_strategy=None, random_state='context',
                 **cv_strategy_args):
        ''' This object is used to define a BPt style custom
        CV strategy, e.g., as KFold

        Parameters
        ----------
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

        cv_strategy: None or :class:`CVStrategy`, optional
            Optional cv_strategy to employ for calculating splits.
            If passed None, use no strategy.

            Can also pass valid cv_strategy args seperately.
            If any passed, they will override any values
            set in the pass cv_strategy if any.

            ::

                default = None

        random_state : 'context', int or None, optional
            The fixed random seed in which this
            CV object should adhere to.

            If left as default value of 'context', then
            the random state will be set based on the
            context of where it is called, i.e., typically
            the random_state set in :class:`ProblemSpec`.

            ::

                default = 'context'
        '''

        self.splits = splits
        self.n_repeats = n_repeats
        self.cv_strategy = cv_strategy
        self.random_state = random_state

        if self.cv_strategy is None:
            self.cv_strategy = CVStrategy()

        self.cv_strategy.set_params(**cv_strategy_args)

    def apply_dataset(self, dataset):
        return get_bpt_cv(self, dataset)

# Depreciations
#############################


def depreciate(func):

    # Get current and new names
    name = str(func.__name__)
    new_name = str(func.new_class.__name__)

    warn_message = name + ' is depreciated and in a future version will'
    warn_message += ' be removed. Use the new class '
    warn_message += new_name + ' to mute this warning.'

    class DepreciatedClass():

        __doc__ = warn_message

        def __new__(cls, *args, **kwargs):

            # Warn on creation
            warnings.warn(warn_message)

            # Return new class instead of this one
            return func.new_class(*args, **kwargs)

    return DepreciatedClass


@depreciate
class Feat_Selector():
    new_class = FeatSelector

@depreciate
class Param_Search():
    new_class = ParamSearch


@depreciate
class Model_Pipeline():
    new_class = ModelPipeline


@depreciate
class Problem_Spec():
    new_class = ProblemSpec
