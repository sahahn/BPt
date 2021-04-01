import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from sklearn.base import clone
import time
import warnings
from ..dataset.helpers import verbose_print
from ..pipeline.helpers import get_mean_fis
from sklearn.utils import Bunch
from scipy.stats import t
from .stats_helpers import corrected_std, compute_corrected_ttest


def is_notebook():

    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
    except NameError:
        pass
    return False


# @TODO
# 1. Store permutation FI's in object after call
# 2. Add methods for plot feature importance's ?


def get_non_nan_Xy(X, y):

    # Check for if any missing targets in the test set to skip
    if pd.isnull(y).any():
        non_nan_subjs = y[~pd.isnull(y)].index
        X, y = X.loc[non_nan_subjs], y.loc[non_nan_subjs]

    return X, y


def fi_to_series(fi, feat_names):

    try:
        fi = fi.squeeze()
    except AttributeError:
        pass

    # Base flat case
    if len(fi.shape) == 1:
        return pd.Series(fi, index=feat_names)

    # Categorical case
    # @TODO is there a better way to have this?
    # e.g., maybe explicitly by class value, see self.estimators[0].classes_
    # could put it as another index level.

    series = []
    for class_fi in fi:
        series.append(pd.Series(class_fi, index=feat_names))

    return series


def fis_to_df(fis):

    # Base case - assume that first element is representative
    if isinstance(fis[0], pd.Series):
        return pd.DataFrame(fis)

    # Categorical case
    dfs = []
    for c in range(len(fis[0])):
        dfs.append(pd.DataFrame([fi[c] for fi in fis]))

    return dfs


def mean_no_zeros(df):

    mean = df.mean()
    return mean[mean != 0]


class BPtEvaluator():
    '''This class is returned from calls to :func:`evaluate`,
    and can be used to store information from evaluate, or
    compute additional feature importances.'''

    # Add verbose print
    _print = verbose_print

    def __init__(self, estimator, ps,
                 encoders=None,
                 progress_bar=True,
                 store_preds=False,
                 store_estimators=False,
                 store_timing=False,
                 eval_verbose=0,
                 progress_loc=None,
                 compare_bars=None):

        # Save base
        self.estimator = estimator
        self.ps = ps
        self.encoders_ = encoders

        # Set if using progress bar
        self._set_progress_bar(progress_bar)

        # If store preds
        self.preds = None
        if store_preds:
            self.preds = {}

        # If store estimator
        self.estimators = None
        if store_estimators:
            self.estimators = []

        # If store timing
        self.timing = None
        if store_timing:
            self.timing = {'fit': [], 'score': []}

        # @TODO Add in progress loc func.
        self.progress_loc = progress_loc
        self.verbose = eval_verbose
        self.compare_bars = compare_bars

    @property
    def mean_scores(self):
        '''This parameter stores the mean scores as
        a dictionary of values, where each dictionary
        is indexed by the name of the scorer, and the dictionary value
        is the mean score for that scorer.'''
        return self._mean_scores

    @mean_scores.setter
    def mean_scores(self, mean_scores):
        self._mean_scores = mean_scores

    @property
    def std_scores(self):
        '''This parameter stores the standard deviation scores as
        a dictionary of values, where each dictionary
        is indexed by the name of the scorer, and value
        contains the standard deviation across evaluation folds
        for that scorer.

        The default scorer key stores the micro standard
        deviation, but in the case that macro standard deviation differs,
        i.e., in the case of multiple repeats in an evaluation, then
        a separate macro standard deviation will be stored under
        the name of the scorer with _macro appended to the key.

        For example if a 3-fold twice repeated evaluation was
        run, with just r2 as the scorer, this parameter might
        look something like:

        ::

            self.std_scores = {'r2': .5, 'r2_macro': .01}

        '''
        return self._std_scores

    @std_scores.setter
    def std_scores(self, std_scores):
        self._std_scores = std_scores

    @property
    def weighted_mean_scores(self):
        '''This property stores the mean scores
        across evaluation folds (simmilar to
        :data:`mean_scores<BPtEvaluator.mean_scores>`),
        but weighted by the
        number of subjects / datapoints in each fold.

        It is scores as a dictionary indexed by the name
        of the scorer as the key, where values are
        the weighted mean score.
        '''
        return self._weighted_mean_scores

    @weighted_mean_scores.setter
    def weighted_mean_scores(self, weighted_mean_scores):
        self._weighted_mean_scores = weighted_mean_scores

    @property
    def scores(self):
        '''This property stores the scores for
        each scorer as a dictionary of lists, where
        the keys are the names of the scorer and the list
        represents the score obtained for each fold, where each
        index corresponds to to a fold of cross validation.'''
        return self._scores

    @scores.setter
    def scores(self, scores):
        self._scores = scores

    @property
    def ps(self):
        '''A saved and pre-processed version of the problem_spec
        used (with any extra_params applied) when running this
        instance of Evaluator.'''
        return self._ps

    @ps.setter
    def ps(self, ps):
        self._ps = ps

    @property
    def feat_names(self):
        '''The features names corresponding to any measures of
        feature importance, stored as a list of lists, where the top
        level list represents each fold of cross validation.

        This parameter may be especially useful when pipeline
        objects such as transformers or feature selectors are used
        as these can drastically change the features passed to an
        eventual model.

        The values stored here may change
        based on the passed
        value of the `decode_feat_names` parameter from
        :func:`evaluate`.

        For example the feat_names from a 3-fold cross-validation
        with input features ['feat1', 'feat2', 'feat3'] with
        feature selection as a piece of the pipeline may look like:

        ::

            self.feat_names = [['feat1', 'feat2'],
                               ['feat2', 'feat3'],
                               ['feat1', 'feat2']]

        '''
        return self._feat_names

    @feat_names.setter
    def feat_names(self, feat_names):
        self._feat_names = feat_names

    @property
    def val_indices(self):
        '''This parameter stores the validation subjects / index
        used in every fold of the cross-validation. It can be
        useful in some cases
        to check to see exactly what cross-validation was applied.'''
        return self._val_indices

    @val_indices.setter
    def val_indices(self, val_indices):
        self._val_indices = val_indices

    @property
    def train_indices(self):
        '''This parameter stores the training subjects / index
        used in every fold of the cross-validation. It can be
        useful in some cases
        to check to see exactly what cross-validation was applied.'''
        return self._train_indices

    @train_indices.setter
    def train_indices(self, train_indices):
        self._train_indices = train_indices

    @property
    def timing(self):
        '''This property stores information on
        the fit and scoring times, if requested by the
        original call to :func:`evaluate`.
        This parameter is a dictionary with two keys,
        'fit' and 'score'.
        Each key stores the time in seconds as a list of
        values for each of the evaluation folds.
        '''
        return self._timing

    @timing.setter
    def timing(self, timing):
        self._timing = timing

    @property
    def mean_timing(self):
        '''This property stores information on
        the fit and scoring times, if requested by the
        original call to :func:`evaluate`.
        This parameter is a dictionary with two keys,
        'fit' and 'score'.
        Each key stores the mean time in seconds across folds.
        '''
        return self._mean_timing

    @mean_timing.setter
    def mean_timing(self, mean_timing):
        self._mean_timing = mean_timing

    @property
    def preds(self):
        '''If the parameter `store_preds` is set to True when
        calling :func:`evaluate`, then this parameter will store the
        predictions from every evaluate fold.

        The parameter preds is a dictionary, where raw predictions made
        can be accessed by the key 'predict'. Values are stored as list
        corresponding to each evaluation fold.

        In the case where other predict-like functions are avaliable, e.g.,
        in the case of a binary problem, where it may be desirable to
        see the predicted probability, then the those predictions
        will be made avaliable under the name of the underlying predict
        function. In this case, that is self.preds['predict_proba'].
        It will also store results from 'predict' as well.

        self.preds also will store under 'y_true' a list, where
        each element of the list corresponds to the corresponding
        true target values for the predictions made.
        '''

        return self._preds

    @preds.setter
    def preds(self, preds):
        self._preds = preds

    @property
    def estimators(self):
        '''If the parameter `store_estimators` is set to True when
        calling :func:`evaluate`, then this parameter will store the fitted
        estimator in a list. Where each element of the list corresponds to one
        of the validation folds.

        For example to access the fitted estimator from this first
        fold ::

            first_est = self.estimators[0]

        '''
        return self._estimators

    @estimators.setter
    def estimators(self, estimators):
        self._estimators = estimators

    def _set_progress_bar(self, progress_bar):

        if not progress_bar:
            self.progress_bar = None
        elif is_notebook():
            self.progress_bar = tqdm_notebook
        else:
            self.progress_bar = tqdm

    def _eval(self, X, y, cv):

        # If verbose is lower than -1,
        # then don't show any warnings no matter the source.
        if self.verbose < -1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._evaluate(X, y, cv)

        # Otherwise, base behavior
        else:
            self._evaluate(X, y, cv)

    def _evaluate(self, X, y, cv):
        '''cv is passed as raw index, X and y as dataframes.'''

        # Compute and warn about num nan targets
        n_nan_targets = pd.isnull(y).sum()
        if n_nan_targets > 0:
            self._print('Warning: There are', str(n_nan_targets) + ' missing',
                        'targets passed to evaluate. Subjects with missing',
                        'target values will be skipped during training and '
                        'scoring. Predictions will still be made for any',
                        'in validation folds (if store_preds=True).')

        # Verbose info.
        self._print('Predicting target =', str(self.ps.target), level=1)
        self._print('Using problem_type =', str(self.ps.problem_type), level=1)
        self._print('Using scope =', str(self.ps.scope),
                    'defining an initial total of',
                    str(X.shape[1]), 'features.', level=1)
        self._print(f'Evaluating {len(X)} total data points.', level=1)

        # Init scores as dictionary of lists
        self.scores = {scorer_str: [] for scorer_str in self.ps.scorer}

        # Save train and test subjs
        self.train_indices, self.val_indices = [], []

        # Save final feat names
        self.feat_names = []

        # Init progress bar / save and compute fold info from cv
        progress_bars = self._init_progress_bars(cv)

        self._print('Using CV: ', cv,
                    'to generate evaluation splits.', level=2)

        # Run each split
        for train_inds, val_inds in cv.split(X, y):

            # Eval
            self._eval_fold(X.iloc[train_inds], y.iloc[train_inds],
                            X.iloc[val_inds], y.iloc[val_inds])

            # Increment progress bars
            progress_bars = self._incr_progress_bars(progress_bars)

        # Clean up progress bars
        self._finish_progress_bars(progress_bars)

        # Compute and score mean and stds
        self._compute_summary_scores()

    def _init_progress_bars(self, cv):

        # Passed cv should have n_repeats param - save in class
        self.n_repeats_ = 1
        if hasattr(cv, 'n_repeats'):
            self.n_repeats_ = cv.n_repeats

        # Passed cv should already be sklearn style
        n_all_splits = cv.get_n_splits()

        # Compute number of splits per repeat
        self.n_splits_ = n_all_splits
        if self.n_repeats_ > 1:
            self.n_splits_ = int(n_all_splits / self.n_repeats_)

        # Skip if no progress bar
        if self.progress_bar is None:
            return []

        # If passed compare bars is int, init top level bar
        if isinstance(self.compare_bars, int):

            # Init and set as new
            compare_bar = self.progress_bar(total=self.compare_bars,
                                            desc='Compare')
            self.compare_bars = [compare_bar]

        # If already init'ed
        elif isinstance(self.compare_bars, list):

            # Return all but last compare bar
            return self.compare_bars[:-1]

        bars = []

        # If 1 repeat, then just folds progress bar
        if self.n_repeats_ == 1:
            folds_bar = self.progress_bar(total=self.n_splits_, desc='Folds')
            bars = [folds_bar]

        # Otherwise folds and repeats bars - init repeats bar first, so on top
        else:
            repeats_bar = self.progress_bar(total=self.n_repeats_,
                                            desc='Repeats')
            folds_bar = self.progress_bar(total=self.n_splits_, desc='Folds')
            bars = [folds_bar, repeats_bar]

        # If compare bars was init'ed this run
        if self.compare_bars is not None:
            self.compare_bars = bars + self.compare_bars

        return bars

    def _incr_progress_bars(self, progress_bars):

        # Skip if not requested
        if self.progress_bar is None:
            return []

        # Increment folds bar
        folds_bar = progress_bars[0]
        folds_bar.n += 1

        # If just folds bar update and return
        if len(progress_bars) == 1:
            folds_bar.refresh()
            return [folds_bar]

        # If both, check to see if n_repeats increments
        repeats_bar = progress_bars[1]
        if folds_bar.n == self.n_splits_:
            folds_bar.n = 0
            repeats_bar.n += 1

            # If end, set to full
            if repeats_bar.n == self.n_repeats_:
                folds_bar.n = self.n_splits_

        # Update and return
        folds_bar.refresh()
        repeats_bar.refresh()
        return [folds_bar, repeats_bar]

    def _finish_progress_bars(self, progress_bars):

        # Close progress bars
        if self.compare_bars is None:
            for bar in progress_bars:
                bar.close()

            return

        # Otherwise compare bars case
        # Reset
        for bar in progress_bars:
            bar.n = 0
            bar.refresh()

        # Increment and refresh compare
        self.compare_bars[-1].n += 1
        self.compare_bars[-1].refresh()

        return

    def _eval_fold(self, X_tr, y_tr, X_val, y_val):

        # Keep track of subjects in each fold
        self.train_indices.append(X_tr.index)
        self.val_indices.append(X_val.index)

        # Get clone of estimator to fit
        estimator_ = clone(self.estimator)

        # Check for if any missing targets in the training set
        # If so, skip those subjects
        X_tr, y_tr = get_non_nan_Xy(X_tr, y_tr)
        self._print('Train shape =', X_tr.shape,
                    '(number of data points x number of feature)',
                    level=1)
        self._print('Val shape =', X_val.shape,
                    '(number of data points x number of feature)',
                    level=1)

        # Fit estimator_, passing as arrays, and with train data index
        start_time = time.time()

        estimator_.fit(X=X_tr, y=np.array(y_tr))
        fit_time = time.time() - start_time
        self._print(f'Fit fold in {fit_time:.3f} seconds.', level=1)

        # Score estimator
        start_time = time.time()
        self._score_estimator(estimator_, X_val, y_val)
        score_time = time.time() - start_time

        # Store timing if requested
        if self.timing is not None:
            self.timing['fit'].append(fit_time)
            self.timing['score'].append(score_time)

        # Save preds
        self._save_preds(estimator_, X_val, y_val)

        # Get and save final transformed feat names
        self.feat_names.append(
            estimator_.transform_feat_names(X_tr,
                                            encoders=self.encoders_))

        # If store estimators, save in self.estimators
        if self.estimators is not None:
            self.estimators.append(estimator_)

    def _score_estimator(self, estimator_, X_val, y_val):

        # Grab Non-Nan
        y_val_len = len(y_val)
        X_val, y_val = get_non_nan_Xy(X_val, y_val)

        dif = y_val_len - len(y_val)
        if dif > 0:
            self._print(f'Skipping scoring {dif} Val data points for NaN.',
                        level=1)

        # Save score for each scorer
        for scorer_str in self.ps.scorer:
            score = self.ps.scorer[scorer_str](estimator_,
                                               X_val,
                                               np.array(y_val))
            self.scores[scorer_str].append(score)
            self._print(scorer_str + ':', str(score), level=1)

    def _save_preds(self, estimator, X_val, y_val):

        if self.preds is None:
            return

        self._print('Saving predictions on validation set.', level=2)

        for predict_func in ['predict', 'predict_proba', 'decision_function']:

            # Get preds, skip if estimator doesn't have predict func
            try:
                preds = getattr(estimator, predict_func)(X_val)
            except AttributeError:
                continue

            # Add to preds dict if estimator has predict func
            try:
                self.preds[predict_func].append(preds)
            except KeyError:
                self.preds[predict_func] = [preds]

        # Add y_true
        try:
            self.preds['y_true'].append(np.array(y_val))
        except KeyError:
            self.preds['y_true'] = [np.array(y_val)]

    def _compute_summary_scores(self):

        self._print('Computing summary scores.', level=2)

        self.mean_scores, self.std_scores = {}, {}
        self.weighted_mean_scores = {}

        for scorer_key in self.scores:

            # Save mean under same name
            scores = self.scores[scorer_key]
            self.mean_scores[scorer_key] = np.mean(scores)

            # Compute scores weighted by number of subjs
            weights = [len(self.val_indices[i])
                       for i in range(len(self.val_indices))]
            self.weighted_mean_scores[scorer_key] =\
                np.average(scores, weights=weights)

            # Compute and add base micro std
            self.std_scores[scorer_key] = np.std(scores)

            # If more than 1 repeat, add the macro std
            if self.n_repeats_ > 1:
                scores = np.reshape(scores,
                                    (self.n_repeats_, self.n_splits_))
                self.std_scores[scorer_key + '_macro'] =\
                    np.std(np.mean(scores, axis=1))

        # Add mean timing
        if self.timing is not None:
            self.mean_timing = {}

            for time_key in self.timing:
                self.mean_timing[time_key] = np.mean(self.timing[time_key])

    def get_preds_dfs(self):
        '''This function can be used to return the raw predictions
        made during evaluation as a list of pandas Dataframes.

        Returns
        ---------
        dfs : list of pandas.DataFrame
            list of dataframe's per fold, where each DataFrame
            contains predictions made.
        '''

        # @TODO
        # Have to handle the different cases for different classes

        dfs = []

        # For each fold
        for fold_indx in range(len(self.val_indices)):

            # Init df
            df = pd.DataFrame(columns=list(self.preds),
                              index=self.val_indices[fold_indx])

            # Add each predict type as a column
            for predict_type in self.preds:
                df[predict_type] = self.preds[predict_type][fold_indx]

            dfs.append(df)

        return dfs

    def __repr__(self):
        rep = 'BPtEvaluator\n'
        rep += '------------\n'

        # Add scores + means
        rep += 'mean_scores = ' + repr(self.mean_scores) + '\n'
        rep += 'std_scores = ' + repr(self.std_scores) + '\n'
        rep += '\n'

        # Show avaliable saved attrs
        saved_attrs = []
        avaliable_methods = []

        if self.estimators is not None:
            saved_attrs.append('estimators')
        if self.preds is not None:
            saved_attrs.append('preds')
            avaliable_methods.append('get_preds_dfs')
        if self.timing is not None:
            saved_attrs.append('timing')

        saved_attrs += ['train_indices', 'val_indices', 'feat_names', 'ps',
                        'mean_scores', 'std_scores',
                        'weighted_mean_scores', 'scores']

        if self.estimators is not None:

            # Either or
            if self.feature_importances_ is not None:
                saved_attrs += ['fis_', 'feature_importances_']
                avaliable_methods += ['get_fis', 'get_feature_importances']
            elif self.coef_ is not None:
                saved_attrs += ['fis_', 'coef_']
                avaliable_methods += ['get_fis', 'get_coef_']

            avaliable_methods.append('permutation_importance')

        rep += 'Saved Attributes: ' + repr(saved_attrs) + '\n\n'
        rep += 'Avaliable Methods: ' + repr(avaliable_methods) + '\n\n'
        rep += 'Evaluated with:\n' + repr(self.ps) + '\n'

        return rep

    def _repr_html_(self):
        pass

    def _estimators_check(self):

        if self.estimators is None:
            raise RuntimeError('This method is not avaliable unless '
                               'evaluate is run with store_estimators=True!')

    @property
    def feature_importances_(self):
        '''This property stores the mean values
        across fitted estimators assuming each fitted estimator
        has a non empty `feature_importances_` attribute.'''

        self._estimators_check()
        return get_mean_fis(self.estimators, 'feature_importances_')

    def get_feature_importances(self):
        '''This function returns each `feature_importances_`
        value across fitted estimators. If None have this parameter,
        it will return a list of None.

        Returns
        --------
        feature_importances : list
            A list of `feature_importances_` where each element
            in the list refers to a fold from the evaluation.
        '''

        self._estimators_check()
        return [estimator.feature_importances_
                for estimator in self.estimators]

    @property
    def coef_(self):
        '''This attribute represents the mean `coef_` as
        a numpy array across all folds. This parameter will only
        be avaliable if all estimators have a non null `coef_` parameter
        and each returns the same shape. See `fis_` for a more flexible
        version of this parameter that can handle when there
        are differing numbers of features.'''

        self._estimators_check()
        return get_mean_fis(self.estimators, 'coef_')

    def get_coefs(self):
        '''This function returns each `coef_`
        value across fitted estimators. If None have this parameter,
        it will return a list of None.

        Returns
        --------
        coefs : list
            A list of `coef_` where each element
            in the list refers to a fold from the evaluation.
        '''

        self._estimators_check()
        return [estimator.coef_
                for estimator in self.estimators]

    @property
    def fis_(self):
        '''This property stores the mean value
        across each fold of the CV for either the `coef_`
        or `feature_importance_` parameter.

        Warnings
        ---------
        If a feature is not present in all folds,
        then it's mean value will be computed from only the
        folds in which it was present.

        When using transformers, for example one hot encoder,
        since the encoding is done on the fly, there is no
        guarantee that 'one hot encoder category_1' is actually
        the same category 1 across folds.

        If for some reason some folds have a model with feature
        importances and other `coef_` they will still all be averaged
        together, so make sure that this parameter is only used when
        all of the underlying models across folds should have comparable
        feature importances.
        '''

        # @TODO incoperate in information about the original
        # class names here // maybe in specific objects like
        # OneHotEncoder.

        self._estimators_check()

        # Grab fis as Dataframe or list of
        fis = self.get_fis()

        # Base case
        if isinstance(fis, pd.DataFrame):
            return fis.mean()

        # Categorical case
        return [fi.mean() for fi in fis]

    def get_fis(self, mean=False, abs=False):
        '''This method will return a pandas DataFrame with
        each row a fold, and each column a feature if
        the underlying model supported either the `coef_`
        or `feature_importance_` parameters.

        In the case that the underlying feature importances
        or `coefs_` are not flat, e.g., in the case
        of a one versus rest categorical model, then a list
        multiple DataFrames will be returned, one for each class.
        The order of the list will correspond to the order of classes.

        Parameters
        -----------
        mean : bool, optional
            If True, return the mean value
            across evaluation folds as a pandas Series.
            Any features with a mean value of 0 will
            also be excluded. Otherwise, if default
            of False, return raw values for each fold
            as a Dataframe.

            ::

                default = False

        abs : bool, optional
            If the feature importances
            should be absolute values
            or not.

            ::

                default = False

        Returns
        --------
        fis : pandas DataFrame or Series
            Assuming mean=False, the
            a pandas DataFrame where each row contains the
            feature importances from an evaluation fold (unless the underlying
            feature importances are categorical, in which a list of DataFrames
            will be returned.)

            If mean=True, then a pandas Series (or in the case of
            underlying categorical feature importances, list of)
            will be returned, with the mean value from each fold
            and all features with a value of 0 excluded.

            Note: To get the mean values without zero's excluded,
            just call .mean() on the result of this method
            with mean=False.


        '''

        # @TODO handle multi-class case ...

        self._estimators_check()

        coefs = self.get_coefs()
        feature_importances = self.get_feature_importances()

        fis = []
        for coef, fi, feat_names in zip(coefs, feature_importances,
                                        self.feat_names):
            if coef is not None:
                fis.append(fi_to_series(coef, feat_names))
            elif fi is not None:
                fis.append(fi_to_series(fi, feat_names))
            else:
                fis.appends(None)

        base = fis_to_df(fis)

        # Proc. abs arg
        if abs:
            if isinstance(base, list):
                base = [b.abs() for b in base]
            else:
                base = base.abs()

        # If not mean, return as is
        if not mean:
            return base

        # Categorical mean case
        if isinstance(base, list):
            return [mean_no_zeros(b) for b in base]

        # Base mean case
        return mean_no_zeros(base)

    def _get_val_fold_Xy(self, estimator, X_df, y_df, fold, just_model=True):

        # Get the X and y df's, without any subjects with missing
        # target values for this fold
        X_val_df, y_val_df =\
            get_non_nan_Xy(X_df.loc[self.val_indices[fold]],
                           y_df.loc[self.val_indices[fold]])

        # Base as array, and all feat names
        X_trans, feat_names = np.array(X_val_df), list(X_val_df)

        # Transform the X df, casts to array if just_model.
        if just_model:
            feat_names =\
                estimator.transform_feat_names(feat_names,
                                               encoders=self.encoders_)
            X_trans = estimator.transform(X_trans,
                                          transform_index=X_val_df.index)
            estimator = estimator._final_estimator

        return estimator, X_trans, np.array(y_val_df), feat_names

    def permutation_importance(self, dataset,
                               n_repeats=10, scorer='default',
                               just_model=True, return_as='dfs',
                               n_jobs=1, random_state='default'):
        '''This function computes the permutation feature importances
        from the base scikit-learn function
        :func:`sklearn.inspection.permutation_importance`

        Parameters
        -----------
        dataset : :class:`Dataset`
            The instance of the Dataset class originally passed to
            :func:`evaluate`. Note: if you pass a different dataset,
            you may get unexpected behavior.

        n_repeats : int, optional
            The number of times to randomly permute each feature.

            ::

                default = 10

        scorer : sklearn-style scoring, optional
            Scorer to use. It can be a single sklearn style str,
            or a callable.

            If left as 'default' will use the first scorer defined when
            evaluating the underlying estimator.

            ::

                default = 'default'


        just_model : bool, optional
            When set to true, the permutation feature importances
            will be computed using the final set of transformed features
            as passed when fitting the base model. This is reccomended
            behavior because it means that the features do not need to
            be re-transformed through the full pipeline to evaluate each
            feature. If set to False, will permute the features in the
            original feature space (which may be useful in some context).

            ::

                default = True

        return_as : {'dfs', 'raw'}, optional
            This parameter controls if calculated permutation
            feature importances should be returned as a DataFrame
            with column names as the corresponding feature names,
            or if it should be returned as a list with the raw
            output from each fold, e.g., sklearn Batch's with
            parameters 'importances_mean', 'importances_std'
            and 'importances'.

            If return as DataFrame is requested, then
            'importances_mean' and 'importances_std'
            will be returned, but not the raw 'importances'.

            ::

                default = 'dfs'


        n_jobs : int, optional
            The number of jobs to use for this function. Note
            that if the underlying estimator supports multiple jobs
            during inference (predicting), and the original
            problem_spec was set with multiple n_jobs then that original
            behavior will still hold, and you may wish to keep this
            parameter as 1. On the otherhand, if the base estimator
            does not use multiple jobs, passing a higher value here
            could greatly speed up computation.

            ::

                default = 1

        random_state : int, 'default' or None, optional
            Pseudo-random number generator to control the permutations
            of each feature.
            If left as 'default' then use the random state defined
            during the initial evaluation of the model. Otherwise, you may
            pass an int for a different fixed random state or None
            for explicitly no
            random state.

            ::

                default = 'default'
        '''

        # @TODO in case of just_model = False, won't pass along
        # transform_index correctly to scorer.

        from sklearn.inspection import permutation_importance

        self._estimators_check()

        # If default scorer, take the first one
        if scorer == 'default':
            first = list(self.ps.scorer)[0]
            scorer = self.ps.scorer[first]
            self._print('Using scorer:', first, level=1)

        # If default random_state use the one saved in
        # original problem spec.
        if random_state == 'default':
            random_state = self.ps.random_state

        # Get X and y from saved problem spec
        X, y = dataset.get_Xy(self.ps)

        # For each estimator
        all_fis, all_feat_names = [], []
        for fold, estimator in enumerate(self.estimators):

            # Get correct estimator, X_val, y_val and feat_names
            estimator, X_val, y_val, feat_names =\
                self._get_val_fold_Xy(estimator, X_df=X, y_df=y,
                                      fold=fold, just_model=just_model)
            all_feat_names.append(feat_names)

            # Run the sklearn feature importances.
            fis = permutation_importance(estimator, X_val, y_val,
                                         scoring=scorer, n_repeats=n_repeats,
                                         n_jobs=n_jobs,
                                         random_state=random_state)
            # Add to all fis
            all_fis.append(fis)

        # If raw, return as raw
        if return_as == 'raw':
            return all_fis

        # Otherwise, use return df
        mean_series, std_series = [], []
        for fis, feat_names in zip(all_fis, all_feat_names):
            mean_series.append(
                fi_to_series(fis['importances_mean'], feat_names))
            std_series.append(
                fi_to_series(fis['importances_std'], feat_names))

        # Return as sklearn bunch of DataFrames
        return Bunch(importances_mean=fis_to_df(mean_series),
                     importances_std=fis_to_df(std_series))

    def get_X_transform_df(self, dataset, fold=0, subjects='tr'):
        '''This method is used as a helper for getting the transformed
        input data for one of the saved models run during evaluate.

        Parameters
        -----------
        dataset : :class:`Dataset`
            The instance of the Dataset class originally passed to
            :func:`evaluate`. Note: if you pass a different dataset,
            you may get unexpected behavior.

        fold : int, optional
            The corresponding fold of the trained
            estimator to use.

        subjects : 'tr', 'val' or :ref:`Subjects`, optional
            The subjects data in which to return. As
            either special strings 'tr' for train subjects
            in the corresponding fold. Special str
            'val' for the validation subjects in the
            selected for or lastly any valid
            :ref:`Subjects` style input.

        Returns
        ----------
        X_trans_df : pandas DataFrame
            The transformed features in a DataFrame,
            according to the saved estimator from a fold,
            for the specified subjects.

            If kept as the default of subjects == 'tr',
            then these represent the feature values as
            passed to trained the actual model component
            of the pipeline.
        '''

        self._estimators_check()

        # Estimator from the fold
        estimator = self.estimators[fold]

        if subjects == 'tr':
            subjects = self.train_indices[fold]
        elif subjects == 'val':
            subjects = self.val_indices[fold]

        # Get feature names from fold
        feat_names = self.feat_names[fold]

        # Get as X
        X_fold, _ = dataset.get_Xy(problem_spec=self.ps,
                                   subjects=subjects)

        # Transform the data up to right before it gets passed to the
        # elastic net
        X_trans_fold = estimator.transform(X_fold)

        # Put the data in a dataframe with associated feature names
        return pd.DataFrame(X_trans_fold, columns=feat_names)

    def compare(self, other, rope_interval=[-0.01, 0.01]):
        '''This method is designed to perform a statistical comparison
        between the results from the evaluation stored in this object
        and another instance of :class:`BPtEvaluator`. The statistics
        produced here are explained in:
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html

        .. note::
            In the case that the sizes of the training and validation sets
            at each fold vary dramatically, it is unclear if this
            statistics are still valid.
            In that case, the mean train size and mean validation sizes
            are employed when computing statistics.

        Parameters
        ------------
        other : :class:`BPtEvaluator`
            Another instance of :class:`BPtEvaluator` in which
            to compare which. The cross-validation used
            should be the same in both instances, otherwise
            statistics will not be generated.

        rope_interval : list or dict of
            | This parameter allows for passing in a custom
                region of practical equivalence interval (or rope interval)
                a concept from bayesian statistics. If passed as
                a list, this should be a list with two elements, describing
                the difference in score which should be treated as two
                models or runs being practically equivalent.

            | Alternatively, in the case of multiple underlying
                scorers / metrics. A dictionary, where keys correspond
                to scorer / metric names can be passed with a separate
                rope_interval for each. For example:

            ::

                rope_interval = {'explained_variance': [-0.01, 0.01],
                                 'neg_mean_squared_error': [-1, 1]}

            This example would define separate rope regions depending
            on the metric.

            ::

                default = [-0.01, 0.01]

        Returns
        -------
        compare_df : pandas DataFrame
            | The returned DataFrame will generate separate rows
                for all overlapping metrics / scorers between the
                evaluators being compared. Further, columns with
                statistics of interest will be generated:

                - 'mean_diff'
                    The mean score minus other's mean score

                - 'std_diff'
                    The std minus other's std

            | Further, only in the case that the cross-validation
                folds are identical between the comparisons,
                the following additional columns will be generated:

                - 't_stat'
                    Corrected paired ttest statistic.

                - 'p_val'
                    The p value for the corrected paired ttest statistic.

                - 'better_prob'
                    The probability that this evaluated option is better than
                    the other evaluated option under a bayesian framework and
                    the passed value of rope_interval. See sklearn example
                    for more details.

                - 'worse_prob'
                    The probability that this evaluated option is worse than
                    the other evaluated option under a bayesian framework and
                    the passed value of rope_interval. See sklearn example
                    for more details.

                - 'rope_prob'
                    The probability that this evaluated option is equivalent to
                    the other evaluated option under a bayesian framework and
                    the passed value of rope_interval. See sklearn example
                    for more details.

        '''

        equal_cv = True

        # Make sure same number of folds
        if len(self.train_indices) != len(other.train_indices):
            equal_cv = False

        # Make sure subjects from folds line up
        for fold in range(len(self.train_indices)):
            if not np.array_equal(self.train_indices[fold],
                                  other.train_indices[fold]):
                equal_cv = False

            if not np.array_equal(self.val_indices[fold],
                                  other.val_indices[fold]):
                equal_cv = False

        # Only compute for the overlapping metrics
        overlap_metrics = set(list(self.mean_scores)).intersection(set(
            list(other.mean_scores)))

        for metric in overlap_metrics:
            if np.array_equal(self.scores[metric], other.scores[metric]):
                raise RuntimeError(
                    f'Cannot compare as scores are identical for {metric}.')

        # Init difference dataframe
        dif_df = pd.DataFrame(index=list(overlap_metrics))

        # Add base differences
        for metric in overlap_metrics:

            dif_df.loc[metric, 'mean_diff'] =\
                self.mean_scores[metric] - other.mean_scores[metric]

            dif_df.loc[metric, 'std_diff'] =\
                self.std_scores[metric] - other.std_scores[metric]

        # Only compute p-values if equal cv
        if equal_cv:
            for metric in overlap_metrics:

                # Grab scores and other info
                scores1 = np.array(self.scores[metric])
                scores2 = np.array(other.scores[metric])

                differences = scores1 - scores2
                n = len(scores1)
                df = n - 1

                # Use the mean train / test size
                n_train = np.mean([len(ti) for ti in self.train_indices])
                n_test = np.mean([len(ti) for ti in self.val_indices])

                # Frequentist Approach
                t_stat, p_val = compute_corrected_ttest(differences, df,
                                                        n_train, n_test)
                dif_df.loc[metric, 't_stat'] = t_stat
                dif_df.loc[metric, 'p_val'] = p_val

                # Bayesian
                t_post = t(df, loc=np.mean(differences),
                           scale=corrected_std(differences, n_train, n_test))

                # Passed as either list of two values or dict
                if isinstance(rope_interval, dict):
                    ri = rope_interval[metric]
                else:
                    ri = rope_interval

                worse_prob = t_post.cdf(ri[0])
                better_prob = 1 - t_post.cdf(ri[1])
                rope_prob =\
                    t_post.cdf(ri[1]) - t_post.cdf(ri[0])

                # Add to dif_df
                dif_df.loc[metric, 'better_prob'] = better_prob
                dif_df.loc[metric, 'worse_prob'] = worse_prob
                dif_df.loc[metric, 'rope_prob'] = rope_prob

        return dif_df
