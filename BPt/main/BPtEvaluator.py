from ..helpers.ML_Helpers import conv_to_list
from ..pipeline.Feat_Importances import get_feat_importances_and_params
from ..pipeline.Scorers import process_scorer
import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from sklearn.base import clone
import time
from ..dataset.helpers import verbose_print
from ..pipeline.helpers import get_mean_fis
from sklearn.utils import Bunch


def is_notebook():

    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
    except NameError:
        pass
    return False


def get_non_nan_Xy(X, y):

    # Check for if any missing targets in the test set to skip
    if pd.isnull(y).any():
        non_nan_subjs = y[~pd.isnull(y)].index
        X, y = X.loc[non_nan_subjs], y.loc[non_nan_subjs]

    return X, y


class BPtEvaluator():

    # Add verbose print
    _print = verbose_print

    def __init__(self, estimator, ps,
                 encoders=None,
                 progress_bar=True,
                 store_preds=False,
                 store_estimators=False,
                 store_timing=False,
                 progress_loc=None,
                 verbose=0):

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
            self.timing = {'fit_time': [], 'score_time': []}

        # @TODO Add in progress loc func.
        self.progress_loc = progress_loc
        self.verbose = verbose

    def _set_progress_bar(self, progress_bar):

        if not progress_bar:
            self.progress_bar = None
        elif is_notebook():
            self.progress_bar = tqdm_notebook
        else:
            self.progress_bar = tqdm

    def _process_feat_importances(self, feat_importances):

        # Grab feat_importance from spec as a list
        feat_importances = conv_to_list(feat_importances)

        if feat_importances is not None:

            # Process each scorer
            scorers = [process_scorer(fi.scorer, self.ps.problem_type)
                       for fi in feat_importances]

            # Get each feature importance object
            feat_importances =\
                [get_feat_importances_and_params(fi, self.ps.problem_type,
                                                 self.ps.n_jobs, scorer)
                    for fi, scorer in zip(feat_importances, scorers)]

            return feat_importances

        # Return empty list if None
        return []

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

        # Init scores as dictionary of lists
        self.scores = {scorer_str: [] for scorer_str in self.ps.scorer}

        # Save train and test subjs
        self.train_subjs, self.val_subjs = [], []

        # Save final feat names
        self.feat_names = []

        # Init progress bar / save and compute fold info from cv
        progress_bars = self._init_progress_bars(cv)

        # Run each split
        for train_inds, val_inds in cv.split(X, y):

            # Eval
            self._eval_fold(X.iloc[train_inds], y.iloc[train_inds],
                            X.iloc[val_inds], y.iloc[val_inds])

            # Increment progress bars
            progress_bars = self._incr_progress_bars(progress_bars)

        # Close progress bars
        for bar in progress_bars:
            bar.close()

        # Compute and score mean and stds
        self._compute_summary_scores()

    def _init_progress_bars(self, cv):

        # Passed cv should have n_repeats param - save in classs
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

        # If 1 repeat, then just folds progress bar
        if self.n_repeats_ == 1:
            folds_bar = self.progress_bar(total=self.n_splits_, desc='Folds')
            return [folds_bar]

        # Otherwise folds and repeats bars - init repeats bar first, so on top
        repeats_bar = self.progress_bar(total=self.n_repeats_, desc='Repeats')
        folds_bar = self.progress_bar(total=self.n_splits_, desc='Folds')
        return [folds_bar, repeats_bar]

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

    def _eval_fold(self, X_tr, y_tr, X_val, y_val):

        # Keep track of subjects in each fold
        self.train_subjs.append(X_tr.index)
        self.val_subjs.append(X_val.index)

        # Get clone of estimator to fit
        estimator_ = clone(self.estimator)

        # Check for if any missing targets in the training set
        # If so, skip those subjects
        X_tr, y_tr = get_non_nan_Xy(X_tr, y_tr)

        # Fit estimator_, passing as arrays, and with train data index
        start_time = time.time()
        estimator_.fit(X=np.array(X_tr), y=np.array(y_tr),
                       train_data_index=X_tr.index)
        fit_time = time.time() - start_time

        # Score estimator
        start_time = time.time()
        self._score_estimator(estimator_, X_val, y_val)
        score_time = time.time() - start_time

        # Store timing if requested
        if self.timing is not None:
            self.timing['fit_time'].append(fit_time)
            self.timing['score_time'].append(score_time)

        # Save preds
        self._save_preds(estimator_, X_val, y_val)

        # Get and save final transformed feat names
        self.feat_names.append(
            estimator_.transform_feat_names(X_tr, fs=True,
                                            encoders=self.encoders_))

        # If store estimators, save in self.estimators
        if self.estimators is not None:
            self.estimators.append(estimator_)

    def _score_estimator(self, estimator_, X_val, y_val):

        # Grab non-nan
        X_val, y_val = get_non_nan_Xy(X_val, y_val)

        # Save score for each scorer
        for scorer_str in self.ps.scorer:
            score = self.ps.scorer[scorer_str](estimator_,
                                               np.array(X_val),
                                               np.array(y_val))
            self.scores[scorer_str].append(score)

    def _save_preds(self, estimator, X_val, y_val):

        if self.preds is None:
            return

        for predict_func in ['predict', 'predict_proba', 'decision_function']:

            # Get preds, skip if estimator doesn't have predict func
            try:
                preds = getattr(estimator, predict_func)(np.array(X_val))
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

        self.mean_scores, self.std_scores = {}, {}
        for scorer_key in self.scores:

            # Save mean under same name
            scores = self.scores[scorer_key]
            self.mean_scores[scorer_key] = np.mean(scores)

            # Compute and add base micro std
            self.std_scores[scorer_key] = np.std(scores)

            # If more than 1 repeat, add the macro std
            if self.n_repeats_ > 1:
                scores = np.reshape(scores,
                                    (self.n_repeats_, self.n_splits_))
                self.std_scores[scorer_key + '_macro'] =\
                    np.std(np.mean(scores, axis=1))

    def get_preds_dfs(self):

        # @TODO
        # Have to handle the different cases for different classes
        pass

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

        saved_attrs += ['train_subjs', 'val_subjs', 'feat_names', 'ps']
        saved_attrs += ['mean_scores', 'std_scores', 'scores']

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

        self._estimators_check()
        return get_mean_fis(self.estimators, 'feature_importances_')

    def get_feature_importances(self):

        self._estimators_check()
        return [estimator.feature_importances_
                for estimator in self.estimators]

    @property
    def coef_(self):
        '''This attribute represents the mean coef_ as
        a numpy array across all folds. This parameter will only
        be avaliable if all estimators have a non null coef_ parameter
        and each returns the same shape. See fis_ for a more flexible
        version of this parameter that can handle when there
        are differing numbers of features.'''

        self._estimators_check()
        return get_mean_fis(self.estimators, 'coef_')

    def get_coefs(self):

        self._estimators_check()
        return [estimator.coef_
                for estimator in self.estimators]

    @property
    def fis_(self):
        '''This property will return the mean value
        across each fold of the CV for either the coef_
        or feature_importance_ parameter. Note: 

        Warnings:
        - If a feature is not present in all folds,
        then it's mean value will be computed from only the
        folds in which it was present.

        - When using transformers, for example one hot encoder,
        since the encoding is done on the fly, there is no
        gaurentee that 'one hot encoder category_1' is actually
        the same category 1 across folds.

        - If for some reason some folds have a model with feature
        importances and other coef_ they will still all be averaged
        together, so make sure that this parameter is only used when
        all of the underlying models across folds should have comparable
        feature importances.
        '''

        # @TODO incoperate in information about the original
        # class names here // maybe in specific objects like
        # OneHotEncoder.

        self._estimators_check()

        # Grab fi's as Dataframe or list of
        fis = self.get_fis()

        # Base case
        if isinstance(fis, pd.DataFrame):
            return fis.mean()

        # Categorical case
        return [fi.mean() for fi in fis]

    def get_fis(self):
        '''This will return a pandas DataFrame with
        each row a fold, and each column a feature if
        the underlying model supported either the coef_
        or feature_importance_ parameters.

        In the case that the underlying feature importances
        or coefs_ are not flat, e.g., in the case
        of a one versus rest categorical model, then a list
        multiple DataFrames will be returned, one for each class.
        The order of the list will correspond to the order of classes.
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

        return fis_to_df(fis)

    def _get_val_fold_Xy(self, estimator, X_df, y_df, fold, just_model=True):

        # Get the X and y df's, without any subjects with missing
        # target values for this fold
        X_val_df, y_val_df =\
            get_non_nan_Xy(X_df.loc[self.val_subjs[fold]],
                           y_df.loc[self.val_subjs[fold]])
        # Base as array, and all feat names
        X_trans, feat_names = np.array(X_val_df), list(X_val_df)

        # Transform the X df, casts to array if just_model.
        if just_model:
            feat_names =\
                estimator.transform_feat_names(feat_names, fs=True,
                                               encoders=self.encoders_)
            X_trans = estimator.transform(X_trans)
            estimator = estimator._final_estimator

        return estimator, X_trans, np.array(y_val_df), feat_names

    def permutation_importance(self, dataset,
                               n_repeats=10, scorer='default',
                               just_model=True, return_as='dfs',
                               n_jobs=1, random_state='default'):
        '''This function computes the permutation feature importances
        from the base scikit-learn function permutation_importance:
        https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html#sklearn.inspection.permutation_importance

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
            evaluating the model.

            ::

                default = 'default'


        just_model : bool, optional
            When set to true, the permuation feature importances
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


def fi_to_series(fi, feat_names):

    # Base flat case
    if len(fi.shape) == 1:
        return pd.Series(fi, index=feat_names)

    # Categorical case
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
