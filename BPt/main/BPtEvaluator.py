from ..helpers.ML_Helpers import conv_to_list
from ..pipeline.Feat_Importances import get_feat_importances_and_params
from ..pipeline.Scorers import process_scorer
import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from sklearn.model_selection import check_cv
from sklearn.base import clone
import time
from ..dataset.helpers import verbose_print
from ..pipeline.helpers import get_mean_fis


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
                 progress_bar=True,
                 store_preds=False,
                 store_estimators=False,
                 store_timing=False,
                 progress_loc=None,
                 verbose=0):

        # Save base
        self.estimator = estimator
        self.ps = ps

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

        if is_notebook():
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

        # Conv passed cv to sklearn style
        is_classifier = self.ps.problem_type != 'regression'
        cv = check_cv(cv=cv, y=y, classifier=is_classifier)
        n_splits = cv.get_n_splits()

        # Init progress bar
        progress_bar = None
        if self.progress_bar is not None:
            progress_bar = self.progress_bar(total=n_splits, desc='Progress')

        # Run each split
        fold = 0
        for train_inds, val_inds in cv.split(X):

            # Eval
            self._eval_fold(X.iloc[train_inds], y.iloc[train_inds],
                            X.iloc[val_inds], y.iloc[val_inds])
            fold += 1

            # Increment progress bar
            if progress_bar is not None:
                progress_bar.n = int(fold)
                progress_bar.refresh()

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
        self.feat_names.append(estimator_.transform_feat_names(X_tr, fs=True))

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

        for predict_func in ['predict', 'predict_proba',
                             'decision_function', 'predict_log_proba']:

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

    def get_preds_dfs(self):

        # @TODO
        # Have to handle the different cases for different classes


        pass

    def __repr__(self):
        pass

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
        return self.get_fis().mean()

    def get_fis(self):
        '''This will return a pandas DataFrame with
        each row a fold, and each column a feature if
        the underlying model supported either the coef_
        or feature_importance_ parameters.'''

        # @TODO handle multi-class case ...

        self._estimators_check()

        coefs = self.get_coefs()
        feature_importances = self.get_feature_importances()

        fis = []
        for coef, fi, feat_names in zip(coefs, feature_importances,
                                        self.feat_names):
            if coef is not None:
                fis.append(pd.Series(coef, index=feat_names))
            elif fi is not None:
                fis.append(pd.Series(fi, index=feat_names))
            else:
                fis.appends(None)

        return pd.DataFrame(fis)

