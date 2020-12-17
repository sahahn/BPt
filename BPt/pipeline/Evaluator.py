import pandas as pd

import numpy as np
import time

from ..helpers.ML_Helpers import conv_to_list
from .Feat_Importances import get_feat_importances_and_params
from .Scorers import process_scorers
from copy import deepcopy
from os.path import dirname, abspath, exists
from sklearn.base import clone


class Evaluator():
    '''Helper class for handling all of the different parameters involved in
    model training, scaling, handling different datatypes ect...
    '''

    def __init__(self, model, problem_spec, cv, all_keys,
                 feat_importances, return_raw_preds, return_models,
                 verbosity, base_dtype, _print=print):

        # Save passed params
        self.model = model
        self.ps = problem_spec
        self.cv = cv
        self.all_keys = all_keys
        self.return_raw_preds = return_raw_preds
        self.return_models = return_models
        self.progress_bar = verbosity['progress_bar']
        self.compute_train_score = verbosity['compute_train_score']
        self.progress_loc = verbosity['progress_loc']
        self.base_dtype = base_dtype
        self._print = _print
        self.models = []

        # Default params
        self._set_default_params()

        # Set / proc scorers
        self.scorer_strs, self.scorers, _ =\
            process_scorers(deepcopy(self.ps.scorer), self.ps.problem_type)

        # Set / proc feat importance info
        self.feat_importances =\
            self._process_feat_importances(feat_importances)

    def _set_default_params(self):

        self.n_splits_ = None

        self.flags = {'linear': False,
                      'tree': False}

    def _process_feat_importances(self, feat_importances):

        # Grab feat_importance from spec as a list
        feat_importances = conv_to_list(feat_importances)

        if feat_importances is not None:

            scorers = [fi.scorer for fi in feat_importances]
            scorers =\
                [process_scorers(m, self.ps.problem_type)[2] for m in scorers]

            feat_importances =\
                [get_feat_importances_and_params(fi, self.ps.problem_type,
                                                 self.ps.n_jobs, scorer)
                    for fi, scorer in zip(feat_importances, scorers)]

            return feat_importances
        return []

    def _get_subjects_overlap(self, subjects):
        '''Computer overlapping subjects with self.ps._final_subjects'''

        if self.ps._final_subjects is None:
            overlap = set(subjects)
        else:
            overlap = self.ps._final_subjects.intersection(set(subjects))

        return np.array(list(overlap))

    def _get_results(self):

        results = {}

        # If raw_preds off, will just return None
        results['raw_preds'] = self.raw_preds_df

        # If no feature importances will just be empty list
        results['FIs'] = self.feat_importances

        # If return models off, will just be empty list
        results['models'] = self.models

        return results

    def Evaluate(self, data, train_subjects, splits,
                 n_repeats, splits_vals, only_fold=None):
        '''Method to perform a full evaluation
        on a provided model type and training subjects, according to
        class set parameters.

        Parameters
        ----------
        data : pandas DataFrame
            BPt formatted, with both training and testing data.

        train_subjects : array-like
            An array or pandas Index of the train subjects should be passed.

        Returns
        ----------
        array-like of array-like
            numpy array of numpy arrays,
            where each internal array contains the raw scores as computed for
            all passed in scorers, computed for each fold within
            each repeat.
            e.g., array will have a length of `n_repeats` * n_splits
            (num folds) and each internal array will have the same length
            as the number of scorers.
        '''

        # Set train_subjects according to self.ps._final_subjects
        train_subjects = self._get_subjects_overlap(train_subjects)

        # Init raw_preds_df
        self._init_raw_preds_df(train_subjects)

        # Setup the desired eval splits
        subject_splits =\
            self._get_eval_splits(train_subjects, splits,
                                  n_repeats, splits_vals,
                                  only_fold=only_fold)

        all_train_scores, all_scores = [], []
        fold_ind = 0

        # Init progress bar if any
        if self.progress_bar is not None:
            repeats_bar = self.progress_bar(total=n_repeats,
                                            desc='Repeats')

            folds_bar = self.progress_bar(total=self.n_splits_,
                                          desc='Folds')

        # Init progress loc if any
        if self.progress_loc is not None:
            with open(self.progress_loc, 'w') as f:
                f.write(str(n_repeats) + ',' + str(self.n_splits_))
                f.write('\n')

        self.n_test_per_fold = []

        # For each split with the repeated K-fold
        for train_subjects, test_subjects in subject_splits:

            self.n_test_per_fold.append(len(test_subjects))

            # Fold name verbosity
            repeat = str((fold_ind // self.n_splits_) + 1)
            fold = str((fold_ind % self.n_splits_) + 1)
            self._print(level='name')
            self._print('Repeat: ', repeat, '/', n_repeats, ' Fold: ',
                        fold, '/', self.n_splits_, sep='', level='name')

            if self.progress_bar is not None:
                repeats_bar.n = int(repeat) - 1
                repeats_bar.refresh()

                folds_bar.n = int(fold) - 1
                folds_bar.refresh()

            # Run actual code for this evaluate fold
            start_time = time.time()
            train_scores, scores = self.Test(data, train_subjects,
                                             test_subjects, fold_ind)

            # Time by fold verbosity
            elapsed_time = time.time() - start_time
            time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            self._print('Time Elapsed:', time_str, level='time')

            # Score by fold verbosity
            if self.compute_train_score:
                for i in range(len(self.scorer_strs)):
                    self._print('train ', self.scorer_strs[i], ': ',
                                train_scores[i], sep='', level='score')

            for i in range(len(self.scorer_strs)):
                self._print('val ', self.scorer_strs[i], ': ',
                            scores[i], sep='', level='score')

            # If progress loc
            if self.progress_loc is not None:

                if not exists(dirname(abspath(self.progress_loc))):
                    raise SystemExit('Folder where progress is stored '
                                     ' was removed!')

                with open(self.progress_loc, 'a') as f:
                    f.write('fold\n')

            all_train_scores.append(train_scores)
            all_scores.append(scores)
            fold_ind += 1

        if self.progress_bar is not None:
            repeats_bar.n = n_repeats
            repeats_bar.refresh()
            repeats_bar.close()

            folds_bar.n = self.n_splits_
            folds_bar.refresh()
            folds_bar.close()

        # If any local feat importances
        for feat_imp in self.feat_importances:
            feat_imp.set_final_local()

        # self.micro_scores = self._compute_micro_scores()

        results = self._get_results()
        return (np.array(all_train_scores), np.array(all_scores), results)

    def _get_eval_splits(self, train_subjects, splits,
                         n_repeats, splits_vals, only_fold=None):

        subject_splits = self.cv.get_cv(train_subjects, splits, n_repeats,
                                        splits_vals, self.ps.random_state,
                                        return_index=False)

        # Want to save n_splits by type
        if splits_vals is not None:
            self.n_splits_ = self.cv.get_num_groups(train_subjects,
                                                    splits_vals)

        elif isinstance(splits, int):
            self.n_splits_ = splits

        else:
            self.n_splits_ = 1

        # Check for passed only_fold
        if only_fold is not None:

            subject_splits = [subject_splits[only_fold]]
            self.n_splits_ = 1

        return subject_splits

    def Test(self, data, train_subjects, test_subjects, fold_ind='test'):
        '''Method to test given input data, training a model on train_subjects
        and testing the model on test_subjects.

        Parameters
        ----------
        data : pandas DataFrame
            BPt formatted, with both training and testing data.

        train_subjects : array-like
            An array or pandas Index of train subjects should be passed.

        test_subjects : array-like
            An array or pandas Index of test subjects should be passed.

        Returns
        ----------
        array-like
            A numpy array of scores as determined by the passed
            metric/scorer(s) on the provided testing set.
        '''

        # Reset progress loc if Test
        if fold_ind == 'test':
            if self.progress_loc is not None:
                if not exists(dirname(abspath(self.progress_loc))):
                    raise SystemExit('Folder where progress is stored '
                                     ' was removed!')

                with open(self.progress_loc, 'w') as f:
                    f.write('test\n')

        # Ensure train and test subjects are just the requested overlap
        train_subjects = self._get_subjects_overlap(train_subjects)

        # Remove any train subjects with NaN targets
        train_targets = data.loc[train_subjects, self.ps.target]
        if pd.isna(train_targets).any():
            valid = train_targets[~pd.isna(train_targets)].index
            train_subjects = self._get_subjects_overlap(valid)

        # For test subjects set to overlap
        all_test_subjects = self._get_subjects_overlap(test_subjects)

        # Get another version of the test_subjects w/o NaN
        test_targets = data.loc[all_test_subjects, self.ps.target]
        if pd.isna(test_targets).any():
            valid = test_targets[~pd.isna(test_targets)].index
            test_subjects = self._get_subjects_overlap(valid)
        else:
            test_subjects = all_test_subjects

        # Ensure data being used is just the selected col / feats
        data = data[self.all_keys]

        # Init raw_preds_df
        if fold_ind == 'test':

            # For raw preds df, keep NaNs, so use all_test_subjects
            if self.compute_train_score:
                self._init_raw_preds_df(np.concatenate([train_subjects,
                                                        all_test_subjects]))
            else:
                self._init_raw_preds_df(all_test_subjects)

        # Assume the train_subjects here are final
        train_data = data.loc[train_subjects]

        self._print('Train shape:', train_data.shape, level='size')
        self._print('Val/Test shape:', data.loc[test_subjects].shape,
                    level='size')
        if len(test_subjects) != len(all_test_subjects):
            self._print('Making predictions for additional target NaN '
                        'subjects:',
                        len(all_test_subjects) - len(test_subjects),
                        level='size')

        # Train the model(s)
        self._train_model(train_data)

        # Proc the different feat importances,
        # Pass only test subjects w/o missing targets here
        self._proc_feat_importance(train_data, data.loc[test_subjects],
                                   fold_ind)

        # Get the scores
        if self.compute_train_score:
            train_scores = self._get_scores(train_data, 'train_', fold_ind)
        else:
            train_scores = 0

        # Pass test_data w/ Nans to get_scores, in order to
        # still record predictions for targets w/ a missing
        # ground truth.
        scores = self._get_scores(data.loc[all_test_subjects], '', fold_ind)

        # Return differently based on if test
        if fold_ind == 'test':
            results = self._get_results()
            return (train_scores, scores, results)

        return train_scores, scores

    def _get_base_fitted_pipeline(self):

        if hasattr(self.model_, 'name') and self.model_.name == 'search':
            return self.model_.best_estimator_

        return self.model_

    def _get_base_fitted_model(self):

        base_pipeline = self._get_base_fitted_pipeline()
        last_name = base_pipeline.steps[-1][0]
        base_model = base_pipeline[last_name]

        return base_model

    def _set_model_flags(self):

        base_model = self._get_base_fitted_model()

        try:
            base_model.coef_
            self.flags['linear'] = True
        except AttributeError:
            pass

        try:
            base_model.feature_importances_
            self.flags['tree'] = True
        except AttributeError:
            pass

    def _proc_feat_importance(self, train_data, test_data, fold_ind):

        # Ensure model flags are set / there are feat importances to proc
        if len(self.feat_importances) > 0:
            self._set_model_flags()
        else:
            return

        # Process each feat importance
        for feat_imp in self.feat_importances:

            split = feat_imp.split

            # Init global feature df
            if fold_ind == 0 or fold_ind == 'test':

                X, y = self._proc_X_test(train_data, fs=False)
                feat_imp.init_global(X, y)

            # Local init - Test
            if fold_ind == 'test':

                if split == 'test':
                    X, y = self._proc_X_test(test_data, fs=False)

                elif split == 'train':
                    X, y = self._proc_X_test(train_data, fs=False)

                elif split == 'all':
                    X, y =\
                        self._proc_X_test(pd.concat([train_data, test_data]),
                                          fs=False)

                # Error if here
                else:
                    X, y = None, None

                feat_imp.init_local(X, y, test=True, n_splits=None)

            # Local init - Evaluate
            elif fold_ind % self.n_splits_ == 0:

                X, y = self._proc_X_test(pd.concat([train_data, test_data]),
                                         fs=False)
                feat_imp.init_local(X, y, n_splits=self.n_splits_)

            # Error if reaches here
            else:
                X, y = None, None

            self._print('Calculate', feat_imp.name, 'feat importances',
                        level='name')

            # Get base fitted model
            base_model = self._get_base_fitted_model()

            # Optionally proc train, though train is always train
            if feat_imp.get_data_needed_flags(self.flags):
                X_train = self._proc_X_train(train_data)
            else:
                X_train = None

            # Test depends on scope
            if split == 'test':
                test = test_data
            elif split == 'train':
                test = train_data
            elif split == 'all':
                test = pd.concat([train_data, test_data])

            # Always proc test.
            X_test, y_test = self._proc_X_test(test)

            try:
                fold = fold_ind % self.n_splits_
            except TypeError:
                fold = 'test'

            # Process the feature importance, provide all needed
            fis =\
                feat_imp.proc_importances(base_model, X_test, y_test=y_test,
                                          X_train=X_train, fold=fold,
                                          random_state=self.ps.random_state)

            # Grab the names of all input features
            feat_names = list(train_data)
            feat_names.remove(self.ps.target)

            # Inverse transform FIs back to original feat_space is requested
            self._inverse_transform_FIs(feat_imp, fis, feat_names)

            # For local, need an intermediate average, move df to dfs
            if isinstance(fold_ind, int):
                if fold_ind % self.n_splits_ == self.n_splits_-1:
                    feat_imp.proc_local()

    def _inverse_transform_FIs(self, feat_imp, fis, feat_names):

        global_fi, local_fi = fis
        pipeline = self._get_base_fitted_pipeline()

        # Only compute the inverse transform FI's if there
        # are either transformers or loaders in the base pipeline
        if not pipeline.has_transforms():
            feat_imp.warning = False
            return

        if feat_imp.inverse_global and global_fi is not None:
            feat_imp.inverse_global_fis.append(
                pipeline.inverse_transform_FIs(global_fi, feat_names))

        if feat_imp.inverse_local and local_fi is not None:
            feat_imp.inverse_local_fis.append(
                pipeline.inverse_transform_FIs(local_fi, feat_names))

        feat_imp.warning = True
        return

    def _get_X_y(self, data, X_as_df=False, copy=False):
        '''Helper method to get X,y data from BPt formatted df.

        Parameters
        ----------
        data : pandas DataFrame
            BPt formatted.

        X_as_df : bool, optional
            If True, return X as a pd DataFrame,
            otherwise, return as a numpy array

            (default = False)

        copy : bool, optional
            If True, return a copy of X

            (default = False)

        Returns
        ----------
        array-like
            X data for ML
        array-like
            y target for ML
        '''

        if copy:
            X = data.drop(self.ps.target, axis=1).copy()
            y = data[self.ps.target].copy()
        else:
            X = data.drop(self.ps.target, axis=1)
            y = data[self.ps.target]

        if not X_as_df:
            X = np.array(X).astype(self.base_dtype)

        y = np.array(y).astype('float64')

        return X, y

    def _train_model(self, train_data):
        '''Helper method to train a models given
        a str indicator and training data.

        Parameters
        ----------
        train_data : pandas DataFrame
            BPt formatted, training data.

        Returns
        ----------
        sklearn api compatible model object
            The trained model.
        '''

        # Data, score split
        X, y = self._get_X_y(train_data)

        # Fit the model
        self.model_ = clone(self.model)
        self.model_.fit(X, y, train_data_index=train_data.index)

        # If return models, save model
        if self.return_models:
            self.models.append(self.model_)

        # If a search object, try to print best score
        try:
            self._show_best_score()
        except KeyError:
            self._print('Error printing best score',
                        level='cv_score')

        # If a search object, show the best params
        try:
            self._show_best_params()
        except KeyError:
            self._print('Error printing best params',
                        level='params')

    def _show_best_score(self):

        try:
            name = self.model_.name
            if name != 'search':
                return None
        except AttributeError:
            return None

        self._print('Best Params Score:',
                    self.model_.best_score_, level='cv_score')

    def _show_best_params(self):

        try:
            name = self.model_.name
            if name != 'search':
                return None
        except AttributeError:
            return None

        self._print('Params Selected by Pipeline:', level='params')
        self._print(self.model_.best_params_, level='params')

        return None

    def _get_all_params(self, params, all_ps, to_show):

        if any(['__select' in p for p in params]):

            all_params = {}
            for p in params:
                ud = params[p]

                if '__select' in p:
                    base_name = '__'.join(p.split('__')[:-1])
                    to_use = all_ps[base_name + '__to_use']
                    to_show.append(base_name + '__selected: ' + str(to_use))
                    extra_ps = ud.choices[to_use]._content
                    extra_ps =\
                        {base_name + '__' + e: extra_ps[e] for e in extra_ps}

                    all_params.update(extra_ps)
                else:
                    all_params[p] = params[p]

            # Recursively call
            return self._get_all_params(all_params, all_ps, to_show)

        return params, to_show

    def _get_scores(self, test_data, eval_type, fold_ind):
        '''Helper method to get the scores of
        the trained model saved in the class on input test data.
        For all metrics/scorers.

        Parameters
        ----------
        test_data : pandas DataFrame
            BPt formatted test data.

        eval_type : {'train_', ''}

        fold_ind : int or 'test'

        Returns
        ----------
        float
            The score of the trained model on the given test data.
        '''

        # Data, score split
        X_test, y_test = self._get_X_y(test_data)

        # For book-keeping set num y classes
        self._set_classes(y_test)

        # Add raw preds to raw_preds_df
        self._add_raw_preds(X_test, y_test,
                            test_data.index, eval_type,
                            fold_ind)

        # Only compute scores on Non-Nan y
        non_nan_mask = ~np.isnan(y_test)

        # Get the scores
        scores = [scorer(self.model_,
                         X_test[non_nan_mask],
                         y_test[non_nan_mask])
                  for scorer in self.scorers]

        return np.array(scores)

    def _set_classes(self, y_test):

        # Get non-nan classes
        self.classes = np.unique(y_test[~np.isnan(y_test)])

        # Catch case where there is only one class present in y_test
        # Assume in this case that it should be binary, 0 and 1
        if len(self.classes) == 1:
            self.classes = np.array([0, 1])

    def _add_raw_preds(self, X_test, y_test, subjects, eval_type, fold_ind):

        # If return_raw_preds set to false, skip
        if not self.return_raw_preds:
            return

        if fold_ind == 'test':
            fold = 'test'
            repeat = ''
        else:
            fold = str((fold_ind % self.n_splits_) + 1)
            repeat = str((fold_ind // self.n_splits_) + 1)

        try:
            raw_prob_preds = self.model_.predict_proba(X_test)
            pred_col = eval_type + repeat + '_prob'

            if len(np.shape(raw_prob_preds)) == 3:

                for i in range(len(raw_prob_preds)):
                    p_col = pred_col + '_class_' + str(self.classes[i])
                    class_preds = [val[1] for val in raw_prob_preds[i]]
                    self.raw_preds_df.loc[subjects, p_col] = class_preds

            elif len(np.shape(raw_prob_preds)) == 2:

                for i in range(np.shape(raw_prob_preds)[1]):
                    p_col = pred_col + '_class_' + str(self.classes[i])
                    class_preds = raw_prob_preds[:, i]
                    self.raw_preds_df.loc[subjects, p_col] = class_preds

            else:
                self.raw_preds_df.loc[subjects, pred_col] = raw_prob_preds

        except AttributeError:
            pass

        raw_preds = self.model_.predict(X_test)
        pred_col = eval_type + repeat

        if len(np.shape(raw_preds)) == 2:
            for i in range(np.shape(raw_preds)[1]):
                p_col = pred_col + '_class_' + str(self.classes[i])
                class_preds = raw_preds[:, i]
                self.raw_preds_df.loc[subjects, p_col] = class_preds

        else:
            self.raw_preds_df.loc[subjects, pred_col] = raw_preds

        self.raw_preds_df.loc[subjects, pred_col + '_fold'] = fold

        # Make copy of true values
        if len(np.shape(y_test)) > 1:
            for i in range(len(self.ps.target)):
                self.raw_preds_df.loc[subjects, self.ps.target[i]] =\
                    y_test[:, i]

        elif isinstance(self.ps.target, list):
            t_base_key = '_'.join(self.ps.target[0].split('_')[:-1])
            self.raw_preds_df.loc[subjects, 'multiclass_' + t_base_key] =\
                y_test

        else:
            self.raw_preds_df.loc[subjects, self.ps.target] = y_test

    def _init_raw_preds_df(self, subjects):

        if self.return_raw_preds:
            self.raw_preds_df = pd.DataFrame(index=subjects)
        else:
            self.raw_preds_df = None

    def _proc_X_test(self, test_data, fs=True):

        # Grab the test data, X as df + copy
        X_test, y_test = self._get_X_y(test_data, X_as_df=True, copy=True)

        # Get the base pipeline
        pipeline = self._get_base_fitted_pipeline()

        return pipeline.proc_X_test(X_test, y_test,
                                    fs=fs, tp=self.base_dtype)

    def _proc_X_train(self, train_data):

        # Get X train
        X_train, _ = self._get_X_y(train_data, copy=True)

        # Get the base pipeline
        pipeline = self._get_base_fitted_pipeline()

        return pipeline.proc_X_train(X_train, tp=self.base_dtype)
