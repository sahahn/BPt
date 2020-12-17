"""
_ML.py
====================================
Main class extension file for the Machine Learning functionality
"""
from copy import deepcopy
import os
import pickle as pkl

from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

from .Input_Tools import is_value_subset, is_values_subset
from ..helpers.Data_Helpers import (get_unique_combo_df,
                                    reverse_unique_combo_df,
                                    get_target_type)
from ..helpers.ML_Helpers import (compute_micro_macro, conv_to_list,
                                  get_avaliable_run_name)
from ..pipeline.Evaluator import Evaluator
from ..main.Params_Classes import (CV_Splits, CV_Split, Feat_Importance,
                                   Model_Pipeline,
                                   Model, Ensemble, Problem_Spec)
from ..pipeline.Model_Pipeline import get_pipe
import pandas as pd
import copy


def Set_Default_ML_Verbosity(
 self, save_results='default', progress_bar='default',
 progress_loc='default',
 pipeline_verbose='default',
 best_params_score='default',
 compute_train_score='default',
 show_init_params='default', fold_name='default',
 time_per_fold='default', score_per_fold='default', fold_sizes='default',
 best_params='default', save_to_logs='default', flush='default'):
    '''This function allows setting various verbosity options that effect
    output during :func:`Evaluate` and :func:`Test`.

    Parameters
    ----------
    save_results : bool, optional
        If True, all results returned by Evaluate
        will be saved within the log dr (if one exists!),
        under run_name + .eval, and simmilarly for results
        returned by Test, but as run_name + .test.

        if 'default', and not already defined, set to False.

        ::

            default = 'default'

    progress_bar : bool, optional
        If True, a progress bar, implemented in the python
        library tqdm, is used to show progress during use of
        :func:`Evaluate` , If False, then no progress bar is shown.
        This bar should work both in a notebook env and outside one,
        assuming self.notebook has been set correctly.

        if 'default', and not already defined, set to True.

        ::

            default = 'default'

    progress_loc : str, Path or None, optional
        If not None, then this will record the progress
        of each Evaluate / Test call in this location.

        if 'default', and not already defined, set to False.

        ::

            default = 'default'

    pipeline_verbose : bool, optional
        This controls the verbose parameter for the pipeline object itself.
        If set to True, then time elapsed while fitting each step will be
        printed.

        if 'default', and not already defined, set to False.

        ::

            default = 'default'

    compute_train_score : bool, optional
        If True, then metrics/scorers and raw preds will also be
        computed on the training set in addition to just the
        eval or testing set.

        if 'default', and not already defined, set to False.

        ::

            default = 'default'

    show_init_params : bool, optional
        If True, then print/show the parameters used before running
        Evaluate / Test. If False, then don't print the params used.

        if 'default', and not already defined, set to True.

        ::

            default = 'default'

    fold_name : bool, optional
        If True, prints a rough measure of progress via
        printing out the current fold (somewhat redundant with the
        progress bar if used, except if used with other params, e.g.
        time per fold, then it is helpful to have the time printed
        with each fold). If False, nothing is shown.

        if 'default', and not already defined, set to False.

        ::

            default = 'default'

    time_per_fold : bool, optional
        If True, prints the full time that a fold took to complete.

        if 'default', and not already defined, set to False.

        ::

            default = 'default'

    score_per_fold : bool, optional
        If True, displays the score for each fold, though slightly less
        formatted then in the final display.

        if 'default', and not already defined, set to False.

        ::

            default = 'default'

    fold_sizes : bool, optional
        If True, will show the number of subjects within each train
        and val/test fold.

        if 'default', and not already defined, set to False.

        ::

            default = 'default'

    best_params : bool, optional
        If True, print the best search params found after every
        param search.

        if 'default', and not already defined, set to False.

        ::

            default = 'default'

    save_to_logs : bool, optional
        If True, then when possible, and with the selected model
        verbosity options, verbosity ouput will be saved to the
        log file.

        if 'default', and not already defined, set to False.

        ::

            default = 'default'

    flush : bool, optional
        If True, then add flush=True to all ML prints, which
        adds a call to flush the std output.

        if 'default', and not already defined, set to False.

        ::

            default = False
    '''

    if save_results != 'default':
        self.default_ML_verbosity['save_results'] = save_results
    elif 'save_results' not in self.default_ML_verbosity:
        self.default_ML_verbosity['save_results'] = False

    if progress_bar != 'default':
        if progress_bar is True:
            if self.notebook:
                self.default_ML_verbosity['progress_bar'] = tqdm_notebook
            else:
                self.default_ML_verbosity['progress_bar'] = tqdm
        else:
            self.default_ML_verbosity['progress_bar'] = None
    elif 'progress_bar' not in self.default_ML_verbosity:
        if self.notebook:
            self.default_ML_verbosity['progress_bar'] = tqdm_notebook
        else:
            self.default_ML_verbosity['progress_bar'] = tqdm

    if progress_loc != 'default':
        self.default_ML_verbosity['progress_loc'] = progress_loc
    elif 'progress_loc' not in self.default_ML_verbosity:
        self.default_ML_verbosity['progress_loc'] = None

    if pipeline_verbose != 'default':
        self.default_ML_verbosity['pipeline_verbose'] = pipeline_verbose
    elif 'pipeline_verbose' not in self.default_ML_verbosity:
        self.default_ML_verbosity['pipeline_verbose'] = False

    if compute_train_score != 'default':
        self.default_ML_verbosity['compute_train_score'] = compute_train_score
    elif 'compute_train_score' not in self.default_ML_verbosity:
        self.default_ML_verbosity['compute_train_score'] = False

    if show_init_params != 'default':
        self.default_ML_verbosity['show_init_params'] = show_init_params
    elif 'show_init_params' not in self.default_ML_verbosity:
        self.default_ML_verbosity['show_init_params'] = True

    if fold_name != 'default':
        self.default_ML_verbosity['fold_name'] = fold_name
    elif 'fold_name' not in self.default_ML_verbosity:
        self.default_ML_verbosity['fold_name'] = False

    if time_per_fold != 'default':
        self.default_ML_verbosity['time_per_fold'] = time_per_fold
    elif 'time_per_fold' not in self.default_ML_verbosity:
        self.default_ML_verbosity['time_per_fold'] = False

    if score_per_fold != 'default':
        self.default_ML_verbosity['score_per_fold'] = score_per_fold
    elif 'score_per_fold' not in self.default_ML_verbosity:
        self.default_ML_verbosity['score_per_fold'] = False

    if fold_sizes != 'default':
        self.default_ML_verbosity['fold_sizes'] = fold_sizes
    elif 'fold_sizes' not in self.default_ML_verbosity:
        self.default_ML_verbosity['fold_sizes'] = False

    if best_params != 'default':
        self.default_ML_verbosity['best_params'] = best_params
    elif 'best_params' not in self.default_ML_verbosity:
        self.default_ML_verbosity['best_params'] = False

    if best_params_score != 'default':
        self.default_ML_verbosity['best_params_score'] = best_params
    elif 'best_params_score' not in self.default_ML_verbosity:
        self.default_ML_verbosity['best_params_score'] = False

    if save_to_logs != 'default':
        self.default_ML_verbosity['save_to_logs'] = save_to_logs
    elif 'save_to_logs' not in self.default_ML_verbosity:
        self.default_ML_verbosity['save_to_logs'] = False

    if flush != 'default':
        self.default_ML_verbosity['flush'] = flush
    elif 'flush' not in self.default_ML_verbosity:
        self.default_ML_verbosity['flush'] = False

    self._print('Default ML verbosity set within self.default_ML_verbosity.')
    self._print('----------------------')
    for param in self.default_ML_verbosity:

        if param == 'progress_bar':
            if self.default_ML_verbosity[param] is None:
                self._print(param + ':', False)
            else:
                self._print(param + ':', True)

        else:
            self._print(param + ':', self.default_ML_verbosity[param])

    self._print()


def _ML_print(self, *args, **kwargs):
    '''Overriding the print function to allow for
    customizable verbosity. This print is setup with specific
    settings for the Model_Pipeline class, for using Evaluate and Test.

    Parameters
    ----------
    args
        Anything that would be passed to default python print
    '''

    if self.default_ML_verbosity['save_to_logs']:
        _print = self._print
    else:
        _print = print

    # If flush specified, add as kwarg
    if self.default_ML_verbosity['flush']:
        kwargs['flush'] = True

    level = kwargs.pop('level', None)

    # If no level passed, always print
    if level is None:
        _print(*args, **kwargs)

    elif level == 'name' and self.default_ML_verbosity['fold_name']:
        _print(*args, **kwargs)

    elif level == 'time' and self.default_ML_verbosity['time_per_fold']:
        _print(*args, **kwargs)

    elif level == 'score' and self.default_ML_verbosity['score_per_fold']:
        _print(*args, **kwargs)

    elif level == 'size' and self.default_ML_verbosity['fold_sizes']:
        _print(*args, **kwargs)

    elif level == 'params' and self.default_ML_verbosity['best_params']:
        _print(*args, **kwargs)

    elif level == 'cv_score' and self.default_ML_verbosity[
                                 'best_params_score']:
        _print(*args, **kwargs)


def Evaluate(self,
             model_pipeline,
             problem_spec='default',
             splits=3,
             n_repeats=2,
             cv='default',
             train_subjects='train',
             feat_importances=None,
             return_raw_preds=False,
             return_models=False,
             run_name='default',
             only_fold=None,
             base_dtype='float32',
             CV='depreciated'):
    ''' The Evaluate function is one of the main interfaces
    for building and evaluating :class:`Model_Pipeline` on the loaded data.
    Specifically, Evaluate is designed to try and estimate the out of sample
    performance of a passed :class:`Model_Pipeline` on a specific
    ML task (as specified by :class:`Problem_Spec`).
    This estimate is done through a defined CV strategy
    (`splits` and `n_repeats`). While Evaluate's ideal usage is
    an expirimental context for exploring
    different choices of :class:`Model_Pipeline` and then ultimately
    with :func:`Test<BPt_ML.Test>` -
    if used carefully (i.e., dont try 50 Pipelines's and only report
    the one that does best), it can be used
    on a full dataset.

    Parameters
    ------------
    model_pipeline : :class:`Model_Pipeline`

        The passed `model_pipeline` should be an instance of the BPt params
        class :class:`Model_Pipeline`.
        This object defines the underlying model pipeline to be evaluated.

        See :class:`Model_Pipeline` for more information /
        how to create a the model pipeline.

    problem_spec : :class:`Problem_Spec` or 'default', optional

        `problem_spec` accepts an instance of the BPt.BPt_ML
        params class :class:`Problem_Spec`.
        This object is essentially a wrapper around commonly used
        parameters needs to define the context
        the model pipeline should be evaluated in.
        It includes parameters like problem_type, scorer, n_jobs,
        random_state, etc...
        See :class:`Problem_Spec` explicitly for more information
        and for how to create an instance of this object.

        If left as 'default', then will just initialize a
        Problem_Spec with default params.

        ::

            default = 'default'

    splits : int, float, str or list of str, optional
        In every fold of the defined CV strategy, the passed `model_pipeline`
        will be fitted on
        a train fold, and evaluated on a validation fold. This parameter
        controls the type of CV, i.e., specifies what the train and validation
        folds should be. These splits are further determined by the subjects
        passed to `train_subjects`.
        Notably, the splits defined will respect any special split behavior
        as defined in
        :func:`Define_Validation_Strategy<BPt_ML.Define_Validation_Strategy>`.

        Specifically, options for split are:

        - int
            The number of k-fold splits to conduct. (E.g., 3 for a
            3-fold CV).

        - float
            Must be 0 < `splits` < 1, and defines a single
            train-test like split,
            with `splits` as the % of the current training data size
            used as a validation/test set.

        - str
            If a str is passed, then it must correspond to a
            loaded Strat variable. In
            this case, a leave-out-group CV will be used according
            to the value of the
            indicated Strat variable (E.g., a leave-out-site CV scheme).

        - list of str
            If multiple str passed, first determine the
            overlapping unique values from
            their corresponing loaded Strat variables,
            and then use this overlapped
            value to define the leave-out-group CV as described above.

        Note that this defines only the base CV strategy,
        and that the following param `n_repeats`
        is optionally used to replicate this base strategy, e.g.,
        for a twice repeated train-test split evaluation.
        Note further that `n_repeats` will work with any of these options,
        but say in the case of
        a leave out group CV, it would be awfully redundant, versus,
        with a passed float value, very reasonable.

        ::

            default = 3

    n_repeats : int, optional
        Given the base CV defined / described in the `splits` param, this
        parameter further controls if the defined train/val splits should
        be repeated
        (w/ different random splits in all cases but the
        leave-out-group passed str option).

        For example, if `n_repeats` is set to 2, and `splits` is 3,
        then a twice repeated 3-fold CV
        will be performed, and results returned with respect to this strategy.

        It can be a good idea to set multiple `n_repeats`
        (assuming enough computation power), as it can
        help you spot cases where you may not have enough training subjects
        to get stable behavior, e.g.,
        say you run a three times repeated 3 fold CV,
        if the mean validation scores from each 3-fold are
        all very close to each other, then you know that
        1 repeat is likely enough. If instead the macro std in
        score (the std from in this case those 3 scores) is high,
        then it indicates you may not have enough subjects
        to get stable results from just one 3-fold CV, and
        that you might want to consider changing some settings.

        ::

            default = 2

    cv : 'default' or CV params object, optional
        If left as default 'default', use the class defined CV behavior
        for the splits, otherwise can pass custom behavior

        ::

            default = 'default'

    train_subjects : :ref:`Subjects`, optional
        This parameter determines the set of training subjects which are
        used in this call to `Evaluate`. Note, this parameter is distinct to
        the `subjects` parameter within :class:`Problem_Spec`, which is
        applied after
        selecting the subset of `train_subjects` specified here.
        These subjects are
        used as the input to `Evaluate`, i.e.,
        so typically any subjects data you want
        to remain untouched (say your global test subjects)
        are considered within `Evaluate`,
        and only those explicitly passed here are.

        By default, this value will be set to the special str indiciator
        'train', which specifies that the full set of globally defined
        training subjects
        (See: :func:`Define_Train_Test_Split`),
        should be used. Other special str indicators
        include 'all' to select all subjects, and 'test'
        to select the test set subjects.

        If `subjects` is passed a str, and that str is not one of the str
        indicators listed above, then it will be interpretted as the location
        of file in which to read subjects from
        (assuming one subjects per line).

        `subjects` may also be a custom array-like of subjects to use.

        See :ref:`Subjects` for how to correctly format input and for
        other special options.

        ::

            default = 'train'

    feat_importances : :class:`Feat_Importance` list of, str or None, optional

        If passed None, by default, no feature importances will be saved.

        Alternatively, one may pass the keyword 'base', to indicate that
        the base feature importances - those automatically calculated
        by base objects (e.g., beta weights from linear models) be
        saved. In this case, the object Feat_Importance('base') will be made.

        Otherwise, for more detailed control provide here either
        a single, or list of
        :class:`Feat_Importance` param objects
        in which to specify what importance values, and with what
        settings should be computed.
        See the base :class:`Feat_Importance` object for more information
        on how to specify
        these objects.

        See :ref:`Feat Importances` to learn more about feature importances
        generally.

        In this case of a passed list, all passed Feat_Importances
        will attempt to be
        computed.

        ::

            default = None

    return_raw_preds : bool, optional
        If True, return the raw predictions from each fold.

        ::

            default = False

    return_models : bool, optional
        If True, return the trained models from each evaluation.

        ::

            default = False

    run_name : str or 'default', optional
        Each run of Evaluate can be optionally associated with
        a specific `run_name`. This name
        is used if `save_results` in
        :func:`Set_Default_ML_Verbosity<BPt_ML.Set_Default_ML_Verbosity>`
        is set to True,
        then will be used as the name output from Evaluate as saved as in
        the specific log_dr
        (if any, and as set when Init'ing the
        :class:`BPt_ML <BPt.BPt_ML>` class object),
        with '.eval' appended to the name.

        If left as 'default', will come up with a kind of
        terrible name passed on the underlying
        model used in the passed `model_pipeline`.

        ::

            default = 'default'

    only_fold : int or None, optional
        This is a special parameter used to only
        Evaluate a specific fold of the specified runs to
        evaluate. Keep as None to ignore.

        ::

            default = None

    base_dtype : numpy dtype
        The dataset is cast to a numpy array of float.
        This parameter can be used to change the default
        behavior, e.g., if more resolution or less is needed.

        ::

            default = 'float32'

    CV : 'depreciated'
        Switching to passing cv parameter as cv instead of CV.
        For now if CV is passed it will still work as if it were
        passed as cv.

        ::

            default = 'depreciated'

    Returns
    ----------
    results : dict
        Dictionary containing:
        'summary_scores', A list representation of the
        printed summary scores, where the 0 index is the mean,
        1 index is the macro std, then second index is the micro std.
        'train_summary_scores', Same as summary scores, but only exists
        if train scores are computed.
        'raw_scores', a numpy array of numpy arrays,
        where each internal array contains the raw scores as computed for
        all passed in scorers, computed for each fold within
        each repeat. e.g., array will have a length of `n_repeats` * number of
        folds, and each internal array will have the same length as
        the number of
        scorers. Optionally, this could instead return a list containing as
        the first element the raw training score in this same format,
        and then the raw testing scores.
        'raw_preds', A pandas dataframe containing the raw predictions
        for each subject, in the test set, and
        'FIs' a list where each element corresponds
        to a passed feature importance.

    Notes
    ----------
    Prints by default the following for each scorer,

    float
        The mean macro score (as set by input scorer) across each
        repeated K-fold.

    float
        The standard deviation of the macro score (as set by input scorer)
        across each repeated K-fold.

    float
        The standard deviation of the micro score (as set by input scorer)
        across each fold with the repeated K-fold.

    '''

    # Perform pre-modeling check
    self._premodel_check()

    # Run initial model pipeline check
    model_pipeline = model_pipeline_check(model_pipeline, self.all_data)

    # Should save the params used here*** before any preproc done
    run_name = get_avaliable_run_name(run_name, model_pipeline)

    # Get the the train subjects to use
    _train_subjects = self._get_subjects_to_use(train_subjects)

    # Proc feat importances
    if feat_importances == 'base':
        feat_importances = Feat_Importance(obj='base')
    elif feat_importances == 'default':
        raise RuntimeError('feat_importaces == default is depreciated')

    if CV != 'depreciated':
        print('Warning: Passing CV is depreciated. Please change to',
              'passing as cv instead.')

        # For now, let it still work
        cv = CV

    # Proc. CV
    if cv == 'default':
        cv_obj = self.cv
    else:
        cv_obj = self._get_cv(cv)

    # Pre-proc problem spec, set as copy ps, right before print
    ps = self._preproc_problem_spec(problem_spec)

    # Run checks before print
    model_pipeline._proc_checks()

    # Print the params being used
    if self.default_ML_verbosity['show_init_params']:

        model_pipeline.print_all(self._print)
        ps.print_all(self._print)

        self._print('Evaluate Params')
        self._print('---------------')
        self._print('splits =', splits)
        self._print('n_repeats =', n_repeats)
        self._print('cv =', cv)
        self._print('train_subjects =', train_subjects)
        self._print('feat_importances =', feat_importances)
        self._print('len(train_subjects) =', len(_train_subjects),
                    '(before overlap w/ problem_spec.subjects)')
        self._print('run_name =', run_name)
        self._print()

    # Init the Model_Pipeline object with modeling params
    self._init_evaluator(
        model_pipeline=model_pipeline,
        ps=ps,
        cv=cv_obj,
        feat_importances=feat_importances,
        return_raw_preds=return_raw_preds,
        return_models=return_models,
        base_dtype=base_dtype)

    # Get the Eval splits
    _, splits_vals, _ = self._get_split_vals(splits)

    # Evaluate the model
    train_scores, scores, results =\
        self.evaluator.Evaluate(self.all_data, _train_subjects,
                                splits, n_repeats, splits_vals,
                                only_fold=only_fold)

    # If only fold is not None, set n_repeats = 1
    if only_fold is not None:
        n_repeats = 1

    if 'FIs' in results:
        for fi in results['FIs']:
            fi.set_target(ps.target)
            fi.set_run_name(run_name)

    self._print()

    # Print out summary stats for all passed scorers
    if self.default_ML_verbosity['compute_train_score']:
        score_list = [train_scores, scores]
        score_type_list = ['Training', 'Validation']
    else:
        score_list = [scores]
        score_type_list = ['Validation']

    summary_dfs = {}

    for scrs, name in zip(score_list, score_type_list):

        summary_scores = self._handle_scores(scrs, name,
                                             ps.weight_scorer,
                                             n_repeats, run_name,
                                             self.evaluator.n_splits_,
                                             summary_dfs)

        if name == 'Validation':
            results['summary_scores'] = summary_scores
        else:
            results['train_summary_scores'] = summary_scores

    # Add scores to results
    results['raw_scores'] = score_list

    # Create single scores
    for scorer_str, sum_scores in zip(self.evaluator.scorer_strs,
                                      results['summary_scores']):
        results[scorer_str] = sum_scores[0]

    if 'train_summary_scores' in results:
        for scorer_str, sum_scores in zip(self.evaluator.scorer_strs,
                                          results['train_summary_scores']):
            results['train_' + scorer_str] = sum_scores[0]

    # Saves based on verbose setting
    self._save_results(results, run_name + '.eval')

    return results


def Test(self,
         model_pipeline,
         problem_spec='default',
         train_subjects='train',
         test_subjects='test',
         feat_importances=None,
         return_raw_preds=False,
         return_models=False,
         run_name='default',
         base_dtype='float32'):
    ''' The test function is one of the main interfaces for testing a specific
    :class:`Model_Pipeline`. Test is conceptually different from
    :func:`Evaluate<BPt_ML.Evaluate>`
    in that it is designed to contrust / train a :class:`Model_Pipeline`
    on one discrete set of `train_subjects`
    and evaluate it on a further discrete set of `test_subjects`.
    Otherwise, these functions are very simmilar as
    they both evaluate a :class:`Model_Pipeline` as defined in the context of
    a :class:`Problem_Spec`, and return
    similar output.

    Parameters
    ------------
    model_pipeline : :class:`Model_Pipeline`
        The passed `model_pipeline` should be an instance of the
        BPt params class :class:`Model_Pipeline`.
        This object defines the underlying model pipeline to be evaluated.

        See :class:`Model_Pipeline` for more information / how to
        create a the model pipeline.

    problem_spec : :class:`Problem_Spec` or 'default', optional

        `problem_spec` accepts an instance of the BPt.BPt_ML
        params class :class:`Problem_Spec`.
        This object is essentially a wrapper around commonly used
        parameters needs to define the context
        the model pipeline should be evaluated in.
        It includes parameters like problem_type, scorer, n_jobs,
        random_state, etc...
        See :class:`Problem_Spec` explicitly for more information
        and for how to create an instance of this object.

        If left as 'default', then will just initialize a
        Problem_Spec with default params.

        ::

            default = 'default'

    train_subjects : str, array-like or Value_Subset, optional
        This parameter determines the set of training subjects which are
        used to train the passed instance of :class:`Model_Pipeline`.

        Note, this parameter and `test_subjects` are distinct,
        but complementary to
        the `subjects` parameter within :class:`Problem_Spec`,
        which is applied after
        selecting the subset of `train_subjects` specified here.

        By default, this value will be set to the special
        str indiciator 'train', which
        specifies that the full set of globally defined training subjects
        (See: :func:`Define_Train_Test_Split`),
        should be used. Other special str indicators
        include 'all' to select all subjects, and 'test'
        to select the test set subjects.

        If `subjects` is passed a str, and that str is not one of
        the str indicators listed above,
        then it will be interpretted as the location of file in which
        to read subjects from (assuming one subjects per line).

        `subjects` may also be a custom array-like of subjects to use.

        Lastly, a special wrapper, Value_Subset, can also be used
        to specify more specific,
        specifically value specific, subsets of subjects to use.
        See :class:`Value_Subset` for how this input wrapper can be used.

        If passing custom input here, be warned that you NEVER
        want to pass an overlap of
        subjects between `train_subjects` and `test_subjects`

        ::

            default = 'train'

    test_subjects : str, array-like or Value_Subset, optional
        This parameter determines the set of testing subjects which are
        used to evaluate the passed instance of :class:`Model_Pipeline`,
        after it
        has been trained on the passed `train_subjects`.

        Note, this parameter and `train_subjects` are distinct,
        but complementary to
        the `subjects` parameter within :class:`Problem_Spec`,
        which is applied after
        selecting the subset of `test_subjects` specified here.

        By default, this value will be set to the special str
        indiciator 'test', which
        specifies that the full set of globally defined
        test subjects
        (See: :func:`Define_Train_Test_Split`),
        should be used. Other special str indicators
        include 'all' to select all subjects, and 'train'
        to select the train set subjects.

        If `subjects` is passed a str, and that str is not one
        of the str indicators listed above,
        then it will be interpretted as the location of file in which
        to read subjects from (assuming one subjects per line).

        `subjects` may also be a custom array-like of subjects to use.

        Lastly, a special wrapper, Value_Subset, can also be used
        to specify more specific,
        specifically value specific, subsets of subjects to use.
        See :class:`Value_Subset` for how this input wrapper can be used.

        If passing custom input here, be warned that you NEVER
        want to pass an overlap of
        subjects between `train_subjects` and `test_subjects`

        ::

            default = 'test'

    feat_importances : :class:`Feat_Importance` list of, str or None, optional

        If passed None, by default, no feature importances will be saved.

        Alternatively, one may pass the keyword 'base', to indicate that
        the base feature importances - those automatically calculated
        by base objects (e.g., beta weights from linear models) be
        saved. In this case, the object Feat_Importance('base') will be made.

        Otherwise, for more detailed control provide here either
        a single, or list of
        :class:`Feat_Importance` param objects
        in which to specify what importance values, and with what
        settings should be computed.
        See the base :class:`Feat_Importance` object for more information
        on how to specify
        these objects.

        See :ref:`Feat Importances` to learn more about feature importances
        generally.

        In this case of a passed list, all passed Feat_Importances
        will attempt to be
        computed.

        ::

            default = None

    return_raw_preds : bool, optional
        If True, return the raw predictions from each fold.

        ::

            default = False

    return_models : bool, optional
        If True, return the trained models from each evaluation.

        ::

            default = False

    run_name : str or 'default', optional
        Each run of test can be optionally
        associated with a specific `run_name`. This name
        is used if `save_results` in
        :func:`Set_Default_ML_Verbosity<BPt_ML.Set_Default_ML_Verbosity>`
        is set to True,
        then will be used as the name output from Test as saved as in the
        specific log_dr
        (if any, and as set when Init'ing
        the :class:`BPt_ML <BPt.BPt_ML>` class object),
        with .test appended to the name.

        If left as 'default', will come up with a kind of
        terrible name passed on the underlying
        model used in the passed `model_pipeline`.

        ::

            default = 'default'

    base_dtype : numpy dtype
        The dataset is cast to a numpy array of float.
        This parameter can be used to change the default
        behavior, e.g., if more resolution or less is needed.

        ::

            default = 'float32'

    Returns
    ----------
    results : dict
        Dictionary containing:
        'scores', the score on the test set by each scorer,
        'raw_preds', A pandas dataframe containing the raw predictions
        for each subject, in the test set, and 'FIs' a list where
        each element corresponds to a passed feature importance.
    '''

    # Perform pre-modeling check
    self._premodel_check()

    # Run initial model pipeline check
    model_pipeline = model_pipeline_check(model_pipeline, self.all_data)

    # Get a free run name
    run_name = get_avaliable_run_name(run_name, model_pipeline)

    # Get the the train subjects + test subjects to use
    _train_subjects = self._get_subjects_to_use(train_subjects)
    _test_subjects = self._get_subjects_to_use(test_subjects)

    # Proc feat importances
    if feat_importances == 'base':
        feat_importances = Feat_Importance(obj='base')
    elif feat_importances == 'default':
        raise RuntimeError('feat_importaces == default is depreciated')

    # Pre-proc problem spec, set as copy ps, right before print
    ps = self._preproc_problem_spec(problem_spec)

    # Run checks before print
    model_pipeline._proc_checks()

    # Print the params being used
    if self.default_ML_verbosity['show_init_params']:

        model_pipeline.print_all(self._print)
        ps.print_all(self._print)

        self._print('Test Params')
        self._print('---------------')
        self._print('train_subjects =', train_subjects)
        self._print('len(train_subjects) =', len(_train_subjects),
                    '(before overlap w/ problem_spec.subjects)')
        self._print('test_subjects =', test_subjects)
        self._print('len(test_subjects) =', len(_test_subjects),
                    '(before overlap w/ problem_spec.subjects)')
        self._print('feat_importances =', feat_importances)
        self._print('run_name =', run_name)
        self._print()

    # Init the Model_Pipeline object with modeling params
    self._init_evaluator(
        model_pipeline=model_pipeline,
        ps=ps,
        cv=None,  # Test doesn't use cv
        feat_importances=feat_importances,
        return_raw_preds=return_raw_preds,
        return_models=return_models,
        base_dtype=base_dtype)

    # Train the model w/ selected parameters and test on test subjects
    train_scores, scores, results =\
        self.evaluator.Test(self.all_data, _train_subjects,
                            _test_subjects)

    results['scores'] = scores

    # Set run name
    if 'FIs' in results:
        for fi in results['FIs']:
            fi.set_target(ps.target)
            fi.set_run_name(run_name)

    # Print out score for all passed scorers
    scorer_strs = self.evaluator.scorer_strs
    self._print()

    score_list, score_type_list = [], []
    if self.default_ML_verbosity['compute_train_score']:
        score_list.append(train_scores)
        score_type_list.append('Training')
        results['train_scores'] = scores

    score_list.append(scores)
    score_type_list.append('Testing')

    for s, name in zip(score_list, score_type_list):

        self._print(name + ' Scores')
        self._print(''.join('_' for i in range(len(name) + 7)))

        for i in range(len(scorer_strs)):

            scorer_name = scorer_strs[i]
            self._print('Scorer: ', scorer_name)

            scr = s[i]
            if len(scr.shape) > 0:

                targets_key = self.evaluator.targets_key

                for score_by_class, class_name in zip(scr, targets_key):
                    self._print('for target class: ', class_name)
                    self._print(name + ' Score: ', score_by_class)
                    self._print()

            else:
                self._print(name + ' Score: ', scr)
                self._print()

    # Add single scores
    for i in range(len(scorer_strs)):
        results[scorer_strs[i]] = scores[i]

        if 'train_scores' in results:
            results['train_' + scorer_strs[i]] = results['train_scores'][i]

    # Save based on default verbosity
    self._save_results(results, run_name + '.test')

    return results


def _premodel_check(self):
    '''Internal helper function
    has been called, and to force a train/test split if not already done.
    Will also call Set_Default_ML_Params if not already called.
    '''

    if self.all_data is None:
        self._print('Calling Prepare_All_Data()',
                    'to change the default merge behavior',
                    'call it again!')
        self.Prepare_All_Data()

    if self.train_subjects is None:

        raise RuntimeError('No train-test set defined!',
                           'If this is intentional, call Train_Test_Split',
                           'with test_size = 0')

    if self.default_ML_verbosity == {}:

        self._print('Setting default ML verbosity settings!')
        self._print('Note, if the following values are not desired,',
                    'call self.Set_Default_ML_Verbosity()')

        self.Set_Default_ML_Verbosity()


def _preproc_param_search(self, object, n_jobs,
                          problem_type,
                          random_state):

    param_search = getattr(object, 'param_search')
    if param_search is None:
        return False

    # Proc param search cv
    if param_search.cv == 'default':
        search_cv = self.cv
    else:
        search_cv =\
            self._get_cv(param_search.cv, show=False)
    param_search.set_cv(search_cv)

    # Set scorer
    param_search.set_scorer(problem_type)

    # Set random_state
    param_search.set_random_state(random_state)

    # Set n_jobs
    param_search.set_n_jobs(n_jobs)

    # Set split vals
    _, split_vals, _ =\
        self._get_split_vals(param_search.splits)
    param_search.set_split_vals(split_vals)

    # Check mp_context for default
    if param_search.mp_context == 'default':
        param_search.mp_context = self.mp_context

    # Set new proc'ed
    setattr(object, 'param_search', param_search)

    return True


def _preproc_cv_splits(self, obj, random_state):

    # Proc cv
    if obj.cv == 'default':
        cv = self.cv
    else:
        cv = self._get_cv(obj.cv, show=False)

    # If CV_Splits
    if isinstance(obj, CV_Splits):

        # Set split vals
        _, split_vals, _ =\
            self._get_split_vals(obj.splits)

        obj.setup(cv=cv, split_vals=split_vals, random_state=random_state)

    # Otherwise must be CV_Split
    else:
        obj.setup(cv=cv, random_state=random_state)


def _preproc_model_pipeline(self, model_pipeline, n_jobs,
                            problem_type, random_state):

    model_pipeline = model_pipeline_check(model_pipeline, self.all_data)

    # Set values across each pipeline pieces params
    model_pipeline.preproc(n_jobs)

    # Pre-proc param search
    has_param_search =\
        self._preproc_param_search(model_pipeline, n_jobs,
                                   problem_type, random_state)

    # If there is a param search set for the Model Pipeline,
    # set n_jobs, the value to pass in the nested check, to 1
    if has_param_search:
        n_jobs = 1

    def nested_model_check(obj):

        # Check for Model or Ensemble
        if isinstance(obj, Model) or isinstance(obj, Ensemble):
            self._preproc_param_search(obj, n_jobs, problem_type, random_state)

        if isinstance(obj, list):
            [nested_model_check(o) for o in obj]
            return

        elif isinstance(obj, dict):
            [nested_model_check(obj[k]) for k in obj]

        elif hasattr(obj, 'get_params'):
            for param in obj.get_params(deep=False):
                nested_model_check(getattr(obj, param))
            return

        return

    # Run nested check
    nested_model_check(model_pipeline)

    def nested_cv_splits_check(obj):

        if isinstance(obj, CV_Splits) or isinstance(obj, CV_Split):
            self._preproc_cv_splits(obj, random_state)

        elif isinstance(obj, list):
            [nested_cv_splits_check(o) for o in obj]

        elif isinstance(obj, dict):
            [nested_cv_splits_check(obj[k]) for k in obj]

        elif hasattr(obj, 'get_params'):
            for param in obj.get_params(deep=False):
                nested_cv_splits_check(getattr(obj, param))

    # Run nested check
    nested_cv_splits_check(model_pipeline)

    return model_pipeline


def _preproc_problem_spec(self, problem_spec):

    # Check if problem_spec is left as default
    if problem_spec == 'default':
        problem_spec = Problem_Spec()

    # Set ps to copy of problem spec and init
    ps = deepcopy(problem_spec)

    # Update target with actual target key
    target_key = self._get_targets_key(ps.target)
    ps.set_params(target=target_key)

    # Replace problem_spec type w/ correct if passed short hands
    pt = ps.problem_type
    if pt == 'default':

        # For future multi-label support...
        if isinstance(target_key, list):
            pt = 'multilabel'
        else:
            pt = get_target_type(self.all_data[target_key])

        self._print('problem_spec problem_type ==  default, setting as:',
                    pt)

    elif pt == 'b':
        pt = 'binary'

    elif pt == 'c' or pt == 'multiclass':
        pt = 'categorical'

    elif pt == 'f' or pt == 'float':
        pt = 'regression'

    ps.problem_type = pt

    # Check for if default scorer
    if ps.scorer == 'default':
        default_scorers = {'regression': ['explained_variance',
                                          'neg_mean_squared_error'],
                           'binary': ['matthews', 'roc_auc',
                                      'balanced_accuracy'],
                           'categorical': ['matthews', 'roc_auc_ovr',
                                           'balanced_accuracy']}
        ps.scorer = default_scorers[pt]
        self._print('problem_spec scorer ==  default, setting as:', ps.scorer)

    # Proc subjects to use
    final_subjects = self._get_subjects_to_use(ps.subjects)
    ps.set_final_subjects(final_subjects)

    # Set by class defaults
    if ps.n_jobs == 'default':
        ps.n_jobs = self.n_jobs

    if ps.random_state == 'default':
        ps.random_state = self.random_state

    # If any input has changed, manually (i.e., not by problem_spec init)
    ps._proc_checks()

    return ps


def _get_split_vals(self, splits):

    if isinstance(splits, int) or isinstance(splits, float):
        split_names, split_vals, sv_le = None, None, None

    else:
        split_names = self._add_strat_u_name(splits)

        if isinstance(split_names, str):
            split_names = [split_names]

        split_vals, sv_le =\
            get_unique_combo_df(self.strat, split_names)

    return split_names, split_vals, sv_le


def _get_subjects_to_use(self, subjects_to_use):

    # If str passed, either loc to load, or train, test, all
    if isinstance(subjects_to_use, str):
        if subjects_to_use == 'all':
            subjects = self.all_data.index
        elif subjects_to_use == 'train':
            subjects = self.train_subjects
        elif subjects_to_use == 'test':
            subjects = self.test_subjects
        else:
            subjects = self._load_set_of_subjects(loc=subjects_to_use)

    # Other case is if passed value subset, determine the values to use
    elif is_value_subset(subjects_to_use):

        split_names, split_vals, sv_le =\
            self._get_split_vals(subjects_to_use.name)

        selected = split_vals[split_vals == subjects_to_use.value]
        subjects = set(selected.index)

        rev_values = reverse_unique_combo_df(selected, sv_le)[0]

        last_subjects_names = []
        for strat_name, value in zip(split_names, rev_values):
            if self.strat_u_name in strat_name:
                strat_name = strat_name.replace(self.strat_u_name, '')

            last_subjects_names.append((strat_name, value))

        self._print('subjects set to: ',
                    last_subjects_names)
        self._print()

    # Can also be values subset
    elif is_values_subset(subjects_to_use):

        # Add strat u name to name if not already added
        name = self._add_strat_u_name(subjects_to_use.name)

        # Extract the values as list
        values = conv_to_list(subjects_to_use.values)

        # Check name to make sure loaded
        if name not in self.all_data:
            raise ValueError(name, 'is not a valid loaded Strat feature!')

        # Get by value
        subjects = self.all_data[self.all_data[name].isin(values)].index

        # Make sure subjects is set-like
        subjects = set(list(subjects))

    # Lastly, if not the above, assume it is an array-like of subjects
    else:
        subjects = self._load_set_of_subjects(subjects=subjects_to_use)

    return subjects


def get_pipeline(self, model_pipeline, problem_spec,
                 progress_loc=None, has_search=False):

    # If has search is False, means this is the top level
    # or the top level didnt have a search
    if not has_search:
        if model_pipeline.param_search is None:
            has_search = False
        else:
            has_search = True

    # If either this set of model_pipeline params or the parent
    # params had search params, then a copy of problem spec with n_jobs set
    # to 1 should be passed to children get pipelines,
    if has_search:
        nested_ps = copy.deepcopy(problem_spec)
        nested_ps.n_jobs = 1
    else:
        nested_ps = problem_spec

    def nested_check(obj):

        if hasattr(obj, 'obj') and isinstance(obj.obj, Model_Pipeline):

            nested_pipe, nested_pipe_params =\
                self.get_pipeline(model_pipeline=obj.obj,
                                  problem_spec=nested_ps,
                                  progress_loc=progress_loc,
                                  has_search=has_search)

            # Set obj as nested pipeline
            setattr(obj, 'obj', nested_pipe)

            # Set obj's params as the nested_pipe_params
            setattr(obj, 'params', nested_pipe_params)
            return

        if isinstance(obj, list):
            [nested_check(o) for o in obj]
            return

        elif isinstance(obj, dict):
            [nested_check(obj[k]) for k in obj]
            return

        if hasattr(obj, 'get_params'):
            for param in obj.get_params(deep=False):
                nested_check(getattr(obj, param))
            return

    # Run nested check on model_pipeline
    nested_check(model_pipeline)

    # Preproc model
    model_pipeline = self._preproc_model_pipeline(model_pipeline,
                                                  problem_spec.n_jobs,
                                                  problem_spec.problem_type,
                                                  problem_spec.random_state)

    # Init data scopes
    self.Data_Scopes.set_all_keys(problem_spec)

    # Return the pipeline
    return get_pipe(pipeline_params=model_pipeline,
                    problem_spec=problem_spec,
                    Data_Scopes=self.Data_Scopes,
                    progress_loc=progress_loc,
                    verbose=self.default_ML_verbosity['pipeline_verbose'])


def _init_evaluator(self, model_pipeline, ps,
                    cv, feat_importances, return_raw_preds,
                    return_models, base_dtype):

    # Make copies of the passed pipeline
    # and only make changes and pass along the copies
    pipe = copy.deepcopy(model_pipeline)

    # Calling get pipeline performs preproc on model_pipeline
    # and Data_Scopes
    model, _ =\
        self.get_pipeline(
            pipe, ps, progress_loc=self.default_ML_verbosity['progress_loc'])

    # Set the evaluator obj
    self.evaluator =\
        Evaluator(model=model,
                  problem_spec=ps,
                  cv=cv,
                  all_keys=self.Data_Scopes.all_keys,
                  feat_importances=feat_importances,
                  return_raw_preds=return_raw_preds,
                  return_models=return_models,
                  verbosity=self.default_ML_verbosity,
                  base_dtype=base_dtype,
                  _print=self._ML_print)


def _handle_scores(self, scores, name, weight_scorer, n_repeats, run_name,
                   n_splits, summary_dfs):

    all_summary_scores = []
    summary_dfs[name] = pd.DataFrame()

    scorer_strs = self.evaluator.scorer_strs

    self._print(name + ' Scores')
    self._print(''.join('_' for i in range(len(name) + 7)))

    weight_scorers = conv_to_list(weight_scorer, len(scorer_strs))

    for i in range(len(scorer_strs)):

        # Weight outputed scores if requested
        if weight_scorers[i]:
            weights = self.evaluator.n_test_per_fold
        else:
            weights = None

        scorer_name = scorer_strs[i]
        self._print('Scorer: ', scorer_name)
        score_by_scorer = scores[:, i]

        if len(score_by_scorer[0].shape) > 0:
            by_class = [[score_by_scorer[i][j] for i in
                        range(len(score_by_scorer))] for j in
                        range(len(score_by_scorer[0]))]

            summary_scores_by_class =\
                [compute_micro_macro(class_scores, n_repeats,
                 n_splits, weights=weights) for class_scores in by_class]

            targets_key = self.evaluator.targets_key
            classes = self.evaluator.classes

            class_names =\
                self.targets_encoders[targets_key].inverse_transform(
                    classes.astype(int))

            for summary_scores, class_name in zip(summary_scores_by_class,
                                                  class_names):

                self._print('Target class: ', class_name)
                self._print_summary_score(name, summary_scores,
                                          n_repeats, run_name,
                                          scorer_name, class_name,
                                          weights=weights)

            all_summary_scores.append(summary_scores_by_class)

        else:

            # Compute macro / micro summary of scores
            summary_scores = compute_micro_macro(score_by_scorer,
                                                 n_repeats,
                                                 n_splits,
                                                 weights=weights)

            self._print_summary_score(name, summary_scores,
                                      n_repeats, run_name,
                                      scorer_name, weights=weights)

            all_summary_scores.append(summary_scores)

    return all_summary_scores


def _print_summary_score(self, name, summary_scores, n_repeats, run_name,
                         scorer_name, class_name=None, weights=None):

    mn = 'Mean'
    if weights is not None:
        mn = 'Weighted ' + mn

    self._print(mn + ' ' + name + ' score: ', summary_scores[0])

    if n_repeats > 1:
        self._print('Micro Std in ' + name + ' score: ',
                    summary_scores[1])
        self._print('Macro Std in ' + name + ' score: ',
                    summary_scores[2])
    else:
        self._print('Std in ' + name + ' score: ',
                    summary_scores[1])

    self._print()


def _save_results(self, results, save_name):

    if self.default_ML_verbosity['save_results'] and self.log_dr is not None:

        save_dr = os.path.join(self.exp_log_dr, 'results')
        os.makedirs(save_dr, exist_ok=True)

        save_spot = os.path.join(save_dr, save_name)

        append, cnt = '', 0
        while os.path.exists(save_spot+append):
            cnt += 1
            append = str(cnt)

        with open(save_spot+append, 'wb') as f:
            pkl.dump(results, f)


def model_pipeline_check(model_pipeline, data):

    # Add checks on Model_Pipeline
    if not isinstance(model_pipeline, Model_Pipeline):

        # Check for if model str first
        if isinstance(model_pipeline, str):
            model_pipeline = Model(obj=model_pipeline)

        # In case of passed valid single model, wrap in Model_Pipeline
        if hasattr(model_pipeline, '_is_model'):
            model_pipeline = Model_Pipeline(imputers=None,
                                            model=model_pipeline)
        else:
            raise RuntimeError('model_pipeline must be a Model_Pipeline',
                               ' model str or Model-like')

    # Early check to see if imputer is needed
    model_pipeline.check_imputer(data)

    return model_pipeline
