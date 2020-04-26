"""
_ML.py
====================================
Main class extension file for the Machine Learning functionality
"""
import os
import pickle as pkl

from tqdm import tqdm, tqdm_notebook

from .Input_Tools import is_value_subset
from ..helpers.Data_Helpers import get_unique_combo_df, reverse_unique_combo_df
from ..helpers.ML_Helpers import (compute_macro_micro, conv_to_list,
                                  get_avaliable_run_name)
from ..pipeline.Model_Pipeline import Model_Pipeline


def Set_Default_ML_Verbosity(
 self, save_results='default', progress_bar='default', compute_train_score='default',
 show_init_params='default', fold_name='default',
 time_per_fold='default', score_per_fold='default', fold_sizes='default',
 best_params='default', save_to_logs='default'):
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
        (default = 'default')

    progress_bar : bool, optional
        If True, a progress bar, implemented in the python
        library tqdm, is used to show progress during use of
        :func:`Evaluate` , If False, then no progress bar is shown.
        This bar should work both in a notebook env and outside one,
        assuming self.notebook has been set correctly.

        if 'default', and not already defined, set to True.
        (default = 'default')

    compute_train_score : bool, optional
        If True, then metrics and raw preds will also be
        computed on the training set in addition to just the 
        eval or testing set.

        if 'default', and not already defined, set to False.
        (default = 'default')

    show_init_params : bool, optional
        If True, then print/show the parameters used before running
        Evaluate / Test. If False, then don't print the params used.

        if 'default', and not already defined, set to True.
        (default = 'default')

    fold_name : bool, optional
        If True, prints a rough measure of progress via
        printing out the current fold (somewhat redundant with the
        progress bar if used, except if used with other params, e.g.
        time per fold, then it is helpful to have the time printed
        with each fold). If False, nothing is shown.

        if 'default', and not already defined, set to False.
        (default = 'default')

    time_per_fold : bool, optional
        If True, prints the full time that a fold took to complete.

        if 'default', and not already defined, set to False.
        (default = 'default')

    score_per_fold : bool, optional
        If True, displays the score for each fold, though slightly less
        formatted then in the final display.

        if 'default', and not already defined, set to False.
        (default = 'default')

    fold_sizes : bool, optional
        If True, will show the number of subjects within each train
        and val/test fold.

        if 'default', and not already defined, set to False.
        (default = 'default')

    best_params : bool, optional
        If True, print the best search params found after every
        param search.

    save_to_logs : bool, optional
        If True, then when possible, and with the selected model
        verbosity options, verbosity ouput will be saved to the
        log file.

        if 'default', and not already defined, set to False.
        (default = 'default')
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

    if save_to_logs != 'default':
        self.default_ML_verbosity['save_to_logs'] = save_to_logs
    elif 'save_to_logs' not in self.default_ML_verbosity:
        self.default_ML_verbosity['save_to_logs'] = False

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


def Evaluate(self,
             model_pipeline,
             problem_spec,
             splits=3,
             n_repeats=2,
             train_subjects='train',
             run_name='default'):
    ''' The Evaluate function is one of the main interfaces
    for building and evaluating :class:`Model_Pipeline` on the loaded data.
    Specifically, Evaluate is designed to try and estimate the out of sample
    performance of a passed :class:`Model_Pipeline` on a specific
    ML task (as specified by :class:`Problem_Spec`).
    This estimate is done through a defined CV strategy
    (`splits` and `n_repeats`). While Evaluate's ideal usage is
    an expirimental context for exploring
    different choices of :class:`Model_Pipeline` and then ultimately
    with :func:`Test<ABCD_ML.Test>` -
    if used carefully (i.e., dont try 50 Pipelines's and only report
    the one that does best), it can be used
    on a full dataset.
    
    Parameters
    ------------
    model_pipeline : :class:`Model_Pipeline`

        The passed `model_pipeline` should be an instance of the ABCD_ML params class :class:`Model_Pipeline`.
        This object defines the underlying model pipeline to be evaluated.

        See :class:`Model_Pipeline` for more information / how to create a the model pipeline.

    problem_spec : :class:`Problem_Spec`

        `problem_spec` accepts an instance of the ABCD_ML params class :class:`Problem_Spec`.
        This object is essentially a wrapper around commonly used parameters needs to define the context
        the model pipeline should be evaluated in. It includes parameters like problem_type, metric, n_jobs, random_state, etc...
        See :class:`Problem_Spec` explicitly for more information and for how to create an instance of this object.

    splits : int, float, str or list of str, optional
        In every fold of the defined CV strategy, the passed `model_pipeline` will be fitted on
        a train fold, and evaluated on a validation fold. This parameter
        controls the type of CV, i.e., specifies what the train and validation
        folds should be. These splits are further determined by the subjects passed to `train_subjects`.
        Notably, the splits defined will respect any special split behavior as defined in
        :func:`Define_Validation_Strategy<ABCD_ML.Define_Validation_Strategy>`.

        Specifically, options for split are:

        - int
            The number of k-fold splits to conduct. (E.g., 3 for a
            3-fold CV).

        - float
            Must be 0 < `splits` < 1, and defines a single train-test like split,
            with `splits` as the % of the current training data size used as a validation/test set.

        - str
            If a str is passed, then it must correspond to a loaded Strat variable. In
            this case, a leave-out-group CV will be used according to the value of the
            indicated Strat variable (E.g., a leave-out-site CV scheme).

        - list of str
            If multiple str passed, first determine the overlapping unique values from
            their corresponing loaded Strat variables, and then use this overlapped
            value to define the leave-out-group CV as described above.

        Note that this defines only the base CV strategy, and that the following param `n_repeats`
        is optionally used to replicate this base strategy, e.g., for a twice repeated train-test split evaluation.
        Note further that `n_repeats` will work with any of these options, but say in the case of
        a leave out group CV, it would be awfully redundant, versus, with a passed float value, very reasonable.

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

    train_subjects : str, array-like or Value_Subset, optional
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
        of file in which to read subjects from (assuming one subjects per line).

        `subjects` may also be a custom array-like of subjects to use.

        Lastly, a special wrapper, Value_Subset, can also be used to
        specify more specific, specifically value specific, subsets of
        subjects to use.
        See :class:`Value_Subset` for how this input wrapper can be used.

        ::

            default = 'train'
       
    run_name : str or 'default', optional
        Each run of Evaluate can be optionally associated with a specific `run_name`. This name
        is used to save scores in self.eval_scores, and also if `save_results` in
        :func:`Set_Default_ML_Verbosity<ABCD_ML.Set_Default_ML_Verbosity>` is set to True,
        then will be used as the name output from Evaluate as saved as in the specific log_dr
        (if any, and as set when Init'ing the :class:`ABCD_ML <ABCD_ML.ABCD_ML>` class object),
        with '.eval' appended to the name.

        If left as 'default', will come up with a kind of terrible name passed on the underlying
        model used in the passed `model_pipeline`.

        ::

            default = 'default'

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
        all passed in metrics, computed for each fold within
        each repeat. e.g., array will have a length of `n_repeats` * number of
        folds, and each internal array will have the same length as the number of
        metrics. Optionally, this could instead return a list containing as
        the first element the raw training score in this same format,
        and then the raw testing scores.
        'raw_preds', A pandas dataframe containing the raw predictions
        for each subject, in the test set, and
        'FIs' a list where each element corresponds to a passed feature importance.

    Notes
    ----------
    Prints by default the following for each metric,

    float
        The mean macro score (as set by input metric) across each
        repeated K-fold.

    float
        The standard deviation of the macro score (as set by input metric)
        across each repeated K-fold.

    float
        The standard deviation of the micro score (as set by input metric)
        across each fold with the repeated K-fold.

    '''

    # Perform pre-modeling check
    self._premodel_check()

    # Should save the params used here*** before any preproc done
    run_name =\
        get_avaliable_run_name(run_name, model_pipeline.model, self.eval_scores)
    self.last_run_name = run_name

    # Preproc model pipeline & specs
    problem_spec = self._preproc_problem_spec(problem_spec)
    model_pipeline = self._preproc_model_pipeline(model_pipeline,
                                                  problem_spec.n_jobs)

    # Get the the train subjects to use
    _train_subjects = self._get_subjects_to_use(train_subjects)
    
    # Print the params being used
    if self.default_ML_verbosity['show_init_params']:

        model_pipeline.print_all(self._print)
        problem_spec.print_all(self._print)

        self._print('Evaluate Params')
        self._print('---------------')
        self._print('splits =', splits)
        self._print('n_repeats =', n_repeats)
        self._print('train_subjects =', train_subjects)
        self._print('len(train_subjects) =', len(_train_subjects),
                    '(before overlap w/ problem_spec.subjects)')
        self._print('run_name =', run_name)
        self._print()

    # Init. the Model_Pipeline object with modeling params
    self._init_model(model_pipeline, problem_spec)

    # Get the Eval splits
    _, splits_vals, _ = self._get_split_vals(splits)

    # Evaluate the model
    train_scores, scores, raw_preds, FIs =\
        self.Model_Pipeline.Evaluate(self.all_data, _train_subjects,
                                     splits, n_repeats, splits_vals)

    # Set target and run name
    for fi in FIs:
        fi.set_target(problem_spec.target)
        fi.set_run_name(run_name)

    self._print()

    # Print out summary stats for all passed metrics
    if self.default_ML_verbosity['compute_train_score']:
        score_list = [train_scores, scores]
        score_type_list = ['Training', 'Validation']
    else:
        score_list = [scores]
        score_type_list = ['Validation']

    results = {}
    for scrs, name in zip(score_list, score_type_list):

        summary_scores = self._handle_scores(scrs, name,
                                             problem_spec.weight_metric,
                                             n_repeats, run_name,
                                             self.Model_Pipeline.n_splits_)

        if name == 'Validation':
            results['summary_scores'] = summary_scores
        else:
            results['train_summary_scores'] = summary_scores

    results['raw_scores'] = score_list
    results['raw_preds'] = raw_preds
    results['FIs'] = FIs

    self._save_results(results, run_name + '.eval')
    return results


def Test(self,
         model_pipeline,
         problem_spec,
         train_subjects='train',
         test_subjects='test',
         run_name='default'):
    ''' The test function is one of the main interfaces for testing a specific 
    :class:`Model_Pipeline`. Test is conceptually different from :func:`Evaluate<ABCD_ML.Evaluate>`
    in that it is designed to contrust / train a :class:`Model_Pipeline` on one discrete set of `train_subjects`
    and evaluate it on a further discrete set of `test_subjects`. Otherwise, these functions are very simmilar as
    they both evaluate a :class:`Model_Pipeline` as defined in the context of a :class:`Problem_Spec`, and return
    simmilar output.
    
    Parameters
    ------------
    model_pipeline : :class:`Model_Pipeline`
        The passed `model_pipeline` should be an instance of the ABCD_ML params class :class:`Model_Pipeline`.
        This object defines the underlying model pipeline to be evaluated.

        See :class:`Model_Pipeline` for more information / how to create a the model pipeline.

    problem_spec : :class:`Problem_Spec`
        `problem_spec` accepts an instance of the ABCD_ML params class :class:`Problem_Spec`.
        This object is essentially a wrapper around commonly used parameters needs to define the context
        the model pipeline should be evaluated in. It includes parameters like problem_type, metric, n_jobs, random_state, etc...
        See :class:`Problem_Spec` explicitly for more information and for how to create an instance of this object.

    train_subjects : str, array-like or Value_Subset, optional
        This parameter determines the set of training subjects which are
        used to train the passed instance of :class:`Model_Pipeline`. 
        
        Note, this parameter and `test_subjects` are distinct, but complementary to
        the `subjects` parameter within :class:`Problem_Spec`, which is applied after
        selecting the subset of `train_subjects` specified here.

        By default, this value will be set to the special str indiciator 'train', which
        specifies that the full set of globally defined training subjects
        (See: :func:`Define_Train_Test_Split`), should be used. Other special str indicators
        include 'all' to select all subjects, and 'test' to select the test set subjects.

        If `subjects` is passed a str, and that str is not one of the str indicators listed above,
        then it will be interpretted as the location of file in which to read subjects from (assuming one subjects per line).

        `subjects` may also be a custom array-like of subjects to use.

        Lastly, a special wrapper, Value_Subset, can also be used to specify more specific,
        specifically value specific, subsets of subjects to use.
        See :class:`Value_Subset` for how this input wrapper can be used.

        If passing custom input here, be warned that you NEVER want to pass an overlap of
        subjects between `train_subjects` and `test_subjects`

        ::

            default = 'train'

    test_subjects : str, array-like or Value_Subset, optional
        This parameter determines the set of testing subjects which are
        used to evaluate the passed instance of :class:`Model_Pipeline`, after it
        has been trained on the passed `train_subjects`.
        
        Note, this parameter and `train_subjects` are distinct, but complementary to
        the `subjects` parameter within :class:`Problem_Spec`, which is applied after
        selecting the subset of `test_subjects` specified here.

        By default, this value will be set to the special str indiciator 'test', which
        specifies that the full set of globally defined test subjects
        (See: :func:`Define_Train_Test_Split`), should be used. Other special str indicators
        include 'all' to select all subjects, and 'train' to select the train set subjects.

        If `subjects` is passed a str, and that str is not one of the str indicators listed above,
        then it will be interpretted as the location of file in which to read subjects from (assuming one subjects per line).

        `subjects` may also be a custom array-like of subjects to use.

        Lastly, a special wrapper, Value_Subset, can also be used to specify more specific,
        specifically value specific, subsets of subjects to use.
        See :class:`Value_Subset` for how this input wrapper can be used.

        If passing custom input here, be warned that you NEVER want to pass an overlap of
        subjects between `train_subjects` and `test_subjects`

        ::

            default = 'test'
       
    run_name : str or 'default', optional
        Each run of test can be optionally associated with a specific `run_name`. This name
        is used to save scores in self.test_scores, and also if `save_results` in
        :func:`Set_Default_ML_Verbosity<ABCD_ML.Set_Default_ML_Verbosity>` is set to True,
        then will be used as the name output from Test as saved as in the specific log_dr
        (if any, and as set when Init'ing the :class:`ABCD_ML <ABCD_ML.ABCD_ML>` class object),
        with .test appended to the name.

        If left as 'default', will come up with a kind of terrible name passed on the underlying
        model used in the passed `model_pipeline`.

        ::

            default = 'default'
  
    Returns
    ----------
    results : dict
        Dictionary containing:
        'scores', the score on the test set by each metric,
        'raw_preds', A pandas dataframe containing the raw predictions
        for each subject, in the test set, and 'FIs' a list where
        each element corresponds to a passed feature importance.

    '''

    # Perform pre-modeling check
    self._premodel_check()

    # Get a free run name
    run_name =\
        get_avaliable_run_name(run_name, model_pipeline.model, self.test_scores)
    self.last_run_name = run_name

    # Preproc model pipeline & specs
    problem_spec = self._preproc_problem_spec(problem_spec)
    model_pipeline = self._preproc_model_pipeline(model_pipeline,
                                                  problem_spec.n_jobs)

    # Get the the train subjects + test subjects to use
    _train_subjects = self._get_subjects_to_use(train_subjects)
    _test_subjects = self._get_subjects_to_use(test_subjects)

    # Print the params being used
    if self.default_ML_verbosity['show_init_params']:

        model_pipeline.print_all(self._print)
        problem_spec.print_all(self._print)

        self._print('Test Params')
        self._print('---------------')
        self._print('train_subjects =', train_subjects)
        self._print('len(train_subjects) =', len(_train_subjects),
                    '(before overlap w/ problem_spec.subjects)')
        self._print('test_subjects =', test_subjects)
        self._print('len(test_subjects) =', len(_test_subjects),
                    '(before overlap w/ problem_spec.subjects)')
        self._print('run_name =', run_name)
        self._print()

    # Init the Model_Pipeline object with modeling params
    self._init_model(model_pipeline, problem_spec)

    # Train the model w/ selected parameters and test on test subjects
    train_scores, scores, raw_preds, FIs =\
        self.Model_Pipeline.Test(self.all_data, _train_subjects,
                                 _test_subjects)

    # Set run name
    for fi in FIs:
        fi.set_target(problem_spec.target)
        fi.set_run_name(run_name)

    # Print out score for all passed metrics
    metric_strs = self.Model_Pipeline.metric_strs
    self._print()

    score_list, score_type_list = [], []
    if self.default_ML_verbosity['compute_train_score']:
        score_list.append(train_scores)
        score_type_list.append('Training')

    score_list.append(scores)
    score_type_list.append('Testing')

    for s, name in zip(score_list, score_type_list):

        self._print(name + ' Scores')
        self._print(''.join('_' for i in range(len(name) + 7)))

        for i in range(len(metric_strs)):

            metric_name = metric_strs[i]
            self._print('Metric: ', metric_name)

            scr = s[i]
            if len(scr.shape) > 0:

                targets_key = self.Model_Pipeline.targets_key

                for score_by_class, class_name in zip(scr, targets_key):
                    self._print('for target class: ', class_name)
                    self._print(name + ' Score: ', score_by_class)
                    self._print()
                    self._add_to_scores(run_name, name, metric_name,
                                        'score', score_by_class,
                                        self.test_scores, class_name)

            else:
                self._print(name + ' Score: ', scr)
                self._print()
                self._add_to_scores(run_name, name, metric_name,
                                    'score', scr, self.test_scores)

    results = {}
    results['scores'] = score_list
    results['raw_preds'] = raw_preds
    results['FIs'] = FIs

    self._save_results(results, run_name + '.test')
    return results


def _premodel_check(self):
    '''Internal helper function to ensure that self._prepare_data()
    has been called, and to force a train/test split if not already done.
    Will also call Set_Default_ML_Params if not already called.
    '''

    if self.all_data is None:
        self._prepare_data()

    if self.train_subjects is None:

        raise RuntimeError('No train-test set defined!',
                           'If this is intentional, call Train_Test_Split',
                           'with test_size = 0')

    if self.default_ML_verbosity == {}:

        self._print('Setting default ML verbosity settings!')
        self._print('Note, if the following values are not desired,',
                    'call self.Set_Default_ML_Verbosity()')

        self.Set_Default_ML_Verbosity()


def _preproc_model_pipeline(self, model_pipeline, n_jobs):

    # Set values across each pipeline pieces params
    model_pipeline.preproc(n_jobs)
    
    # Proc sample_on if needed (by adding strat name)
    model_pipeline.check_samplers(self._add_strat_u_name)

    # Set split vals if search
    if model_pipeline.param_search is not None:

        _, split_vals, _ =\
            self._get_split_vals(model_pipeline.param_search.splits)
        model_pipeline.param_search.set_split_vals(split_vals)

    # Early check to see if imputer could even be needed
    model_pipeline.check_imputer(self.all_data)

    return model_pipeline


def _preproc_problem_spec(self, problem_spec):

    # Update target with actual target key
    target_key = self._get_targets_key(problem_spec.target)
    problem_spec.set_params(target=target_key)

    # Proc subjects to use
    final_subjects = self._get_subjects_to_use(problem_spec.subjects)
    problem_spec.set_final_subjects(final_subjects)

    # Set by class defaults
    if problem_spec.n_jobs == 'default':
        problem_spec.n_jobs = self.n_jobs

    if problem_spec.random_state == 'default':
        problem_spec.random_state = self.random_state

    # If any input has changed, manually (i.e., not by problem_spec init)
    problem_spec._proc_checks()
    
    return problem_spec


def _get_split_vals(self, splits):

    if isinstance(splits, int):
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

        self.last_subjects_names = []
        for strat_name, value in zip(split_names, rev_values):
            if self.strat_u_name in strat_name:
                strat_name = strat_name.replace(self.strat_u_name, '')

            self.last_subjects_names.append((strat_name, value))

        self._print('subjects set to: ',
                    self.last_subjects_names)
        self._print()

    # Lastly, if not the above, assume it is an array-like of subjects
    else:
        subjects = self._load_set_of_subjects(subjects=subjects_to_use)

    return subjects


def _init_model(self, model_pipeline, problem_specs):

    # Set Model_Pipeline
    self.Model_Pipeline =\
        Model_Pipeline(model_pipeline, problem_specs, self.CV, self.Data_Scopes,
                       self.default_ML_verbosity['progress_bar'],
                       self.default_ML_verbosity['compute_train_score'],
                       self._ML_print)

def _handle_scores(self, scores, name, weight_metric, n_repeats, run_name, n_splits):

    all_summary_scores = []
    metric_strs = self.Model_Pipeline.metric_strs

    self._print(name + ' Scores')
    self._print(''.join('_' for i in range(len(name) + 7)))

    weight_metrics = conv_to_list(weight_metric, len(metric_strs))

    for i in range(len(metric_strs)):

        # Weight outputed scores if requested
        if weight_metrics[i]:
            weights = self.Model_Pipeline.n_test_per_fold
        else:
            weights = None

        metric_name = metric_strs[i]
        self._print('Metric: ', metric_name)
        score_by_metric = scores[:, i]

        if len(score_by_metric[0].shape) > 0:
            by_class = [[score_by_metric[i][j] for i in
                        range(len(score_by_metric))] for j in
                        range(len(score_by_metric[0]))]

            summary_scores_by_class =\
                [compute_macro_micro(class_scores, n_repeats,
                 n_splits, weights=weights) for class_scores in by_class]

            targets_key = self.Model_Pipeline.targets_key
            classes = self.Model_Pipeline.classes

            class_names =\
                self.targets_encoders[targets_key].inverse_transform(
                    classes.astype(int))

            for summary_scores, class_name in zip(summary_scores_by_class,
                                                  class_names):

                self._print('Target class: ', class_name)
                self._print_summary_score(name, summary_scores,
                                          n_repeats, run_name,
                                          metric_name, class_name, weights=weights)

            all_summary_scores.append(summary_scores_by_class)

        else:

            # Compute macro / micro summary of scores
            summary_scores = compute_macro_micro(score_by_metric,
                                                 n_repeats,
                                                 n_splits,
                                                 weights=weights)

            self._print_summary_score(name, summary_scores,
                                      n_repeats, run_name,
                                      metric_name, weights=weights)

            all_summary_scores.append(summary_scores)

    return all_summary_scores


def _print_summary_score(self, name, summary_scores, n_repeats, run_name,
                         metric_name, class_name=None, weights=None):
    '''Besides printing, also adds scores to self.eval_scores dict
    under run name.'''

    mn = 'Mean'
    if weights is not None:
        mn = 'Weighted ' + mn

    self._print(mn + ' ' + name + ' score: ', summary_scores[0])
    self._add_to_scores(run_name, name, metric_name, mn,
                        summary_scores[0], self.eval_scores,  class_name)

    if n_repeats > 1:
        self._print('Macro Std in ' + name + ' score: ',
                    summary_scores[1])
        self._print('Micro Std in ' + name + ' score: ',
                    summary_scores[2])
        self._add_to_scores(run_name, name, metric_name, 'Macro Std',
                            summary_scores[1], self.eval_scores, class_name)
        self._add_to_scores(run_name, name, metric_name, 'Micro Std',
                            summary_scores[2], self.eval_scores, class_name)
    else:
        self._print('Std in ' + name + ' score: ',
                    summary_scores[2])
        self._add_to_scores(run_name, name, metric_name, 'Std',
                            summary_scores[2], self.eval_scores, class_name)

    self._print()


def _add_to_scores(self, run_name, name, metric_name, val_type, val, scores,
                   class_name=None):

    if run_name not in scores:
        scores[run_name] = {}

    if name not in scores[run_name]:
        scores[run_name][name] = {}

    if metric_name not in scores[run_name][name]:
        scores[run_name][name][metric_name] = {}

    if class_name is None:
        scores[run_name][name][metric_name][val_type] = val

    else:
        if class_name not in scores[run_name][name][metric_name]:
            scores[run_name][name][metric_name][class_name] = {}

        scores[run_name][name][metric_name][class_name][val_type] =\
            val


def _save_results(self, results, save_name):

    if self.default_ML_verbosity['save_results'] and self.log_dr is not None:

        save_dr = os.path.join(self.exp_log_dr, 'results')
        os.makedirs(save_dr, exist_ok=True)

        save_spot = os.path.join(save_dr, save_name)
        with open(save_spot, 'wb') as f:
            pkl.dump(results, f)
