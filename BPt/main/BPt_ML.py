"""
main.py
====================================
The main project class.
"""
import pandas as pd
import shutil
import os
import pickle as pkl

from ..helpers.Docstring_Helpers import get_new_docstring
# from ..helpers.Params_Classes import ML_Params
from ..helpers.CV import CV


def Load(loc, exp_name='default', log_dr='default', existing_log='default',
         verbose='default', notebook='default', random_state='default'):
    '''
    This function is designed to load in a saved previously created
    BPt_ML object.

    See :func:`Save <BPt.BPt_ML.Save>` for saving an object.
    See :func:`Init <BPt.BPt_ML>` for the
    rest of changable param descriptions, e.g., log_dr, existing_log, ect...

    Parameters
    ----------
    loc : str or Path

        A path/str to a saved BPt_ML object,
        (One saved with :func:`Save <BPt.BPt_ML.Save>`), then that object will be
        loaded. Notably, if any additional params are passed along
        with it, e.g., exp_name, notebook, ect... they will override
        the saved values with the newly passed values.
        If left as 'default', all params will be set to the loaded value,
        though see the warning below.

        .. WARNING::
            The exp_name or log_dr may need to be changed, especially
            in the case where the object is being loaded in a new
            location or enviroment from where the original was created,
            as it will by default try to create logs with the saved path
            information as the original.

        You can only change exp_name, log_dr, existing_log, verbose,
        notebook and random_state when loading a new object, for the
        remaining params, even if a value is passed, it will not be
        applied. If the user really wishes to change one of these params,
        they can change it manually via self.name_of_param = whatever.
    '''

    with open(loc, 'rb') as f:
        ML = pkl.load(f)

    if exp_name != 'default':
        ML.exp_name = exp_name
    if log_dr != 'default':
        ML.log_dr = log_dr
    if existing_log != 'default':
        ML.existing_log = existing_log
    if verbose != 'default':
        ML.verbose = verbose

    ML._init_logs()

    if notebook != 'default':
        ML.notebook = notebook
    if random_state != 'default':
        ML.random_state = random_state

    ML._print('ML object loaded from save!')
    return ML


class BPt_ML():

    def __init__(self, exp_name='My_Exp', log_dr='', existing_log='append',
                 verbose=True, notebook=True,
                 use_abcd_subject_ids=False,
                 low_memory_mode=False, strat_u_name='_Strat',
                 random_state=534, n_jobs=1, dpi=100, mp_context='loky'):
        '''Main class used within BPt for interfacing with Data Loading
        and Modeling / Other funcationality.

        Parameters
        ----------
        exp_name : str, optional
            The name of this experimental run,
            used explicitly in saving logs, and figures, where the passed
            `exp_name` is used as the name of the log folder.
            If log_dr is not set to None,
            (if not None then saves logs and figures)
            then a folder is created within the log dr
            with the exp_name.

            ::

                default = 'My_Exp'

        log_dr : str, Path or None, optional
            The directory in which to store logs...
            If set to None, then will not save any logs!
            If set to empty str, will save in the current dr.

            ::

                default = ''

        existing_log : {'new', 'append', 'overwrite'}, optional
            This parameter dictates different choices for when
            an a folder with exp_name already exists in the specified
            log_dr.

            These choices are:

            - 'new'
                If the log folder already exists, then
                just increment `exp_name` until a free name is found,
                and use that as the log folder / `exp_name`.

            - 'append'
                If existing_log is 'append' then log entries
                and new figures will be added to the existing folder.

            - 'overwrite'
                If existing_log is 'overwrite', then the existing
                log folder with the same exp_name will be cleared
                upon __init__.

            ::

                default = 'append'

        verbose: bool, optional
            If `verbose` is set to True, the BPt_ML object
            will print output, diagnostic and more general, directly
            to std out. If set to False, no output will be printed, though
            output will still be recorded within the logs assuming
            log_dr is not None.

            ::

                default = True

        notebook : bool, optional
            If True, then assumes the user is running
            the code in an interactive jupyter notebook. 
            In this case, certain features will either be enabled or disabled,
            e.g., type of progress bar.

            ::

                default = Trues

        use_abcd_subject_ids : bool, optional
            Flag to determine the usage of ABCD speficic 'default'
            subject id behavior.
            If set to True, this will convert input NDAR subject ids
            into upper case, with prepended NDAR - type format.
            If set to False, then all input subject names must be entered
            explicitly the same, no preprocessing will be done on them.

            ::

                default = False

        low_memory_mode : bool, optional
            This parameter dictates behavior around loading in data,
            specifically,
            If set to True, individual dataframes self.data, self.covars ect...
            will be deleted from memory as soon as modeling begins.
            This parameter also controls the pandas read_csv behavior,
            which also has a low_memory flag.

            ::

                default = False

        strat_u_name : str, optional
            A unique str identifier to be appended to every loaded
            strat value (to keep them seperate from covars and data).

            You should only need to change or ever worry about this in
            the case that one of your input variables happens to have the
            default value of '_Strat' in it...

            ::

                default = '_Strat'

        random_state : int, RandomState instance or None, optional
            The default random state, either as int for a specific seed,
            or if None then the random seed is set by np.random.
            This parameters if set will be the default random_state class-wide,
            so any place random_state is left to default, unless a different
            default is set (e.g. default load value or default ML value) this
            random state will be used.

            ::

                default = 534

        n_jobs : int, optional
            The default number of jobs / processors to use (if avaliable) where
            ever avaliable class-wide across the BPt.

            ::

                default = 1

        dpi : int, optional
            The default dpi in which to save any automatically saved fiugres
            with.
            Where this parameter can also be set to specific values
            for specific plots.

            ::

                default = 1

        mp_context : str, optional

            When a hyper-parameter search is launched, there are different
            ways through python that the multi-processing can be launched
            (assuming n_jobs > 1). Occassionally some choices can lead to
            unexpected errors.

            Choices are:

            - 'loky': Create and use the python library
                loky backend.

            - 'fork': Python default fork mp_context

            - 'forkserver': Python default forkserver mp_context

            - 'spawn': Python default spawn mp_context

            ::

                default = 'loky'
        '''
        # Load logging class params
        self.exp_name = exp_name
        self.log_dr = log_dr
        self.existing_log = existing_log
        self.verbose = verbose

        self._init_logs()

        self._print('exp_name =', self.exp_name)
        self._print('log_dr =', self.log_dr)
        self._print('existing_log =', self.existing_log)
        self._print('verbose =', self.verbose)
        self._print('exp log dr setup at:', self.exp_log_dr)
        self._print('log file at:', self.log_file)

        # Set rest of class params
        self.notebook = notebook
        self.use_abcd_subject_ids = use_abcd_subject_ids
        self.low_memory_mode = low_memory_mode
        self.strat_u_name = strat_u_name
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.dpi = dpi
        self.mp_context = mp_context

        self._print('Default params set:')
        self._print('notebook =', self.notebook)
        self._print('use_abcd_subject_ids =', self.use_abcd_subject_ids)
        self._print('low memory mode =', self.low_memory_mode)
        self._print('strat_u_name =', self.strat_u_name)
        self._print('random state =', self.random_state)
        self._print('n_jobs =', self.n_jobs)
        self._print('dpi =', self.dpi)
        self._print('mp_context =', self.mp_context)

        # Initialze various variables
        self.name_map, self.exclusions, self.inclusions = {}, set(), set()
        self.data, self.covars = pd.DataFrame(), pd.DataFrame()
        self.targets, self.strat = pd.DataFrame(), pd.DataFrame()

        # Dict objects to hold encoders
        self.covars_encoders = {}
        self.targets_encoders = {}
        self.strat_encoders = {}

        # Class values to be set later
        self.all_data = None
        self.targets_keys = []

        # Stores the gloabl train/test split
        self.train_subjects, self.test_subjects = None, None

        # CV by default is just random splits
        self.cv = CV()

        # Store default dicts as init empty
        self.default_load_params, self.default_ML_verbosity = {}, {}

        self.subject_id = 'src_subject_id'

        self.file_mapping = {}
        self.data_file_keys = []

        self._print('BPt_ML object initialized')

    def Save(self, loc, low_memory=False):
        '''This class method is used to save an existing BPt_ML
        object for further use.

        Parameters
        ----------
        loc : str or Path
            The location in which the pickle of the BPt_ML object
            should be saved! This is the same loc which should be
            passed to :func:`Load <BPt.main.BPt_ML.Load>` in order to
            re-load the object.

        low_memory : bool, optional
            If this parameter is set to True, then self.data,
            self.targets, self.covars, self.strat will be deleted
            before saving. The assumption for the param to be used is
            that self.all_data has already been created, and therefore
            the individual dataframes with data, covars ect... can safely
            be deleted as the user will not need to work with them directly
            any more.

            ::

                default = False
        '''

        if low_memory:
            self.data, self.covars = pd.DataFrame(), pd.DataFrame()
            self.targets, self.strat = pd.DataFrame(), pd.DataFrame()

            try:
                del self.evaluator
            except AttributeError:
                pass

        with open(loc, 'wb') as f:
            pkl.dump(self, f)

    def _init_logs(self):

        if self.log_dr is not None:

            if self.log_dr == '':
                self.log_dr = os.getcwd()

            # Ensure log_dr exists, if not make it
            os.makedirs(self.log_dr, exist_ok=True)

            # Get exp_log_dr name
            self.exp_log_dr = os.path.join(self.log_dr, self.exp_name)

            if os.path.isdir(self.exp_log_dr):

                if self.existing_log == 'new':

                    cnt = 1
                    while os.path.isdir(self.exp_log_dr +
                                        '(' + str(cnt) + ')'):
                        cnt += 1

                    self.exp_log_dr += '(' + str(cnt) + ')'

                # If overwrite, delete everything, then make new blank
                elif self.existing_log == 'overwrite':
                    shutil.rmtree(self.exp_log_dr)

                # Make the new dr
                if self.existing_log != 'append':
                    os.mkdir(self.exp_log_dr)

            # If the dr doesn't already exist, regardless of existing log
            # Just make new dr.
            else:
                os.mkdir(self.exp_log_dr)

            # Make the log file if not already made.
            self.log_file = os.path.join(self.exp_log_dr, 'logs.txt')

        else:
            self.exp_log_dr = None
            self.log_file = None

    def _print(self, *args, **kwargs):
        '''Overriding the print function to allow for
        customizable verbosity within class methods. Will also
        take care of logging behavior.

        Parameters
        ----------
        args
            Anything that would be passed to default python print
        '''

        dont_print = kwargs.pop('dont_print', False)

        if self.verbose and not dont_print:
            print(*args, **kwargs)

        if self.log_file is not None:
            log = open(self.log_file, 'a')
            print(*args, **kwargs, file=log)
            log.close()

    def _print_nothing(self, *args, **kwargs):
        pass

    # Data loader functionality
    from ._Data import (Set_Default_Load_Params,
                        _make_load_params,
                        _get_data_file_cnt,
                        Load_Name_Map,
                        Load_Data,
                        Load_Data_Files,
                        Load_Targets,
                        _proc_target,
                        _print_loaded_targets,
                        Load_Covars,
                        _proc_covar,
                        Load_Strat,
                        _proc_strat,
                        Load_Exclusions,
                        Load_Inclusions,
                        Drop_Data_Cols,
                        _drop_data_cols,
                        Filter_Data_Cols,
                        Filter_Data_Files_Cols,
                        Consolidate_Data_Files,
                        Proc_Data_Unique_Cols,
                        _proc_data_unique_cols,
                        Drop_Data_Duplicates,
                        Binarize_Target,
                        _proc_threshold,
                        Binarize_Covar,
                        Get_Overlapping_Subjects,
                        Clear_Name_Map,
                        Clear_Data,
                        Clear_Covars,
                        Clear_Targets,
                        Clear_Strat,
                        Clear_Exclusions,
                        Clear_Inclusions,
                        Get_Nan_Subjects,
                        _get_targets_key,
                        _load_datasets,
                        _load_user_passed,
                        _load_dataset,
                        _common_load,
                        _load,
                        _set_overlap,
                        _merge_existing,
                        _proc_df,
                        _load_set_of_subjects,
                        _process_subject_name,
                        _drop_na,
                        _filter_by_eventname,
                        _show_na_info,
                        _drop_excluded,
                        _drop_included,
                        _filter_excluded,
                        _filter_included,
                        _get_overlapping_subjects,
                        Prepare_All_Data,
                        _get_cat_keys,
                        _set_data_scopes,
                        _get_base_targets_names)

    # Update loader docstrings
    Load_Name_Map.__doc__ =\
        get_new_docstring(Set_Default_Load_Params, Load_Name_Map)
    Load_Data.__doc__ =\
        get_new_docstring(Set_Default_Load_Params, Load_Data)
    Load_Data_Files.__doc__ =\
        get_new_docstring(Load_Data, Load_Data_Files)
    Load_Targets.__doc__ =\
        get_new_docstring(Set_Default_Load_Params, Load_Targets)
    Load_Covars.__doc__ =\
        get_new_docstring(Set_Default_Load_Params, Load_Covars)
    Load_Strat.__doc__ =\
        get_new_docstring(Set_Default_Load_Params, Load_Strat)
    Filter_Data_Cols.__doc__ =\
        get_new_docstring(Set_Default_Load_Params, Filter_Data_Cols)
    Proc_Data_Unique_Cols.__doc__ =\
        get_new_docstring(Set_Default_Load_Params, Proc_Data_Unique_Cols)
    Drop_Data_Duplicates.__doc__ =\
        get_new_docstring(Set_Default_Load_Params, Drop_Data_Duplicates)

    # Validation / CV funcationality
    from ._Validation import (_get_cv,
                              Define_Validation_Strategy,
                              Train_Test_Split,
                              _add_strat_u_name,
                              _get_info_on)

    # Machine Learning functionality
    from ._ML import (Set_Default_ML_Verbosity,
                      _ML_print,
                      Evaluate,
                      Test,
                      _premodel_check,
                      _preproc_param_search,
                      _preproc_model_pipeline,
                      _preproc_cv_splits,
                      _preproc_problem_spec,
                      _get_split_vals,
                      _get_subjects_to_use,
                      _init_evaluator,
                      _handle_scores,
                      _print_summary_score,
                      _save_results,
                      get_pipeline)

    from ._Plotting import (_plot,
                            _proc_subjects,
                            _get_plot_df,
                            Show_Data_Dist,
                            _input_targets,
                            _input_covars,
                            _input_strat,
                            Show_Targets_Dist,
                            Show_Covars_Dist,
                            Show_Strat_Dist,
                            _get_single_df,
                            _show_single_dist,
                            _get_cat_display_df,
                            _show_dist,
                            _display_df,
                            _get_top_global,
                            Plot_Global_Feat_Importances,
                            _plot_multiclass_global_feat_importances,
                            _plot_global_feat_importances,
                            Plot_Local_Feat_Importances,
                            _plot_shap_summary)

    from ._Tables import (Save_Table,
                          _get_single_dfs,
                          _get_table_contents,
                          _get_group_titles)
