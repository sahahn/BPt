"""
ABCD_ML.py
====================================
The main project class.
"""
import pandas as pd
import shutil
import shap
import os
import pickle as pkl

from ..helpers.Docstring_Helpers import get_new_docstring
from ..helpers.CV import CV


def Load(loc, exp_name='default', log_dr='default', existing_log='default',
         verbose='default', notebook='default', random_state='default'):
    '''
    This function is designed to load in a saved previously created
    ABCD_ML object.

    See :func:`Save <ABCD_ML.ABCD_ML.Save>` for saving an object.
    See :func:`Init <ABCD_ML.ABCD_ML>` for the
    rest of changable param descriptions, e.g., log_dr, existing_log, ect...

    Parameters
    ----------
    loc : str or Path

        A path/str to a saved ABCD_ML object,
        (One saved with :func:`Save <ABCD_ML.ABCD_ML.Save>`), then that object will be
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

    ML._print('ABCD_ML object loaded from save!')
    return ML


class ABCD_ML():
    '''The main class used in ABCD_ML project'''

    def __init__(self, exp_name='Exp', log_dr='', existing_log='append',
                 verbose=True, notebook=True,
                 use_default_subject_ids=False,
                 low_memory_mode=False, strat_u_name='_Strat', dpi=100,
                 random_state=None):
        '''Main class init

        Parameters
        ----------
        exp_name : str, optional
            The name of this experimental run,
            used explicitly in saving logs, and figures.
            If log_dr is not set to None-
            (if not None then saves logs and figures)
            then a folder is created within the log dr
            with the exp_name.

            (default = 'Exp')

        log_dr : str, Path or None, optional
            The directory in which to store logs...
            If set to None, then will not save any logs!
            If set to '', will save in the current dr.

            (default = '')

        existing_log : {'new', 'append', 'overwrite'}, optional
            By default, if an exp_name folder already
            exists within the log_dr, then the exp_name will
            be incremented until a free name is avaliable.
            This behavior is existing_log is 'new',
            If existing_log is 'append' then log entries
            and new figures will be added to the existing folder.
            If existing_log is 'overwrite', then the existing
            log folder with the same exp_name will be cleared
            upon __init__.

            (default = 'append')

        verbose: bool, optional
            If set to true will print diagnostic and other output during
            dataloading and model training ect... if set to False this output
            will not print. If log_dr is not None, then will still
            record as log output.

            (default = True)

        notebook : bool, optional
            If True, then assumes the user is running
            the code in an interactive notebook. In this case,
            any plots will be showed interactively
            (as well as saved as long as log_dr != None)

            (default = True)

        use_default_subject_ids : bool, optional
            Flag to determine the usage of 'default' subject id behavior.
            If set to True, this will convert input NDAR subject ids
            into upper case, with prepended NDAR - type format.
            If set to False, then all input subject names must be entered
            explicitly the same, no preprocessing will be done on them.

            (default = False)

        low_memory_mode : bool, optional
            This parameter dictates behavior around loading in data,
            specifically, if `low_memory_mode` is set to True,
            then when loading data from multiple sources, only common
            subjects will be saved as each data source is loaded.
            For comparison, when low memory mode if off, the dropping
            of non-common subjects occurs later. Though regardless of if low
            memory mode is on or off, subjects will be dropped right away
            when exclusions or strat is loaded. Non-low memory mode
            behavior is useful when the user wants to try loading different
            data, and doesn't want automatic drops to occur.
            If set to True, individual dataframes self.data, self.covars ect...
            will also be deleted from memory as soon as modeling begins.
            This parameter also controls the pandas read_csv behavior,
            which also has a low_memory flag.

            (default = False)

        strat_u_name : str, optional
            A unique str identifier to be appended to every loaded
            strat value (to keep it seperate from covars and data).

            (default = '_Strat')

        dpi : int, optional
            The default dpi in which to save any distribution or feature
            importance plots.

            (default = 100)

        random_state : int, RandomState instance or None, optional
            The default random state, either as int for a specific seed,
            or if None then the random seed is set by np.random.
            This parameters if set will be the default random_state class-wide,
            so any place random_state is left to default, unless a different
            default is set (e.g. default load value or default ML value) this
            random state will be used.

            (default = None)
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
        self.use_default_subject_ids = use_default_subject_ids
        self.low_memory_mode = low_memory_mode
        self.strat_u_name = strat_u_name
        self.dpi = dpi
        self.random_state = random_state

        self._print('notebook =', self.notebook)
        self._print('use_default_subject_ids =', self.use_default_subject_ids)
        self._print('low memory mode =', self.low_memory_mode)
        self._print('strat_u_name =', self.strat_u_name)
        self._print('random state =', self.random_state)

        # Initialze various variables
        self.name_map, self.exclusions, self.inclusions = {}, set(), set()
        self.data, self.covars = pd.DataFrame(), pd.DataFrame()
        self.targets, self.strat = pd.DataFrame(), pd.DataFrame()

        # Dict objects to hold encoders
        self.covars_encoders = {}
        self.targets_encoders = {}
        self.strat_encoders = {}

        # Class values to be set later
        self.all_data, self.all_data_keys = None, {}
        self.targets_keys = []

        # Stores the gloabl train/test split
        self.train_subjects, self.test_subjects = None, None

        # CV by default is just random splits
        self.CV = CV()

        # Store default dicts as init empty
        self.default_load_params, self.default_ML_params, \
            self.default_ML_verbosity = {}, {}, {}

        # Scores and settings are saved after each eval or test run
        self.eval_scores, self.eval_settings = {}, {}
        self.test_scores, self.test_settings = {}, {}

        self.subject_id = 'src_subject_id'

        self.last_run_name = None
        self.last_subjects_to_use_names = None

        self.file_mapping = {}
        self.data_file_keys = []

        self._print('ABCD_ML object initialized')

    def Save(self, loc, low_memory=False):
        '''This class method is used to save an existing ABCD_ML
        object for further use.

        Parameters
        ----------
        loc : str or Path
            The location in which the pickle of the ABCD_ML object
            should be saved! This is the same loc which should be
            passed to :func:`Load <ABCD_ML.main.ABCD_ML.Load>` in order to
            re-load the object.

        low_memory : bool, optional
            If this parameter is set to True, then self.data,
            self.targets, self.covars, self.strat will be deleted
            before saving. The assumption for the param to be used is
            that self.all_data has already been created, and therefore
            the individual dataframes with data, covars ect... can safely
            be deleted as the user will not need to work with them directly
            any more.

            In addition, self.Model_Pipeline (which contains
            information about the last run Evaluate or Test call) will be
            deleted.
        '''

        if low_memory:
            self.data, self.covars = pd.DataFrame(), pd.DataFrame()
            self.targets, self.strat = pd.DataFrame(), pd.DataFrame()

            try:
                del self.Model_Pipeline
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
                        _filter_data_cols,
                        Proc_Data_Unique_Cols,
                        _proc_data_unique_cols,
                        Drop_Data_Duplicates,
                        Binarize_Target,
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
                        _drop_from_filter,
                        _filter_by_eventname,
                        _show_na_info,
                        _drop_excluded,
                        _drop_included,
                        _filter_excluded,
                        _filter_included,
                        _get_overlapping_subjects,
                        _process_new,
                        _prepare_data,
                        _get_cat_keys,
                        _set_all_data_keys,
                        _get_base_targets_names,
                        _get_covar_scopes)

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
    from ._Validation import (Define_Validation_Strategy,
                              Train_Test_Split,
                              _add_strat_u_name,
                              _get_info_on)

    # Machine Learning functionality
    from ._ML import (Set_Default_ML_Params,
                      Set_Default_ML_Verbosity,
                      _ML_print,
                      Evaluate,
                      Test,
                      _premodel_check,
                      _make_ML_params,
                      _print_model_params,
                      _get_split_vals,
                      _get_final_subjects_to_use,
                      _init_model,
                      _get_avaliable_run_name,
                      _handle_scores,
                      _print_summary_score,
                      _add_to_scores,
                      _save_results)

    # Fill Evaluate and Test's docstring
    Evaluate.__doc__ = get_new_docstring(Set_Default_ML_Params, Evaluate)
    Test.__doc__ = get_new_docstring(Evaluate, Test)

    from ..pipeline.Models import Show_Models
    from ..pipeline.Metrics import Show_Metrics
    from ..pipeline.Loaders import Show_Loaders
    from ..pipeline.Transformers import Show_Transformers
    from ..pipeline.Scalers import Show_Scalers
    from ..pipeline.Imputers import Show_Imputers
    from ..pipeline.Samplers import Show_Samplers
    from ..pipeline.Feature_Selectors import Show_Feat_Selectors
    from ..pipeline.Ensembles import Show_Ensembles

    from ._Plotting import (_plot,
                            _proc_subjects,
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
