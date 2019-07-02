import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.stats.mstats import winsorize
from ABCD_ML.Data_Helpers import (process_binary_input, process_categorical_input,
                                  filter_float_by_outlier, get_unique_combo)
from ABCD_ML.CV import CV

class ABCD_ML():

    def __init__(self,
                 eventname = 'baseline_year_1_arm_1',
                 use_default_subject_ids = True,
                 default_na_values = ['777', '999'],
                 verbose = True
                 ):

        self.eventname = eventname
        self.use_default_subject_ids = use_default_subject_ids
        self.default_na_values = default_na_values
        self.verbose = verbose

        self.data, self.covars = [], []
        self.scores, self.strat = [], []
        self.name_map, self.exclusions = {}, set()
        self.covar_encoders, self.score_encoder, self.strat_encoders = {}, None, {}
        self.score_key = 'score'

        self.n_jobs = 8

        self.all_data, self.train_subjects, self.test_subjects = None, None, None
        self.CV = CV()

    def print(self, *args):
        '''Overriding the print function to allow from verbosity'''

        if self.verbose:
            print(*args)
    
    def merge_existing(self, class_data, local_data):
        '''Internal helper function to handle either merging dataframes after load,
           or if not loaded then setting class value'''

        #If other covars data is already loaded, merge with it
        if len(class_data) > 0:
            class_data = pd.merge(class_data, local_data, on='src_subject_id')
            self.print('Merged with existing!')
            return class_data
        else:
            return local_data

    def load_data(self,
                  loc,
                  drop_keys = [],
                  filter_outlier_percent = None,
                  winsorize_val = None
                  ):
        ''' 
        Load a 2.0_ABCD_Data_Explorer release formatted neuroimaging dataset of ROI's

        data_loc -- The location of the dataset csv file
        eventname -- The eventname to select datapoints by, if None then ignore
        drop_keys -- A list of keys to drop columns by, where if any key given in a columns name,
            then that column will be dropped (Note: if a name mapping exists,
            this drop step will be conducted after renaming)
        filter_outlier_percent -- For float/ordinal only:
            A percent of values to exclude from either end of the score distribution,
            provided as either 1 number, or a tuple (% from lower, % from higher). None, to perform no filtering.
        winsorize_val -- The (limits[0])th lowest values are set to the (limits[0])th percentile,
            and the (limits[1])th highest values are set to the (1 - limits[1])th
            percentile. If one value passed, used for both ends.
        '''
        
        self.print('Loading', loc) 
        data = pd.read_csv(loc, na_values=self.default_na_values)

        #Drop the first two columns by default (typically specific id, then dataset id for NDA csvs)
        first_cols = list(data)[:2]
        data = data.drop(first_cols, axis=1)
        self.print('dropped', first_cols, 'columns by default')

        #Perform common operations (check subject id, drop duplicate subjects ect...)
        data = self.proc_df(data)

        #Drop any columns if any of the drop keys occur in the column name
        column_names = list(data)
        to_drop = [name for name in column_names for drop_key in drop_keys if drop_key in name]
        data = data.drop(to_drop, axis=1)
        self.print('Dropped', len(to_drop), 'columns, per drop_keys argument')

        #Drop any rows with missing data
        data = self.drop_na(data)
        self.print('Dropped rows with missing data')

        data_keys = list(data)

        if filter_outlier_percent != None:
            for key in data_keys:
                data = filter_float_by_outlier(data, key, filter_outlier_percent, in_place=False, verbose=False)

            data = data.dropna() #To actually remove the outliers
            self.print('Filtered data for outliers with value: ', filter_outlier_percent)

        if winsorize_val != None:
            
            if type(winsorize_val) != tuple:
                winsorize_val = (winsorize_val)

            data[data_keys] = winsorize(data[data_keys], winsorize_val, axis=0)
            self.print('Winsorized data with value: ', winsorize_val)

        self.print('loaded shape: ', data.shape)

        #If other data is already loaded, merge this data with existing loaded data
        self.data = self.merge_existing(self.data, data)
        self.process_new()
    
    def load_custom_data(self):
        pass

    def common_load(self, loc, col_name=None, col_names=None):
        '''
        Internal helpfer function to perform set of loading functions used by multiple loaders.
        2.0_ABCD_Data_Explorer release formatted.

        If col_name passed, just returns df
        If col_names passed, returns df, col_names
        '''

        #Read csv data from loc
        data = pd.read_csv(loc, na_values=self.default_na_values)

        #Perform common df corrections
        data = self.proc_df(data)

        if col_name != None:

            data = data[[col_name]].dropna()
            return data

        if type(col_names) != list:
            col_names = list(col_names)

        data = data[col_names].dropna()
        return data, col_names

    def load_covars(self,
                    loc,
                    col_names,
                    data_types,
                    dummy_code_categorical = True,
                    filter_float_outlier_percent = None,
                    standardize = True,
                    normalize = True,
                    ):

        drop = None
        if dummy_code_categorical:
            drop = 'first'

        self.print('Reading covariates from', loc)
        covars, col_names = self.common_load(loc, col_names=col_names)

        if type(data_types) != list:
            data_types = list(data_types)

        assert len(data_types) == len(col_names), "You must provide the same # of datatypes as column names!"

        for key, d_type in zip(col_names, data_types):

            self.print('load:', key)

            if d_type == 'binary' or d_type == 'b':
                covars, encoder = process_binary_input(covars, key, self.verbose)
                self.covar_encoders[key] = encoder

            elif d_type == "categorical" or d_type == 'c':
                covars, new_keys, encoder = process_categorical_input(covars, key,
                            drop=drop, verbose=self.verbose)
                self.covar_encoders[key] = encoder

            elif (d_type == 'float' or d_type == 'ordinal' or
                  d_type == 'f' or d_type == 'o'):

                if filter_float_outlier_percent != None and (d_type == 'float' or d_type == 'f'):
                    covars = filter_float_by_outlier(covars, key, filter_outlier_percent,
                                                     in_place=False, verbose=self.verbose)

                if standardize:
                    covars[key] -= np.mean(covars[key])
                    covars[key] /= np.std(covars[key])

                if normalize:
                    min_val, max_val = np.min(covars[key]), np.max(covars[key])
                    covars[key] = (covars[key] - min_val) / (max_val - min_val)

        #Filter float by outlier just replaces with nan, so actually remove here
        covars = covars.dropna()

        #If other data is already loaded, merge this data with existing loaded data
        self.covars = self.merge_existing(self.covars, covars)
        self.process_new() 

    def load_scores(self,
                    loc,
                    col_name,
                    data_type = 'float',
                    filter_outlier_percent = None
                    ):

        '''
        Loads in a set of subject ids and associated scores from a 2.0_ABCD_Data_Explorer release formatted csv.
        Scores can be either 'binary', 'categorical', 'ordinal' or 'float', where ordinal and float are treated the same.
        
        For binary: scores are read in and label encoded to be 0 and 1, 
        (Will work if passed column of unique string also, e.g. 'M' and 'F')
       
        For categorical: scores are read in and by default one-hot encoded,
        Note: This function is designed only to work with categorical scores read in from one column!
        *Reading multiple scores from multiple places is not supported as of now.

        For ordinal + float: scores are read in as a floating point number,
        and optionally then filtered for outliers with the filter_outlier_percent flag.
        
        scores_loc -- The location of the scores csv file
        score_col_name -- The name of the column with the score of interest
            Note: if a name mapping exists, the score col name will refer to the changes name
        score_data_type -- The datatype of the score, 'binary', 'categorical', 'ordinal' or 'float',
            explained in more detail above. Can also pass in 'b', 'c', 'o' or 'f'
        filter_outlier_percent -- For float/ordinal only:
            A percent of values to exclude from either end of the score distribution,
            provided as either 1 number, or a tuple (% from lower, % from higher). None, to perform no filtering.
        
        '''

        self.score_key = 'score'

        self.print('Loading ', loc)
        scores = self.common_load(loc, col_name=col_name)
       
        #Rename the column with score to default score key name
        scores = scores.rename({col_name: self.score_key}, axis=1)

        if data_type == 'binary' or data_type == 'b':

            #Processing for binary, with some tolerance to funky input
            scores, self.score_encoder = process_binary_input(scores, self.score_key, self.verbose)

        elif data_type == 'categorical' or data_type == 'c':
            
            #Processing for categorical input
            scores, self.score_key, self.score_encoder = process_categorical_input(scores, self.score_key,
                                                                         drop=None, verbose=self.verbose)
        elif (data_type == 'float' or data_type == 'ordinal' or
              data_type == 'f' or data_type == 'o'):

            if filter_outlier_percent != None:
                scores = filter_float_by_outlier(scores, self.score_key, filter_outlier_percent,
                                                         in_place=True, verbose=self.verbose)
               
        self.print('Final shape: ', scores.shape)
        self.scores = scores #By default on store one scores, so don't merge
        self.process_new()

    def load_strat_values(self,
                          loc,
                          col_names
                          ):
        '''
        Load stratification values from a 2.0_ABCD_Data_Explorer release formatted csv

        strat_loc -- Location of the csv with stratification values 
        strat_col_names -- list of column names to load
        '''
        
        self.print('Reading stratification values from', loc)
        strat, col_names = self.common_load(loc, col_names=col_names)

        #Encode each column into unique values
        for col in col_names:
            
            label_encoder = LabelEncoder()
            strat[col] = label_encoder.fit_transform(strat[col])
            self.strat_encoders[col] = label_encoder

        self.strat = self.merge_existing(self.strat, strat)
        self.process_new()

    def load_custom_scores(self,
                           scores_loc,
                           subject_ind=0,
                           score_ind=1,
                           filter_outlier_percent = None
                           ):

        '''
        Loads in a set of subject ids and associated scores from a custom csv scores dataset

        scores_loc -- The location of the scores csv file
        subject_ind -- The column index within the csv where subject is saved, if not already named src_subject_id
        score_ind -- The column index within the csv where the score is saved, if not already named score
        filter_outlier_percent -- A percent of values to exclude from either end of the score distribution,
            provided as either 1 number, or a tuple (% from lower, % from higher). None, to perform no filtering.
        '''

        pass

        scores = pd.read_csv(scores_loc, na_values=self.default_na_values)
        column_names = list(scores)
       
        #Rename subject column to src_subject_id
        if 'src_subject_id' not in column_names:
            
            scores = scores.rename({column_names[subject_ind]: 'src_subject_id'}, axis=1)
            self.print(column_names[subject_ind], 'renamed to: src_subject_id')

        #Perform common corrections on the scores dataframe
        scores = self.proc_df(scores)

        #Rename the column with score, to score
        if 'score' not in column_names:
            
            scores = scores.rename({column_names[score_ind]: 'score'}, axis=1)
            self.print(column_names[score_ind], 'renamed to: score')

        #Dropping missing scores, or scores that are NaN
        invalid_score_inds = scores[np.isnan(scores.score)].index
        scores = scores.drop(index=invalid_score_inds)
        self.print('Dropped', len(invalid_score_inds), 'scores/subjects for NaN scores')
        self.print('Min-Max Score (before outlier filtering):', np.min(scores.score), np.max(scores.score))

        if filter_outlier_percent != None:
            scores = self.filter_by_outlier(scores, 'score', filter_outlier_percent)
            self.print('Filtered score for outliers, dropping rows with params: ', filter_outlier_percent)
            self.print('Min-Max Score (post outlier filtering):', np.min(scores.score), np.max(scores.score))

        self.print('Final shape: ', scores.shape)
        self.scores = scores
        self.process_new()

    def load_exclusions(self,
                        loc=None,
                        exclusions=None
                        ):
        '''
        Loads in a set of excluded subjects, from either a file or directly passed in

        exclusion_loc -- Location of a file with a list of excluded subjects.
        Set to None if passing in exclusion instead.
        File should be formatted as one subject per line
        exclusions -- A python list or set containing subjects to be excluded

        Note: if default subject id behavior is set to false, reading subjects from exclusion loc might break
        '''

        self.exclusions.update(self.load_set_of_subjects(loc=loc, subjects=exclusions))
        self.print('Total excluded subjects: ', len(self.exclusions))
        self.process_new()

    def load_set_of_subjects(self,
                             loc=None,
                             subjects=None):

        '''
        Function to load in a set of subjects from either a saved location, or
        directly passed in as a set or list of subjects. 
        '''

        loaded_subjects = set()

        if loc is not None:
            with open(loc, 'r') as f:
                lines = f.readlines()

                for line in lines:
                    subject = line.rstrip()
                    loaded_subjects.add(self.process_subject_name(subject))

        if subjects is not None:
            loaded_subjects = set([self.process_subject_name(s) for s in subjects])
        
        return loaded_subjects

    def clear_exclusions(self):
        ''' Resets exclusions to be an empty set '''

        self.exclusions = set()

    def define_validation_strategy(self,
                                   groups = None,
                                   stratify = None
                                   ):
        
        '''
        Define a validation stratagy to be used during different train/test splits,
        in addition to model selection and model hyperparameter CV. 
           
        In general, these options are:
        
            Random: Just make splits randomly
            Group Preserving: Make splits that ensure subjects that are part of specific group
                are all within the same fold e.g., split by family, so that people with the same family id
                are always a part of the same fold.
            Stratifying: Make splits such that the distribution of a given group is as equally split between
                two folds as possible, so simmilar to matched halves or e.g., in a binary or categorical predictive
                context, splits could be done to ensure roughly equal distribution of dependent class.

        For now, it is possible to define only one overarching stratagy (One could imagine combining group preserving splits
            while also trying to stratify for class, but the logistics become more complicated).
            Though, within one strategy it is certainly possible to provide multiple values e.g.,
            for stratification you can stratify by score (the dependent variable to be predicted) as well as say sex,
            though with addition of unique value, the size of the smallest unique group decreases.

        By default (if this function is not called) just random.

        groups -- string or list (if merging multiple groups) of loaded strat column names, to preserve by.
        stratify -- string or list (if merging multiple) of loaded strat column names (AND/OR 'score' for binary/categorical),
            to preserve distribution between folds.
        '''

        if groups is not None:

            if type(groups) is str:
                self.CV = CV(groups = self.strat[groups])
            elif type(groups) is list:
                self.CV = CV(groups = get_unique_combo(self.strat, groups))

        elif stratify is not None:
            
            if type(stratify) is str:

                if stratify == 'score':
                    self.strat[self.score_key] = self.scores[self.score_key]
                    stratify = self.score_key

                self.CV = CV(stratify = self.strat[stratify])

            elif type(stratify) is list:

                if 'score' in list:
                    self.strat[self.score_key] = self.scores[self.score_key]
                    stratify = [self.score_key if s == 'score' else s for s in stratify]

                self.CV = CV(stratify = get_unique_combo(self.strat, stratify))

    def process_new(self):
        '''Internal function to handle keeping an overlapping subject list'''

        valid_subjects = []

        if len(self.data) > 0:
            valid_subjects.append(set(self.data.index))
        if len(self.covars) > 0:
            valid_subjects.append(set(self.covars.index))
        if len(self.scores) > 0:
            valid_subjects.append(set(self.scores.index))
        if len(self.strat) > 0:
            valid_subjects.append(set(self.strat.index))

        overlap = set.intersection(*valid_subjects)
        overlap = overlap - self.exclusions

        self.print('Removing non overlapping + excluded subjects')

        if len(self.data) > 0:
            self.data = self.data[self.data.index.isin(overlap)]
        if len(self.covars) > 0:
            self.covars = self.covars[self.covars.index.isin(overlap)]
        if len(self.scores) > 0:
            self.scores = self.scores[self.scores.index.isin(overlap)]
        if len(self.strat) > 0:
            self.strat = self.strat[self.strat.index.isin(overlap)]

        self.print('Total subjects = ', len(overlap))
        self.print()

    def load_name_mapping(self,
                          loc,
                          existing_name_col = "NDAR name",
                          changed_name_col = "REDCap name/NDA alias"
                          ):
        '''
        Loads a mapping dictionary for column names

        Keyword arguments:
        name_map_loc -- The location of the csv file which contains the mapping
        existing_name_col -- The column name with the file which lists names to be changed
        changed_name_col -- The column name within the file which lists the new name
        '''
    
        mapping = pd.read_csv(loc)

        try:
            self.name_map = dict(zip(mapping[existing_name_col], mapping[changed_name_col]))
        except KeyError:
            print('Error: One or both provided column names do not exist!')

        self.print('Loaded map file')

    def proc_df(self,
                data
                ):
        '''
        Internal function, when passed a 2.0_ABCD_Data_Explorer release formatted dataframe
        perform common processing steps.

        data - Input pandas dataframe
        '''

        assert 'src_subject_id' in data, "Missing subject id column!"

        #Perform corrects on subject ID
        data.src_subject_id = data.src_subject_id.apply(self.process_subject_name)

        #Filter by eventname is applicable
        data = self.filter_by_eventname(data)

        #Drop any duplicate subjects, default behavior for now
        #though, could imagine a case where you wouldn't want to when there are future releases
        data = data.drop_duplicates(subset = 'src_subject_id')

        #Rename columns if loaded name map
        if self.name_map:
            data = data.rename(self.name_map, axis=1)

        data = data.set_index('src_subject_id')

        return data

    def filter_by_eventname(self, data):
        '''Helper function to filter a dataframe by eventname'''

        #Filter data by eventname
        if self.eventname:
            data = data[data['eventname'] == self.eventname]
            data = data.drop('eventname', axis=1)
        
        #If eventname none, but still exists, take out the column
        else:
            if 'eventname' in list(data):
                data = data.drop('eventname', axis=1)

        return data

    def process_subject_name(self, subject):
        '''Function to be applied to subject name in order to ensure they are in a
           standardized format'''

        if self.use_default_subject_ids:
            subject = subject.strip().upper()
            
            if 'NDAR_' not in subject:
                subject = 'NDAR_' + subject
            return subject

        else:
            return subject
    
    def drop_na(self, data):
        '''Helper function to drop rows with na'''
        
        missing_values = data.isna().any(axis=1)
        data = data.dropna()
        self.print('Dropped', sum(missing_values), 'rows for missing values')

        return data
    
    def train_test_split(self,
                         test_size=None,
                         test_loc=None,
                         test_subjects=None,
                         random_state=None):

        '''
        Define the overarching train / test split.
           
        test_size -- If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split.
            If int, represents the absolute number of test groups.
            Set to None if using test_loc or test_subjects.
        test_loc -- Location of file to load in test subjects from.
            File should be formatted as one subject per line.
        test_subjects -- List or set of test subjects to pass in directly.
        random_state -- If using test_size, then can optionally provide a random state
        '''
        
        if self.all_data is None:
            self.prepare_data()

        if test_size is not None:
            self.train_subjects, self.test_subjects = self.CV.train_test_split(self.all_data.index, test_size, random_state)

        else:
            test_subjects = self.load_set_of_subjects(loc=test_loc, subjects=test_subjects)
            train_subjects = [subject for subject in self.all_data.index if subject not in test_subjects]
            self.train_subjects, self.test_subjects = pd.Index(train_subjects, name='src_subject_id'), pd.Index(test_subjects, name='src_subject_id')
        
        self.print('Performed train/test split, train size:', len(self.train_subjects), 'test size: ', len(self.test_subjects))

    def prepare_data(self):
        '''Prepares all loaded data into self.all_data for use directly in ML'''

        dfs = []

        assert len(self.scores > 0), 'Scores must be loaded!'
        assert len(self.data) > 0 or len(self.covars) > 0, 'Some data must be loaded!'

        if len(self.data) > 0:
            dfs.append(self.data)
            self.data_keys = list(self.data)

        if len(self.covars) > 0:
            dfs.append(self.covars)
            self.covars_keys = list(self.covars)
        
        dfs.append(self.scores)

        self.all_data = dfs[0]
        for i in range(1, len(dfs)):
            self.all_data = pd.merge(self.all_data, dfs[i], on='src_subject_id')

    from ABCD_ML._ML import evaluate_regression_model
    from ABCD_ML._ML import test_regression_model
    from ABCD_ML._ML import premodel_check
        
        
    