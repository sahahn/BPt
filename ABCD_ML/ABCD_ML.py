import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class ABCD_ML():

    def __init__(self,
                 eventname = 'baseline_year_1_arm_1',
                 use_default_subject_ids = True,
                 verbose = True
                 ):

        self.eventname = eventname
        self.use_default_subject_ids = use_default_subject_ids
        self.verbose = verbose

        self.data, self.covars = [], []
        self.scores, self.strat = [], []
        self.name_map, self.exclusions = {}, set()

    def print(self, *args):
        '''Overriding the print function to allow from verbosity'''

        if self.verbose:
            print(*args)
    
    def load_data(self,
                  data_loc,
                  drop_keys = []
                  ):
        ''' 
        Load a 2.0_ABCD_Data_Explorer release formatted neuroimaging dataset of ROI's

        data_loc -- The location of the dataset csv file
        eventname -- The eventname to select datapoints by, if None then ignore
        drop_keys -- A list of keys to drop columns by, where if any key given in a columns name,
        then that column will be dropped (Note: if a name mapping exists, this drop step will be conducted after renaming)
        '''
        
        self.print('Loading', data_loc) 
        data = pd.read_csv(data_loc)

        #Drop the first two columns by default (typically specific id, then dataset id for NDA csvs)
        first_cols = list(data)[:2]
        data = data.drop(first_cols, axis=1)
        self.print('dropped', first_cols)

        #Perform common operations (check subject id, drop duplicate subjects ect...)
        data = self.proc_df(data)

        #Drop any columns if any of the drop keys occur in the column name
        column_names = list(data)

        to_drop = [name for name in column_names for drop_key in drop_keys if drop_key in name]
        data = data.drop(to_drop, axis=1)
        self.print('Dropped', len(to_drop), ' columns, per drop_keys argument')

        missing_values = data.isna().any(axis=1)
        data = data.dropna()
        self.print('Dropped', sum(missing_values), 'rows for missing values')

        self.print('loaded shape: ', data.shape)

        #If other data is already loaded, merge this data with existing loaded data
        if len(self.data) > 0:

            self.data = pd.merge(self.data, data, on='src_subject_id')
            self.print('Merged with existing data, new data shape: ', self.data.shape)
        
        else:
            self.data = data

        self.process_new()

    def load_custom_data(self):
        pass

    def load_covars(self):
        pass

    def load_scores(self,
                    scores_loc,
                    score_col_name,
                    ):

        '''
        Loads in a set of subject ids and associated scores from a 2.0_ABCD_Data_Explorer release formatted csv

        scores_loc -- The location of the scores csv file
        score_col_name -- The name of the column with the score of interest
        eventname -- The eventname to select datapoints by, if None then ignore
        (Note: if a name mapping exists, the score col name will refer to the changes name)
        '''
        
        self.print('Loading ', scores_loc)

        #Read in the dataframe
        scores = pd.read_csv(scores_loc)

        #Perform common corrections on the scores dataframe
        scores = self.proc_df(scores)

        assert score_col_name in scores, "No column with name " + score_col_name

        #Rename the column with score, to score
        scores = scores.rename({score_col_name: 'score'}, axis=1)

        #Limit the dataframe to just subject and score columns
        scores = scores[['src_subject_id', 'score']]

        #Dropping missing scores, or scores that are NaN
        invalid_score_inds = scores[np.isnan(scores.score)].index
        scores = scores.drop(index=invalid_score_inds)
        self.print('Dropped', len(invalid_score_inds), 'scores/subjects for NaN scores')

        self.print('Final shape: ', scores.shape)
        self.scores = scores
        self.process_new()

    def load_custom_scores(self,
                           scores_loc,
                           subject_ind=0,
                           score_ind=1
                           ):

        '''
        Loads in a set of subject ids and associated scores from a custom csv scores dataset

        scores_loc -- The location of the scores csv file
        subject_ind -- The column index within the csv where subject is saved, if not already named src_subject_id
        score_ind -- The column index within the csv where the score is saved, if not already named score
        '''

        scores = pd.read_csv(scores_loc)
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

        self.scores = scores
        self.process_new()

    def load_strat_values(self,
                          strat_loc,
                          strat_col_names
                          ):
        '''
        Load stratification values from a 2.0_ABCD_Data_Explorer release formatted csv

        strat_loc -- Location of the csv with stratification values 
        strat_col_names -- list of column names to load
        '''
        
        self.print('Reading stratification values from', strat_loc)
        strat = pd.read_csv(strat_loc)

        #Perform common df corrections
        strat = self.proc_df(strat)

        if type(strat_col_names) != list:
            strat_col_names = list(strat_col_names)

        #Limit the dataframe to just the subject id and columns of interest
        to_keep = ['src_subject_id'] + strat_col_names
        strat = strat[to_keep]

        #Save the various encoders, in case they need to be decoded
        self.strat_encoders = {}
        
        #Encode each column into unique values
        for col in strat_col_names:
            
            le = LabelEncoder()
            strat[col] = le.fit_transform(strat[col])
            self.strat_encoders[col] = le

        self.strat = strat
        self.process_new()

    def load_exclusions(self,
                        exclusions_loc=None,
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

        if exclusions_loc != None:
            with open(exclusions_loc, 'r') as f:
                lines = f.readlines()

                for line in lines:
                    subject = line.rstrip()
                    self.exclusions.add(self.process_subject_name(subject))

        if exclusions != None:

            exclusions = set([self.process_subject_name(s) for s in exclusions])
            self.exclusions.update(exclusions)

        self.print('Total excluded subjects: ', len(self.exclusions))
        self.process_new()

    def clear_exclusions(self):
        ''' Resets exclusions to be an empty set '''

        self.exclusions = set()

    def process_new(self):
        '''Internal function to handle keeping an overlapping subject list'''

        valid_subjects = []

        if len(self.data) > 0:
            valid_subjects.append(set(self.data['src_subject_id']))
        if len(self.covars) > 0:
            valid_subjects.append(set(self.covars['src_subject_id']))
        if len(self.scores) > 0:
            valid_subjects.append(set(self.scores['src_subject_id']))
        if len(self.strat) > 0:
            valid_subjects.append(set(self.strat['src_subject_id']))

        overlap = set.intersection(*valid_subjects)
        overlap = overlap - self.exclusions

        self.print('Removing non overlapping + excluded subjects')

        if len(self.data) > 0:
            self.data = self.data[self.data['src_subject_id'].isin(overlap)]
        if len(self.covars) > 0:
            self.covars = self.covars[self.covars['src_subject_id'].isin(overlap)]
        if len(self.scores) > 0:
            self.scores = self.scores[self.scores['src_subject_id'].isin(overlap)]
        if len(self.strat) > 0:
            self.strat = self.strat[self.strat['src_subject_id'].isin(overlap)]

        self.print('Total subjects = ', len(overlap))

    def load_name_mapping(self,
                          name_map_loc,
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
    
        mapping = pd.read_csv(name_map_loc)

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




    

    