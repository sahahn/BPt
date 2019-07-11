
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.model_selection import ParameterSampler, KFold, train_test_split
from scipy.stats import (randint as sp_randint, uniform as sp_uniform)
import numpy as np

DEFAULT_PARAM_GRID = {
            'num_leaves'       : sp_randint(6, 50), 
            'min_child_samples': sp_randint(100, 500), 
            'min_child_weight' : [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
            'subsample'        : sp_uniform(loc=0.2, scale=0.8), 
            'colsample_bytree' : sp_uniform(loc=0.4, scale=0.6),
            'reg_alpha'        : [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
            'reg_lambda'       : [0, 1e-1, 1, 5, 10, 20, 50, 100]
            }

def Train_Light_GBM(X, y, int_cv=3, regression=True, n_params=10, test_size=.2, n_jobs=1, e_stop_rounds=300, **kwargs):
    '''
    Wrapper function to train a Light GBM regression or classifier model
       X - Training input
       y - Training labels
       int_cv - Number of internal cross validation folds
       regression - True for regression, False for binary classification
       n_params - Number of different random hyperparam combinations to explore
       test_size - Size in (%) of the outer test fold (to use for final validation fit)
       n_jobs - Number of proc. to use
       e_stop_rounds - Number of early stop rounds used in checking parameters (double used in final fit)
       
    '''

    if regression:
        Base_Model = LGBMRegressor
    else:
        Base_Model = LGBMClassifier

    #Train val split, for final fit
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)
    
    param_scores = []
    param_list = list(ParameterSampler(DEFAULT_PARAM_GRID, n_iter=n_params))
    int_skf = KFold(n_splits=int_cv)
    
    for p in range(n_params):
        best_scores = []
       
        for train_ind, test_ind in int_skf.split(X_train, y_train):
            
            int_X_train, int_y_train = X_train[train_ind], y_train[train_ind]
            int_X_test, int_y_test = X_train[test_ind], y_train[test_ind]
            
            model = Base_Model(n_jobs=n_jobs, silent=True, n_estimators=5000, **param_list[p])
            model.fit(int_X_train, int_y_train, eval_set=(int_X_test, int_y_test),
                        verbose=False, early_stopping_rounds=e_stop_rounds)
            
            best_score = list(model.best_score_['valid_0'].values())[0]
            best_scores.append(best_score)
        
        param_scores.append(np.mean(best_scores))
    
    bp_ind = np.argmin(param_scores) #Index of best parameters
    model = Base_Model(n_jobs=n_jobs, silent=True, n_estimators=5000, **param_list[bp_ind])
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False, early_stopping_rounds=int(e_stop_rounds*2))

    return model




    '''
def load_custom_scores(self,
                        scores_loc,
                        subject_ind=0,
                        score_ind=1,
                        filter_outlier_percent = None
                        ):


    Loads in a set of subject ids and associated scores from a custom csv scores dataset

    scores_loc -- The location of the scores csv file
    subject_ind -- The column index within the csv where subject is saved, if not already named src_subject_id
    score_ind -- The column index within the csv where the score is saved, if not already named score
    filter_outlier_percent -- A percent of values to exclude from either end of the score distribution,
        provided as either 1 number, or a tuple (% from lower, % from higher). None, to perform no filtering.


    print('Not implemented!')
    pass

    #Old version below~

    scores = pd.read_csv(scores_loc, na_values=self.default_na_values)
    column_names = list(scores)

    #Rename subject column to src_subject_id
    if 'src_subject_id' not in column_names:

        scores = scores.rename({column_names[subject_ind]: 'src_subject_id'}, axis=1)
        self._print(column_names[subject_ind], 'renamed to: src_subject_id')

    #Perform common corrections on the scores dataframe
    scores = self.proc_df(scores)

    #Rename the column with score, to score
    if 'score' not in column_names:

        scores = scores.rename({column_names[score_ind]: 'score'}, axis=1)
        self._print(column_names[score_ind], 'renamed to: score')

    #Dropping missing scores, or scores that are NaN
    invalid_score_inds = scores[np.isnan(scores.score)].index
    scores = scores.drop(index=invalid_score_inds)
    self._print('Dropped', len(invalid_score_inds), 'scores/subjects for NaN scores')
    self._print('Min-Max Score (before outlier filtering):', np.min(scores.score), np.max(scores.score))

    if filter_outlier_percent != None:
        scores = self.filter_by_outlier(scores, 'score', filter_outlier_percent)
        self._print('Filtered score for outliers, dropping rows with params: ', filter_outlier_percent)
        self._print('Min-Max Score (post outlier filtering):', np.min(scores.score), np.max(scores.score))

    self._print('Final shape: ', scores.shape)
    self.scores = scores
    self._process_new()
'''
