import sklearn.model_selection as MS
import numpy as np

class CV():
    
    def __init__(self, groups=None, stratify=None):
        '''
        If not groups or stratify is passed, then by default just uses random.
        If groups is passed uses Group Folds, if Stratify, then Stratified Folds.
        '''

        self.groups = groups
        self.stratify = stratify

    def train_test_split(self, subjects, test_size=.2, random_state=None):

        if self.groups is not None:
            splitter = MS.GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            [*inds] = splitter.split(subjects, groups=self.groups.loc[subjects])
        
        elif self.stratify is not None:
            splitter = MS.StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            [*inds] = splitter.split(subjects, y=self.stratify.loc[subjects])

        else:
            splitter = MS.ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            [*inds] = splitter.split(subjects)

        inds = inds[0]

        train_subjects, test_subjects = subjects[inds[0]], subjects[inds[1]]
        return train_subjects, test_subjects

    def repeated_k_fold(self, subjects, n_repeats, n_splits, random_state=None):

        subjects = []
        for n in range(n_repeats):
            
            if random_state is not None:
                random_state += 1
            
            subjects += self.k_fold(subjects, n_splits, random_state)
        
        return subjects

    def k_fold(self, subjects, n_splits, random_state=None):

        #Special implementation for group K fold, just do KFold on unique groups
        if self.groups is not None:

            groups = self.groups.loc[subjects]

            unique_groups = np.unique(groups)
            splitter = MS.KFold(n_splits=n_splits, random_state=random_state)

            [*inds] =  splitter.split(unique_groups)
            subjects = [(groups.index[groups.isin(unique_groups[i[0]])],
                         groups.index[groups.isin(unique_groups[i[1]])]) for i in inds]

            return subjects

        elif self.stratify is not None:

            splitter = MS.StratifiedKFold(n_splits=n_splits, random_state=random_state)
            [*inds] = splitter.split(subjects, y=stratify.loc[subjects])

        else:
            
            splitter = MS.KFold(n_splits=n_splits, random_state=random_state)
            [*inds] = splitter.split(subjects)

        subjects = [(subjects[i[0]], subjects[i[1]]) for i in inds]
        return subjects


            





        
