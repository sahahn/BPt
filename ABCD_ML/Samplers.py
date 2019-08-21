from ABCD_ML.ML_Helpers import (get_obj_and_params, show_objects,
                                get_possible_init_params)
from imblearn.over_sampling import (RandomOverSampler, SMOTE, ADASYN,
                                    BorderlineSMOTE, SVMSMOTE, KMeansSMOTE,
                                    SMOTENC)
from imblearn.under_sampling import (ClusterCentroids, RandomUnderSampler,
                                     NearMiss, TomekLinks,
                                     EditedNearestNeighbours,
                                     RepeatedEditedNearestNeighbours,
                                     AllKNN, CondensedNearestNeighbour,
                                     OneSidedSelection,
                                     NeighbourhoodCleaningRule)
from imblearn.combine import SMOTEENN, SMOTETomek


AVALIABLE = {
        'binary': {
                'random over sampler': 'random over sampler',
                'smote': 'smote',
                'adasyn': 'adasyn',
                'borderline smote': 'borderline smote',
                'svm smote': 'svm smote',
                'kmeans smote': 'kmeans smote',
                'smote nc': 'smote nc',
                'cluster centroids': 'cluster centroids',
                'random under sampler': 'random under sampler',
                'near miss': 'near miss',
                'tomek links': 'tomek links',
                'enn': 'enn',
                'renn': 'renn',
                'all knn': 'all knn',
                'condensed nn': 'condensed nn',
                'one sided selection': 'one sided selection',
                'neighbourhood cleaning rule': 'neighbourhood cleaning rule',
                'smote enn': 'smote enn',
                'smote tomek': 'smote tomek',
        },
        'regression': {
        },
        'categorical': {
            'multilabel': {
            },
        }
}

AVALIABLE['categorical']['multiclass'] = AVALIABLE['binary'].copy()

SAMPLERS = {
    'random over sampler': (RandomOverSampler, ['default']),
    'smote': (SMOTE, ['default']),
    'adasyn': (ADASYN, ['default']),
    'borderline smote': (BorderlineSMOTE, ['default']),
    'svm smote': (SVMSMOTE, ['default']),
    'kmeans smote': (KMeansSMOTE, ['default']),
    'smote nc': (SMOTENC, ['default']),
    'cluster centroids': (ClusterCentroids, ['default']),
    'random under sampler': (RandomUnderSampler, ['default']),
    'near miss': (NearMiss, ['default']),
    'tomek links': (TomekLinks, ['default']),
    'enn': (EditedNearestNeighbours, ['default']),
    'renn': (RepeatedEditedNearestNeighbours, ['default']),
    'all knn': (AllKNN, ['default']),
    'condensed nn': (CondensedNearestNeighbour, ['default']),
    'one sided selection': (OneSidedSelection, ['default']),
    'neighbourhood cleaning rule': (NeighbourhoodCleaningRule, ['default']),
    'smote enn': (SMOTEENN, ['default']),
    'smote tomek': (SMOTETomek, ['default']),
}


def get_sampler_and_params(sampler_str, extra_params, param_ind, search_type):

    sampler, extra_sampler_params, sampler_params =\
        get_obj_and_params(sampler_str, SAMPLERS, extra_params,
                           param_ind, search_type)

    # Need to check if categorical features is a possible param
    possible_params = get_possible_init_params(sampler)
    if 'categorical_features' in possible_params:
            if 'categorical_features' not in extra_sampler_params:
                extra_sampler_params['categorical_features'] = None

    return sampler(**extra_sampler_params), sampler_params


def Show_Samplers(self, problem_type=None, sampler_str=None,
                  param_ind_options=False, show_object=False,
                  possible_params=False):
    '''Print out the avaliable feature selectors,
    optionally restricted by problem type + other diagnostic args.

    Parameters
    ----------
    problem_type : {binary, categorical, regression, None}, optional
        Where `problem_type` is the underlying ML problem

        (default = None)

    sampler_str : str or list, optional
        If `sampler_str` is passed, will just show the specific
        sampler, according to the rest of the params passed.
        Note : You must pass the specific sampler indicator str
        as limited preproc will be done on this input!
        If list, will show all samplers within list

        (default = None)

    show_param_ind_options : bool, optional
        Flag, if set to True, then will display the ABCD_ML
        param ind options for each sampler.

        (default = False)

    show_object : bool, optional
        Flag, if set to True, then will print the
        raw sampler object.

        (default = False)

    possible_params: bool, optional
        Flag, if set to True, then will print all
        possible arguments to the classes __init__

        (default = False)
    '''
    print('These are the different implemented options for re-sampling',
          ' imbalanced data.')
    print('Please check out:')
    print('https://imbalanced-learn.readthedocs.io/en/stable/user_guide.html')
    print('For detailed use on the different samplers.')
    print('More information through this function is avaliable')
    print('By passing optional extra optional params! Please view',
          'the help function for more info!')
    print('Note: the str indicator actually passed during Evaluate / Test')
    print('is listed as ("str indicator")')
    print()

    show_objects(problem_type, sampler_str,
                 param_ind_options, show_object, possible_params,
                 AVALIABLE, SAMPLERS)
