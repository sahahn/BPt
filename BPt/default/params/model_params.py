from .Params import TransitionChoice, Log, Scalar, Array
import numpy as np

P = {}
P['default'] = {}

cls_weight = TransitionChoice([None, 'balanced'])

# Logistic
P['base logistic'] =\
    {'max_iter': 100,
     'multi_class': 'auto',
     'penalty': 'none',
     'class_weight': None,
     'solver': 'lbfgs'}

# Ridge classifier
P['base ridge'] = {'max_iter': 100,
                   'penalty': 'l2',
                   'solver': 'saga'}

P['ridge C'] =\
    {'max_iter': 100,
     'solver': 'saga',
     'C': Log(lower=1e-5, upper=1e3),
     'class_weight': cls_weight}

P['ridge C extra'] = P['ridge C'].copy()
P['ridge C extra']['max_iter'] =\
    Scalar(lower=100, upper=1000).set_integer_casting()
P['ridge C extra']['tol'] = Log(lower=1e-6, upper=.01)

# Ridge regressor
P['base ridge regressor'] = {'max_iter': 100,
                             'solver': 'lsqr'}

P['ridge regressor dist'] = P['base ridge regressor'].copy()
P['ridge regressor dist']['alpha'] = Log(lower=1e-3, upper=1e5)

# Lasso regressor
P['base lasso regressor'] = {'max_iter': 100}
P['lasso regressor dist'] = P['base lasso regressor'].copy()
P['lasso regressor dist']['alpha'] = Log(lower=1e-5, upper=1e5)

# Lasso classifier
P['base lasso'] = P['base logistic'].copy()
P['base lasso']['solver'] = 'liblinear'
P['base lasso']['penalty'] = 'l1'

P['lasso C'] = P['base lasso'].copy()
P['lasso C']['C'] = Log(lower=1e-5, upper=1e3)
P['lasso C']['class_weight'] = cls_weight

P['lasso C extra'] = P['lasso C'].copy()
P['lasso C extra']['max_iter'] =\
    Scalar(lower=100, upper=1000).set_integer_casting()
P['lasso C extra']['tol'] = Log(lower=1e-6, upper=.01)

# Elastic net classifier
P['base elastic'] = P['base logistic'].copy()
P['base elastic']['penalty'] = 'elasticnet'
P['base elastic']['l1_ratio'] = .5
P['base elastic']['solver'] = 'saga'

P['elastic classifier'] = P['base elastic'].copy()
P['elastic classifier']['C'] = Log(lower=1e-5, upper=1e5)
P['elastic classifier']['l1_ratio'] = Scalar(lower=.01, upper=1)
P['elastic classifier']['class_weight'] = cls_weight

P['elastic clf v2'] = P['elastic classifier'].copy()
P['elastic clf v2']['C'] = Log(lower=1e-2, upper=1e5)

P['elastic classifier extra'] = P['elastic classifier'].copy()
P['elastic classifier extra']['max_iter'] =\
    Scalar(lower=100, upper=1000).set_integer_casting()
P['elastic classifier extra']['tol'] = Log(lower=1e-6, upper=.01)

# Elastic net regression
P['base elastic net'] = {'max_iter': 100}
P['elastic regression'] = P['base elastic net'].copy()
P['elastic regression']['alpha'] = Log(lower=1e-5, upper=1e5)
P['elastic regression']['l1_ratio'] = Scalar(lower=.01, upper=1)

P['elastic regression extra'] = P['elastic regression'].copy()
P['elastic regression extra']['max_iter'] =\
    Scalar(lower=100, upper=1000).set_integer_casting()
P['elastic regression extra']['tol'] = Log(lower=1e-6, upper=.01)

# Other
P['base huber'] = {'epsilon': 1.35}
P['base gnb'] = {'var_smoothing': 1e-9}

P['base knn'] = {'n_neighbors': 5}
P['knn dist'] = {'weights': TransitionChoice(['uniform', 'distance']),
                 'n_neighbors':
                 Scalar(lower=2, upper=25).set_integer_casting()}
P['base knn regression'] = P['base knn'].copy()
P['knn dist regression'] = P['knn dist'].copy()

P['base dt'] = {}
P['dt dist'] = {'max_depth':
                Scalar(lower=1, upper=30).set_integer_casting(),
                'min_samples_split':
                Scalar(lower=2, upper=50).set_integer_casting()}

P['dt classifier dist'] = P['dt dist'].copy()
P['dt classifier dist']['class_weight'] = cls_weight

P['base linear'] = {'fit_intercept': True}

P['base rf'] = {'n_estimators': 100}
P['base rf regressor'] = P['base rf'].copy()

# Forest based
n_estimators = Scalar(init=100, lower=3, upper=500).set_integer_casting()
depths =\
    TransitionChoice(
        [None, Scalar(init=25, lower=2, upper=200).set_integer_casting()])

P['rf dist'] = {'n_estimators': n_estimators,
                'max_depth': depths,
                'max_features': Scalar(lower=.1, upper=1.0),
                'min_samples_split': Scalar(lower=.1, upper=1.0),
                'bootstrap': True}

P['rf classifier dist'] = P['rf dist'].copy()
P['rf classifier dist']['class_weight'] = cls_weight

# Light gbm params
P['base lgbm'] = {'silent': True}


P['lgbm dist1'] =\
    {'silent': True,
     'boosting_type': TransitionChoice(['gbdt', 'dart', 'goss']),
     'n_estimators': n_estimators,
     'num_leaves':
         Scalar(init=20, lower=6, upper=80).set_integer_casting(),
         'min_child_samples':
         Scalar(lower=10, upper=500).set_integer_casting(),
         'min_child_weight':
         Log(lower=1e-5, upper=1e4),
         'subsample':
         Scalar(lower=.3, upper=.95),
         'colsample_bytree':
         Scalar(lower=.3, upper=.95),
         'reg_alpha': TransitionChoice([0, Log(lower=1e-5, upper=1)]),
         'reg_lambda': TransitionChoice([0, Log(lower=1e-5, upper=1)])}

P['lgbm dist2'] =\
    {'silent': True,
     'lambda_l2': 0.001,
     'boosting_type': TransitionChoice(['gbdt', 'dart']),
     'min_child_samples':
         TransitionChoice([1, 5, 7, 10, 15, 20, 35, 50, 100, 200, 500, 1000]),
         'num_leaves':
         TransitionChoice([2, 4, 7, 10, 15, 20, 25, 30, 35,
                           40, 50, 65, 80, 100, 125, 150, 200, 250]),
         'colsample_bytree':
         TransitionChoice([0.7, 0.9, 1.0]),
         'subsample':
         Scalar(lower=.3, upper=1),
         'learning_rate':
         TransitionChoice([0.01, 0.05, 0.1]),
         'n_estimators':
         TransitionChoice([5, 20, 35, 50, 75, 100, 150,
                           200, 350, 500, 750, 1000])}

P['lgbm dist3'] = {'silent': True,
                   'n_estimators': 1000,
                   'early_stopping_rounds': 150,
                   'eval_split': .2,
                   'boosting_type': 'gbdt',
                   'learning_rate':
                   'Log(lower=5e-3, upper=.2, init=.1)',
                   'colsample_bytree':
                   'Scalar(lower=.75, upper=1, init=1)',
                   'min_child_samples':
                   Scalar(lower=2, upper=30, init=20).set_integer_casting(),
                   'num_leaves':
                   Scalar(lower=16, upper=96, init=31).set_integer_casting()}

P['lgbm classifier dist1'] = P['lgbm dist1'].copy()
P['lgbm classifier dist1']['class_weight'] = cls_weight

P['lgbm classifier dist2'] = P['lgbm dist2'].copy()
P['lgbm classifier dist2']['class_weight'] = cls_weight

P['lgbm classifier dist3'] = P['lgbm dist3'].copy()
P['lgbm classifier dist3']['class_weight'] = cls_weight

P['base xgb'] = {'verbosity': 0,
                 'objective': 'reg:squarederror'}

P['base xgb classifier'] = P['base xgb'].copy()
P['base xgb classifier']['objective'] = 'binary:logistic'

P['xgb dist1'] =\
    {'verbosity': 0,
     'objective': 'reg:squarederror',
     'n_estimators': n_estimators,
     'min_child_weight': Log(lower=1e-5, upper=1e4),
     'subsample': Scalar(lower=.3, upper=.95),
     'colsample_bytree': Scalar(lower=.3, upper=.95),
     'reg_alpha': TransitionChoice([0, Log(lower=1e-5, upper=1)]),
     'reg_lambda': TransitionChoice([0, Log(lower=1e-5, upper=1)])}

P['xgb dist2'] =\
    {'verbosity': 0,
     'objective': 'reg:squarederror',
     'max_depth': depths,
     'learning_rate': Scalar(lower=.01, upper=.5),
     'n_estimators': Scalar(lower=3, upper=500).set_integer_casting(),
     'min_child_weight': TransitionChoice([1, 5, 10, 50]),
     'subsample': Scalar(lower=.5, upper=1),
     'colsample_bytree': Scalar(lower=.4, upper=.95)}

P['xgb dist3'] =\
    {'verbosity': 0,
     'objective': 'reg:squarederror',
     'learning_rare': Scalar(lower=.005, upper=.3),
     'min_child_weight': Scalar(lower=.5, upper=10),
     'max_depth': TransitionChoice(np.arange(3, 10)),
     'subsample': Scalar(lower=.5, upper=1),
     'colsample_bytree': Scalar(lower=.5, upper=1),
     'reg_alpha': Log(lower=.00001, upper=1)}

P['xgb classifier dist1'] = P['xgb dist1'].copy()
P['xgb classifier dist1']['objective'] = 'binary:logistic'

P['xgb classifier dist2'] = P['xgb dist2'].copy()
P['xgb classifier dist2']['objective'] = 'binary:logistic'

P['xgb classifier dist3'] = P['xgb dist3'].copy()
P['xgb classifier dist3']['objective'] = 'binary:logistic'

P['base gp regressor'] = {'n_restarts_optimizer': 5,
                          'normalize_y': True}
P['base gp classifier'] = {'n_restarts_optimizer': 5}

# probability = True
P['base svm'] = {'kernel': 'rbf',
                 'gamma': 'scale'}

P['svm dist'] = P['base svm'].copy()
P['svm dist']['C'] = Log(lower=1e-4, upper=1e4)
P['svm dist']['gamma'] = Log(lower=1e-6, upper=1)

P['base svm classifier'] = P['base svm'].copy()
P['base svm classifier']['probability'] = True

P['svm classifier dist'] = P['svm dist'].copy()
P['svm classifier dist']['probability'] = True
P['svm classifier dist']['class_weight'] = cls_weight


# Define different choices for the mlp
P['base mlp'] = {}

sizes = Scalar(init=200, lower=50, upper=400).set_integer_casting()
batch_size = TransitionChoice(['auto', sizes])

P['mlp dist 1 layer'] =\
    {'hidden_layer_sizes':
     Scalar(init=100, lower=2, upper=300).set_integer_casting(),
     'activation':
         TransitionChoice(['identity', 'logistic', 'tanh', 'relu']),
         'alpha': Log(lower=1e-5, upper=1e2),
         'batch_size': batch_size,
         'learning_rate':
         TransitionChoice(['constant', 'invscaling', 'adaptive']),
         'learning_rate_init': Log(lower=1e-5, upper=1e2),
         'max_iter':
         Scalar(init=200, lower=100, upper=1000).set_integer_casting(),
         'beta_1': Scalar(init=.9, lower=.1, upper=.99),
         'beta_2': Scalar(init=.999, lower=.1, upper=.9999)}

P['mlp dist es 1 layer'] = P['mlp dist 1 layer'].copy()
P['mlp dist es 1 layer']['early_stopping'] = True
P['mlp dist es 1 layer']['n_iter_no_change'] =\
    Scalar(lower=5, upper=50).set_integer_casting()

two_layer = Array(init=(100, 100)).set_mutation(sigma=50)
two_layer = two_layer.set_bounds(lower=1, upper=300).set_integer_casting()

P['mlp dist 2 layer'] = P['mlp dist 1 layer'].copy()
P['mlp dist 2 layer']['hidden_layer_sizes'] = two_layer

P['mlp dist es 2 layer'] = P['mlp dist es 1 layer'].copy()
P['mlp dist 2 layer']['hidden_layer_sizes'] = two_layer

three_layer = Array(init=(100, 100, 100)).set_mutation(sigma=50)
three_layer = three_layer.set_bounds(lower=1, upper=300).set_integer_casting()

P['mlp dist 3 layer'] = P['mlp dist 1 layer'].copy()
P['mlp dist 3 layer']['hidden_layer_sizes'] = three_layer

P['mlp dist es 3 layer'] = P['mlp dist es 1 layer'].copy()
P['mlp dist 3 layer']['hidden_layer_sizes'] = three_layer

P['base linear svc'] = {'max_iter': 100}

P['linear svc dist'] = P['base linear svc'].copy()
P['linear svc dist']['C'] = Log(lower=1e-4, upper=1e4)
P['linear svc dist']['class_weight'] = cls_weight

P['base linear svr'] = {'loss': 'epsilon_insensitive',
                        'max_iter': 100}

P['linear svr dist'] = P['base linear svr'].copy()
P['linear svr dist']['C'] = Log(lower=1e-4, upper=1e4)

P['base sgd'] = {'loss': 'squared_loss'}


loss_choice = TransitionChoice(['hinge', 'log', 'modified_huber',
                                'squared_hinge', 'perceptron'])

lr_choice = TransitionChoice(['optimal', 'invscaling',
                              'adaptive', 'constant'])

P['sgd classifier big search'] =\
    {'loss': loss_choice,
     'penalty': TransitionChoice(['l2', 'l1', 'elasticnet']),
     'alpha': Log(lower=1e-5, upper=1e2),
     'l1_ratio': Scalar(lower=.01, upper=1),
     'max_iter': 100,
     'learning_rate': lr_choice,
     'eta0': Log(lower=1e-6, upper=1e3),
     'power_t': Scalar(lower=.1, upper=.9),
     'early_stopping': TransitionChoice([False, True]),
     'validation_fraction': Scalar(lower=.05, upper=.5),
     'n_iter_no_change': Scalar(lower=5, upper=30).set_integer_casting(),
     'class_weight': cls_weight}

# Make elastic net version
P['sgd elastic'] =\
    {'loss': 'squared_epsilon_insensitive',
     'penalty': 'elasticnet',
     'alpha': Log(lower=1e-5, upper=1e5),
     'l1_ratio': Scalar(lower=.01, upper=1)}

P['sgd elastic classifier'] = P['sgd elastic'].copy()
P['sgd elastic classifier']['class_weight'] = cls_weight

# Auto gluon
P['pt binary'] = {'problem_type': 'binary'}
P['pt multiclass'] = {'problem_type': 'multiclass'}
P['pt regression'] = {'problem_type': 'regression'}
