.. _Models:
 
******
Models
******

Different availible choices for the `model` parameter are shown below.
`model` is accepted by :func:`Evaluate <ABCD_ML.ABCD_ML.ABCD_ML.Evaluate>` and :func:`Test <ABCD_ML.ABCD_ML.ABCD_ML.Test>`.
The exact str indicator for each `model` is represented by the sub-heading (within "")
The avaliable models are further broken down by which can workwith different problem_types.
Additionally, a link to the original models documentation as well as the implemented parameter distributions are shown.

binary
======
"dt classifier"
***************

  Base Class Documenation: :class:`sklearn.tree.DecisionTreeClassifier`

  Param Distributions

	0. "base dt" ::

		defaults only

	1. "dt classifier dist" ::

		max_depth: Array((1,), [ArctanBound(a_max=[30], a_min=[1], name=At(1,30), shape=(1,))])
		min_samples_split: Array((1,), [ArctanBound(a_max=[50], a_min=[2], name=At(2,50), shape=(1,))])
		class_weight: UnorderedDiscrete([None, 'balanced'])


"elastic net logistic"
**********************

  Base Class Documenation: :class:`sklearn.linear_model.LogisticRegression`

  Param Distributions

	0. "base elastic" ::

		max_iter: 5000
		multi_class: auto
		penalty: elasticnet
		class_weight: None
		solver: saga
		l1_ratio: 0.5

	1. "elastic classifier" ::

		max_iter: 5000
		multi_class: auto
		penalty: elasticnet
		class_weight: UnorderedDiscrete([None, 'balanced'])
		solver: saga
		l1_ratio: Array((1,), [ArctanBound(a_max=[1], a_min=[0.01], name=At(0.01,1), shape=(1,))])
		C: Log({a_min},{a_max},{width})

	2. "elastic classifier extra" ::

		max_iter: Array((1,), [ArctanBound(a_max=[10000], a_min=[1000], name=At(1000,10000), shape=(1,))])
		multi_class: auto
		penalty: elasticnet
		class_weight: UnorderedDiscrete([None, 'balanced'])
		solver: saga
		l1_ratio: Array((1,), [ArctanBound(a_max=[1], a_min=[0.01], name=At(0.01,1), shape=(1,))])
		C: Log({a_min},{a_max},{width})
		tol: Log({a_min},{a_max},{width})


"gaussian nb"
*************

  Base Class Documenation: :class:`sklearn.naive_bayes.GaussianNB`

  Param Distributions

	0. "base gnb" ::

		var_smoothing: 1e-09


"gp classifier"
***************

  Base Class Documenation: :class:`sklearn.gaussian_process.GaussianProcessClassifier`

  Param Distributions

	0. "base gp classifier" ::

		n_restarts_optimizer: 5


"knn classifier"
****************

  Base Class Documenation: :class:`sklearn.neighbors.KNeighborsClassifier`

  Param Distributions

	0. "base knn" ::

		n_neighbors: 5

	1. "knn dist" ::

		weights: UnorderedDiscrete(['uniform', 'distance'])
		n_neighbors: Array((1,), [ArctanBound(a_max=[25], a_min=[2], name=At(2,25), shape=(1,))])


"lasso logistic"
****************

  Base Class Documenation: :class:`sklearn.linear_model.LogisticRegression`

  Param Distributions

	0. "base lasso" ::

		max_iter: 5000
		multi_class: auto
		penalty: l1
		class_weight: None
		solver: liblinear

	1. "lasso C" ::

		max_iter: 5000
		multi_class: auto
		penalty: l1
		class_weight: UnorderedDiscrete([None, 'balanced'])
		solver: liblinear
		C: Log({a_min},{a_max},{width})

	2. "lasso C extra" ::

		max_iter: Array((1,), [ArctanBound(a_max=[10000], a_min=[1000], name=At(1000,10000), shape=(1,))])
		multi_class: auto
		penalty: l1
		class_weight: UnorderedDiscrete([None, 'balanced'])
		solver: liblinear
		C: Log({a_min},{a_max},{width})
		tol: Log({a_min},{a_max},{width})


"light gbm classifier"
**********************

  Base Class Documenation: :class:`lightgbm.LGBMClassifier`

  Param Distributions

	0. "base lgbm" ::

		silent: True

	1. "lgbm classifier dist1" ::

		silent: True
		boosting_type: UnorderedDiscrete(['gbdt', 'dart', 'goss'])
		n_estimators: Array((1,), [ArctanBound(a_max=[500], a_min=[3], name=At(3,500), shape=(1,))])
		num_leaves: Array((1,), [ArctanBound(a_max=[80], a_min=[6], name=At(6,80), shape=(1,))])
		min_child_samples: Array((1,), [ArctanBound(a_max=[500], a_min=[10], name=At(10,500), shape=(1,))])
		min_child_weight: Log({a_min},{a_max},{width})
		subsample: Array((1,), [ArctanBound(a_max=[0.95], a_min=[0.3], name=At(0.3,0.95), shape=(1,))])
		colsample_bytree: Array((1,), [ArctanBound(a_max=[0.95], a_min=[0.3], name=At(0.3,0.95), shape=(1,))])
		reg_alpha: Array((1,), [ArctanBound(a_max=[1], a_min=[-2], name=At(-2,1), shape=(1,)), Exponentiate(base=10, coeff=-1, name=Ex(10,-1))])
		reg_lambda: Array((1,), [ArctanBound(a_max=[1], a_min=[-2], name=At(-2,1), shape=(1,)), Exponentiate(base=10, coeff=-1, name=Ex(10,-1))])
		class_weight: UnorderedDiscrete([None, 'balanced'])

	2. "lgbm classifier dist2" ::

		silent: True
		lambda_l2: 0.001
		boosting_type: UnorderedDiscrete(['gbdt', 'dart'])
		min_child_samples: OrderedDiscrete([1, 5, 7, 10, 15, 20, 35, 50, 100, 200, 500, 1000])
		num_leaves: OrderedDiscrete([2, 4, 7, 10, 15, 20, 25, 30, 35, 40, 50, 65, 80, 100, 125, 150, 200, 250])
		colsample_bytree: OrderedDiscrete([0.7, 0.9, 1.0])
		subsample: Array((1,), [ArctanBound(a_max=[1], a_min=[0.3], name=At(0.3,1), shape=(1,))])
		learning_rate: OrderedDiscrete([0.01, 0.05, 0.1])
		n_estimators: OrderedDiscrete([5, 20, 35, 50, 75, 100, 150, 200, 350, 500, 750, 1000])
		class_weight: UnorderedDiscrete([None, 'balanced'])


"linear svm classifier"
***********************

  Base Class Documenation: :class:`sklearn.svm.LinearSVC`

  Param Distributions

	0. "base linear svc" ::

		max_iter: 5000

	1. "linear svc dist" ::

		max_iter: 5000
		C: Log({a_min},{a_max},{width})
		class_weight: UnorderedDiscrete([None, 'balanced'])


"logistic"
**********

  Base Class Documenation: :class:`sklearn.linear_model.LogisticRegression`

  Param Distributions

	0. "base logistic" ::

		max_iter: 5000
		multi_class: auto
		penalty: none
		class_weight: None
		solver: lbfgs


"mlp classifier"
****************

  Base Class Documenation: :class:`sklearn.neural_network.MLPClassifier`

  Param Distributions

	0. "base mlp" ::

		defaults only

	1. "mlp dist 1 layer" ::

		hidden_layer_sizes: Array((1,), [ArctanBound(a_max=[200], a_min=[2], name=At(2,200), shape=(1,))])
		activation: UnorderedDiscrete(['identity', 'logistic', 'tanh', 'relu'])
		alpha: Log({a_min},{a_max},{width})
		batch_size: Array((1,), [ArctanBound(a_max=[200], a_min=[2], name=At(2,200), shape=(1,))])
		learning_rate: UnorderedDiscrete(['constant', 'invscaling', 'adaptive'])
		learning_rate_init: Log({a_min},{a_max},{width})
		max_iter: Array((1,), [ArctanBound(a_max=[500], a_min=[100], name=At(100,500), shape=(1,))])
		beta_1: Array((1,), [ArctanBound(a_max=[0.95], a_min=[0.1], name=At(0.1,0.95), shape=(1,))])
		beta_2: Array((1,), [ArctanBound(a_max=[0.95], a_min=[0.1], name=At(0.1,0.95), shape=(1,))])

	2. "mlp dist es 1 layer" ::

		hidden_layer_sizes: Array((1,), [ArctanBound(a_max=[200], a_min=[2], name=At(2,200), shape=(1,))])
		activation: UnorderedDiscrete(['identity', 'logistic', 'tanh', 'relu'])
		alpha: Log({a_min},{a_max},{width})
		batch_size: Array((1,), [ArctanBound(a_max=[200], a_min=[2], name=At(2,200), shape=(1,))])
		learning_rate: UnorderedDiscrete(['constant', 'invscaling', 'adaptive'])
		learning_rate_init: Log({a_min},{a_max},{width})
		max_iter: Array((1,), [ArctanBound(a_max=[500], a_min=[100], name=At(100,500), shape=(1,))])
		beta_1: Array((1,), [ArctanBound(a_max=[0.95], a_min=[0.1], name=At(0.1,0.95), shape=(1,))])
		beta_2: Array((1,), [ArctanBound(a_max=[0.95], a_min=[0.1], name=At(0.1,0.95), shape=(1,))])
		early_stopping: True
		n_iter_no_change: Array((1,), [ArctanBound(a_max=[50], a_min=[5], name=At(5,50), shape=(1,))])


"random forest classifier"
**************************

  Base Class Documenation: :class:`sklearn.ensemble.RandomForestClassifier`

  Param Distributions

	0. "base rf" ::

		n_estimators: 100

	1. "rf classifier dist" ::

		n_estimators: Array((1,), [ArctanBound(a_max=[500], a_min=[3], name=At(3,500), shape=(1,))])
		max_depth: Array((1,), [ArctanBound(a_max=[200], a_min=[2], name=At(2,200), shape=(1,))])
		max_features: Array((1,), [ArctanBound(a_max=[1], a_min=[0], name=At(0,1), shape=(1,))])
		min_samples_split: Array((1,), [ArctanBound(a_max=[1], a_min=[0], name=At(0,1), shape=(1,))])
		bootstrap: True
		class_weight: UnorderedDiscrete([None, 'balanced'])


"ridge logistic"
****************

  Base Class Documenation: :class:`sklearn.linear_model.LogisticRegression`

  Param Distributions

	0. "base ridge" ::

		max_iter: 5000
		penalty: l2
		solver: saga

	1. "ridge C" ::

		max_iter: 5000
		solver: saga
		C: Log({a_min},{a_max},{width})
		class_weight: UnorderedDiscrete([None, 'balanced'])

	2. "ridge C extra" ::

		max_iter: Array((1,), [ArctanBound(a_max=[10000], a_min=[1000], name=At(1000,10000), shape=(1,))])
		solver: saga
		C: Log({a_min},{a_max},{width})
		class_weight: UnorderedDiscrete([None, 'balanced'])
		tol: Log({a_min},{a_max},{width})


"sgd classifier"
****************

  Base Class Documenation: :class:`sklearn.linear_model.SGDClassifier`

  Param Distributions

	0. "base sgd" ::

		loss: hinge

	1. "sgd classifier" ::

		loss: UnorderedDiscrete(['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'])
		penalty: UnorderedDiscrete(['l2', 'l1', 'elasticnet'])
		alpha: Log({a_min},{a_max},{width})
		l1_ratio: Array((1,), [ArctanBound(a_max=[1], a_min=[0], name=At(0,1), shape=(1,))])
		max_iter: 5000
		learning_rate: UnorderedDiscrete(['optimal', 'invscaling', 'adaptive', 'constant'])
		eta0: Log({a_min},{a_max},{width})
		power_t: Array((1,), [ArctanBound(a_max=[0.9], a_min=[0.1], name=At(0.1,0.9), shape=(1,))])
		early_stopping: UnorderedDiscrete([False, True])
		validation_fraction: Array((1,), [ArctanBound(a_max=[0.5], a_min=[0.05], name=At(0.05,0.5), shape=(1,))])
		n_iter_no_change: Array((1,), [ArctanBound(a_max=[20], a_min=[2], name=At(2,20), shape=(1,))])
		class_weight: UnorderedDiscrete([None, 'balanced'])


"svm classifier"
****************

  Base Class Documenation: :class:`sklearn.svm.SVC`

  Param Distributions

	0. "base svm classifier" ::

		kernel: rbf
		gamma: scale
		probability: True

	1. "svm classifier dist" ::

		kernel: rbf
		gamma: Log({a_min},{a_max},{width})
		C: Log({a_min},{a_max},{width})
		probability: True
		class_weight: UnorderedDiscrete([None, 'balanced'])


"xgb classifier"
****************

  Base Class Documenation: :class:`xgboost.XGBClassifier`

  Param Distributions

	0. "base xgb classifier" ::

		verbosity: 0
		objective: binary:logistic

	1. "xgb classifier dist1" ::

		verbosity: 0
		objective: binary:logistic
		n_estimators: Array((1,), [ArctanBound(a_max=[500], a_min=[3], name=At(3,500), shape=(1,))])
		min_child_weight: Log({a_min},{a_max},{width})
		subsample: Array((1,), [ArctanBound(a_max=[0.95], a_min=[0.3], name=At(0.3,0.95), shape=(1,))])
		colsample_bytree: Array((1,), [ArctanBound(a_max=[0.95], a_min=[0.3], name=At(0.3,0.95), shape=(1,))])
		reg_alpha: Array((1,), [ArctanBound(a_max=[1], a_min=[-2], name=At(-2,1), shape=(1,)), Exponentiate(base=10, coeff=-1, name=Ex(10,-1))])
		reg_lambda: Array((1,), [ArctanBound(a_max=[1], a_min=[-2], name=At(-2,1), shape=(1,)), Exponentiate(base=10, coeff=-1, name=Ex(10,-1))])

	2. "xgb classifier dist2" ::

		verbosity: 0
		objective: binary:logistic
		max_depth: Array((1,), [ArctanBound(a_max=[50], a_min=[2], name=At(2,50), shape=(1,))])
		learning_rate: Array((1,), [ArctanBound(a_max=[0.5], a_min=[0.01], name=At(0.01,0.5), shape=(1,))])
		n_estimators: Array((1,), [ArctanBound(a_max=[500], a_min=[3], name=At(3,500), shape=(1,))])
		min_child_weight: OrderedDiscrete([1, 5, 10, 50])
		subsample: Array((1,), [ArctanBound(a_max=[1], a_min=[0.5], name=At(0.5,1), shape=(1,))])
		colsample_bytree: Array((1,), [ArctanBound(a_max=[0.95], a_min=[0.4], name=At(0.4,0.95), shape=(1,))])

	3. "xgb classifier dist3" ::

		verbosity: 0
		objective: binary:logistic
		learning_rare: Array((1,), [ArctanBound(a_max=[0.3], a_min=[0.005], name=At(0.005,0.3), shape=(1,))])
		min_child_weight: Array((1,), [ArctanBound(a_max=[10], a_min=[0.5], name=At(0.5,10), shape=(1,))])
		max_depth: Array((1,), [ArctanBound(a_max=[10], a_min=[3], name=At(3,10), shape=(1,))])
		subsample: Array((1,), [ArctanBound(a_max=[1], a_min=[0.5], name=At(0.5,1), shape=(1,))])
		colsample_bytree: Array((1,), [ArctanBound(a_max=[1], a_min=[0.5], name=At(0.5,1), shape=(1,))])
		reg_alpha: Log({a_min},{a_max},{width})



regression
==========
"dt regressor"
**************

  Base Class Documenation: :class:`sklearn.tree.DecisionTreeRegressor`

  Param Distributions

	0. "base dt" ::

		defaults only

	1. "dt dist" ::

		max_depth: Array((1,), [ArctanBound(a_max=[30], a_min=[1], name=At(1,30), shape=(1,))])
		min_samples_split: Array((1,), [ArctanBound(a_max=[50], a_min=[2], name=At(2,50), shape=(1,))])


"elastic net regressor"
***********************

  Base Class Documenation: :class:`sklearn.linear_model.ElasticNet`

  Param Distributions

	0. "base elastic net" ::

		max_iter: 5000

	1. "elastic regression" ::

		max_iter: 5000
		alpha: Log({a_min},{a_max},{width})
		l1_ratio: Array((1,), [ArctanBound(a_max=[1], a_min=[0.01], name=At(0.01,1), shape=(1,))])

	2. "elastic regression extra" ::

		max_iter: Array((1,), [ArctanBound(a_max=[10000], a_min=[1000], name=At(1000,10000), shape=(1,))])
		alpha: Log({a_min},{a_max},{width})
		l1_ratio: Array((1,), [ArctanBound(a_max=[1], a_min=[0.01], name=At(0.01,1), shape=(1,))])
		tol: Log({a_min},{a_max},{width})


"gp regressor"
**************

  Base Class Documenation: :class:`sklearn.gaussian_process.GaussianProcessRegressor`

  Param Distributions

	0. "base gp regressor" ::

		n_restarts_optimizer: 5
		normalize_y: True


"knn regressor"
***************

  Base Class Documenation: :class:`sklearn.neighbors.KNeighborsRegressor`

  Param Distributions

	0. "base knn" ::

		n_neighbors: 5

	1. "knn dist" ::

		weights: UnorderedDiscrete(['uniform', 'distance'])
		n_neighbors: Array((1,), [ArctanBound(a_max=[25], a_min=[2], name=At(2,25), shape=(1,))])


"lasso regressor"
*****************

  Base Class Documenation: :class:`sklearn.linear_model.Lasso`

  Param Distributions

	0. "base lasso regressor" ::

		max_iter: 5000

	1. "lasso regressor dist" ::

		max_iter: 5000
		alpha: Log({a_min},{a_max},{width})


"light gbm regressor"
*********************

  Base Class Documenation: :class:`lightgbm.LGBMRegressor`

  Param Distributions

	0. "base lgbm" ::

		silent: True

	1. "lgbm dist1" ::

		silent: True
		boosting_type: UnorderedDiscrete(['gbdt', 'dart', 'goss'])
		n_estimators: Array((1,), [ArctanBound(a_max=[500], a_min=[3], name=At(3,500), shape=(1,))])
		num_leaves: Array((1,), [ArctanBound(a_max=[80], a_min=[6], name=At(6,80), shape=(1,))])
		min_child_samples: Array((1,), [ArctanBound(a_max=[500], a_min=[10], name=At(10,500), shape=(1,))])
		min_child_weight: Log({a_min},{a_max},{width})
		subsample: Array((1,), [ArctanBound(a_max=[0.95], a_min=[0.3], name=At(0.3,0.95), shape=(1,))])
		colsample_bytree: Array((1,), [ArctanBound(a_max=[0.95], a_min=[0.3], name=At(0.3,0.95), shape=(1,))])
		reg_alpha: Array((1,), [ArctanBound(a_max=[1], a_min=[-2], name=At(-2,1), shape=(1,)), Exponentiate(base=10, coeff=-1, name=Ex(10,-1))])
		reg_lambda: Array((1,), [ArctanBound(a_max=[1], a_min=[-2], name=At(-2,1), shape=(1,)), Exponentiate(base=10, coeff=-1, name=Ex(10,-1))])

	2. "lgbm dist2" ::

		silent: True
		lambda_l2: 0.001
		boosting_type: UnorderedDiscrete(['gbdt', 'dart'])
		min_child_samples: OrderedDiscrete([1, 5, 7, 10, 15, 20, 35, 50, 100, 200, 500, 1000])
		num_leaves: OrderedDiscrete([2, 4, 7, 10, 15, 20, 25, 30, 35, 40, 50, 65, 80, 100, 125, 150, 200, 250])
		colsample_bytree: OrderedDiscrete([0.7, 0.9, 1.0])
		subsample: Array((1,), [ArctanBound(a_max=[1], a_min=[0.3], name=At(0.3,1), shape=(1,))])
		learning_rate: OrderedDiscrete([0.01, 0.05, 0.1])
		n_estimators: OrderedDiscrete([5, 20, 35, 50, 75, 100, 150, 200, 350, 500, 750, 1000])


"linear regressor"
******************

  Base Class Documenation: :class:`sklearn.linear_model.LinearRegression`

  Param Distributions

	0. "base linear" ::

		fit_intercept: True


"linear svm regressor"
**********************

  Base Class Documenation: :class:`sklearn.svm.LinearSVR`

  Param Distributions

	0. "base linear svr" ::

		loss: epsilon_insensitive
		max_iter: 5000

	1. "linear svr dist" ::

		loss: epsilon_insensitive
		max_iter: 5000
		C: Log({a_min},{a_max},{width})


"mlp regressor"
***************

  Base Class Documenation: :class:`sklearn.neural_network.MLPRegressor`

  Param Distributions

	0. "base mlp" ::

		defaults only

	1. "mlp dist 1 layer" ::

		hidden_layer_sizes: Array((1,), [ArctanBound(a_max=[200], a_min=[2], name=At(2,200), shape=(1,))])
		activation: UnorderedDiscrete(['identity', 'logistic', 'tanh', 'relu'])
		alpha: Log({a_min},{a_max},{width})
		batch_size: Array((1,), [ArctanBound(a_max=[200], a_min=[2], name=At(2,200), shape=(1,))])
		learning_rate: UnorderedDiscrete(['constant', 'invscaling', 'adaptive'])
		learning_rate_init: Log({a_min},{a_max},{width})
		max_iter: Array((1,), [ArctanBound(a_max=[500], a_min=[100], name=At(100,500), shape=(1,))])
		beta_1: Array((1,), [ArctanBound(a_max=[0.95], a_min=[0.1], name=At(0.1,0.95), shape=(1,))])
		beta_2: Array((1,), [ArctanBound(a_max=[0.95], a_min=[0.1], name=At(0.1,0.95), shape=(1,))])

	2. "mlp dist es 1 layer" ::

		hidden_layer_sizes: Array((1,), [ArctanBound(a_max=[200], a_min=[2], name=At(2,200), shape=(1,))])
		activation: UnorderedDiscrete(['identity', 'logistic', 'tanh', 'relu'])
		alpha: Log({a_min},{a_max},{width})
		batch_size: Array((1,), [ArctanBound(a_max=[200], a_min=[2], name=At(2,200), shape=(1,))])
		learning_rate: UnorderedDiscrete(['constant', 'invscaling', 'adaptive'])
		learning_rate_init: Log({a_min},{a_max},{width})
		max_iter: Array((1,), [ArctanBound(a_max=[500], a_min=[100], name=At(100,500), shape=(1,))])
		beta_1: Array((1,), [ArctanBound(a_max=[0.95], a_min=[0.1], name=At(0.1,0.95), shape=(1,))])
		beta_2: Array((1,), [ArctanBound(a_max=[0.95], a_min=[0.1], name=At(0.1,0.95), shape=(1,))])
		early_stopping: True
		n_iter_no_change: Array((1,), [ArctanBound(a_max=[50], a_min=[5], name=At(5,50), shape=(1,))])


"random forest regressor"
*************************

  Base Class Documenation: :class:`sklearn.ensemble.RandomForestRegressor`

  Param Distributions

	0. "base rf" ::

		n_estimators: 100

	1. "rf dist" ::

		n_estimators: Array((1,), [ArctanBound(a_max=[500], a_min=[3], name=At(3,500), shape=(1,))])
		max_depth: Array((1,), [ArctanBound(a_max=[200], a_min=[2], name=At(2,200), shape=(1,))])
		max_features: Array((1,), [ArctanBound(a_max=[1], a_min=[0], name=At(0,1), shape=(1,))])
		min_samples_split: Array((1,), [ArctanBound(a_max=[1], a_min=[0], name=At(0,1), shape=(1,))])
		bootstrap: True


"ridge regressor"
*****************

  Base Class Documenation: :class:`sklearn.linear_model.Ridge`

  Param Distributions

	0. "base ridge regressor" ::

		max_iter: 5000
		solver: lsqr

	1. "ridge regressor dist" ::

		max_iter: 5000
		solver: lsqr
		alpha: Log({a_min},{a_max},{width})


"svm regressor"
***************

  Base Class Documenation: :class:`sklearn.svm.SVR`

  Param Distributions

	0. "base svm" ::

		kernel: rbf
		gamma: scale

	1. "svm dist" ::

		kernel: rbf
		gamma: Log({a_min},{a_max},{width})
		C: Log({a_min},{a_max},{width})


"xgb regressor"
***************

  Base Class Documenation: :class:`xgboost.XGBRegressor`

  Param Distributions

	0. "base xgb" ::

		verbosity: 0
		objective: reg:squarederror

	1. "xgb dist1" ::

		verbosity: 0
		objective: reg:squarederror
		n_estimators: Array((1,), [ArctanBound(a_max=[500], a_min=[3], name=At(3,500), shape=(1,))])
		min_child_weight: Log({a_min},{a_max},{width})
		subsample: Array((1,), [ArctanBound(a_max=[0.95], a_min=[0.3], name=At(0.3,0.95), shape=(1,))])
		colsample_bytree: Array((1,), [ArctanBound(a_max=[0.95], a_min=[0.3], name=At(0.3,0.95), shape=(1,))])
		reg_alpha: Array((1,), [ArctanBound(a_max=[1], a_min=[-2], name=At(-2,1), shape=(1,)), Exponentiate(base=10, coeff=-1, name=Ex(10,-1))])
		reg_lambda: Array((1,), [ArctanBound(a_max=[1], a_min=[-2], name=At(-2,1), shape=(1,)), Exponentiate(base=10, coeff=-1, name=Ex(10,-1))])

	2. "xgb dist2" ::

		verbosity: 0
		objective: reg:squarederror
		max_depth: Array((1,), [ArctanBound(a_max=[50], a_min=[2], name=At(2,50), shape=(1,))])
		learning_rate: Array((1,), [ArctanBound(a_max=[0.5], a_min=[0.01], name=At(0.01,0.5), shape=(1,))])
		n_estimators: Array((1,), [ArctanBound(a_max=[500], a_min=[3], name=At(3,500), shape=(1,))])
		min_child_weight: OrderedDiscrete([1, 5, 10, 50])
		subsample: Array((1,), [ArctanBound(a_max=[1], a_min=[0.5], name=At(0.5,1), shape=(1,))])
		colsample_bytree: Array((1,), [ArctanBound(a_max=[0.95], a_min=[0.4], name=At(0.4,0.95), shape=(1,))])

	3. "xgb dist3" ::

		verbosity: 0
		objective: reg:squarederror
		learning_rare: Array((1,), [ArctanBound(a_max=[0.3], a_min=[0.005], name=At(0.005,0.3), shape=(1,))])
		min_child_weight: Array((1,), [ArctanBound(a_max=[10], a_min=[0.5], name=At(0.5,10), shape=(1,))])
		max_depth: Array((1,), [ArctanBound(a_max=[10], a_min=[3], name=At(3,10), shape=(1,))])
		subsample: Array((1,), [ArctanBound(a_max=[1], a_min=[0.5], name=At(0.5,1), shape=(1,))])
		colsample_bytree: Array((1,), [ArctanBound(a_max=[1], a_min=[0.5], name=At(0.5,1), shape=(1,))])
		reg_alpha: Log({a_min},{a_max},{width})



categorical
===========
"dt classifier"
***************

  Base Class Documenation: :class:`sklearn.tree.DecisionTreeClassifier`

  Param Distributions

	0. "base dt" ::

		defaults only

	1. "dt classifier dist" ::

		max_depth: Array((1,), [ArctanBound(a_max=[30], a_min=[1], name=At(1,30), shape=(1,))])
		min_samples_split: Array((1,), [ArctanBound(a_max=[50], a_min=[2], name=At(2,50), shape=(1,))])
		class_weight: UnorderedDiscrete([None, 'balanced'])


"elastic net logistic"
**********************

  Base Class Documenation: :class:`sklearn.linear_model.LogisticRegression`

  Param Distributions

	0. "base elastic" ::

		max_iter: 5000
		multi_class: auto
		penalty: elasticnet
		class_weight: None
		solver: saga
		l1_ratio: 0.5

	1. "elastic classifier" ::

		max_iter: 5000
		multi_class: auto
		penalty: elasticnet
		class_weight: UnorderedDiscrete([None, 'balanced'])
		solver: saga
		l1_ratio: Array((1,), [ArctanBound(a_max=[1], a_min=[0.01], name=At(0.01,1), shape=(1,))])
		C: Log({a_min},{a_max},{width})

	2. "elastic classifier extra" ::

		max_iter: Array((1,), [ArctanBound(a_max=[10000], a_min=[1000], name=At(1000,10000), shape=(1,))])
		multi_class: auto
		penalty: elasticnet
		class_weight: UnorderedDiscrete([None, 'balanced'])
		solver: saga
		l1_ratio: Array((1,), [ArctanBound(a_max=[1], a_min=[0.01], name=At(0.01,1), shape=(1,))])
		C: Log({a_min},{a_max},{width})
		tol: Log({a_min},{a_max},{width})


"gaussian nb"
*************

  Base Class Documenation: :class:`sklearn.naive_bayes.GaussianNB`

  Param Distributions

	0. "base gnb" ::

		var_smoothing: 1e-09


"gp classifier"
***************

  Base Class Documenation: :class:`sklearn.gaussian_process.GaussianProcessClassifier`

  Param Distributions

	0. "base gp classifier" ::

		n_restarts_optimizer: 5


"knn classifier"
****************

  Base Class Documenation: :class:`sklearn.neighbors.KNeighborsClassifier`

  Param Distributions

	0. "base knn" ::

		n_neighbors: 5

	1. "knn dist" ::

		weights: UnorderedDiscrete(['uniform', 'distance'])
		n_neighbors: Array((1,), [ArctanBound(a_max=[25], a_min=[2], name=At(2,25), shape=(1,))])


"lasso logistic"
****************

  Base Class Documenation: :class:`sklearn.linear_model.LogisticRegression`

  Param Distributions

	0. "base lasso" ::

		max_iter: 5000
		multi_class: auto
		penalty: l1
		class_weight: None
		solver: liblinear

	1. "lasso C" ::

		max_iter: 5000
		multi_class: auto
		penalty: l1
		class_weight: UnorderedDiscrete([None, 'balanced'])
		solver: liblinear
		C: Log({a_min},{a_max},{width})

	2. "lasso C extra" ::

		max_iter: Array((1,), [ArctanBound(a_max=[10000], a_min=[1000], name=At(1000,10000), shape=(1,))])
		multi_class: auto
		penalty: l1
		class_weight: UnorderedDiscrete([None, 'balanced'])
		solver: liblinear
		C: Log({a_min},{a_max},{width})
		tol: Log({a_min},{a_max},{width})


"light gbm classifier"
**********************

  Base Class Documenation: :class:`lightgbm.LGBMClassifier`

  Param Distributions

	0. "base lgbm" ::

		silent: True

	1. "lgbm classifier dist1" ::

		silent: True
		boosting_type: UnorderedDiscrete(['gbdt', 'dart', 'goss'])
		n_estimators: Array((1,), [ArctanBound(a_max=[500], a_min=[3], name=At(3,500), shape=(1,))])
		num_leaves: Array((1,), [ArctanBound(a_max=[80], a_min=[6], name=At(6,80), shape=(1,))])
		min_child_samples: Array((1,), [ArctanBound(a_max=[500], a_min=[10], name=At(10,500), shape=(1,))])
		min_child_weight: Log({a_min},{a_max},{width})
		subsample: Array((1,), [ArctanBound(a_max=[0.95], a_min=[0.3], name=At(0.3,0.95), shape=(1,))])
		colsample_bytree: Array((1,), [ArctanBound(a_max=[0.95], a_min=[0.3], name=At(0.3,0.95), shape=(1,))])
		reg_alpha: Array((1,), [ArctanBound(a_max=[1], a_min=[-2], name=At(-2,1), shape=(1,)), Exponentiate(base=10, coeff=-1, name=Ex(10,-1))])
		reg_lambda: Array((1,), [ArctanBound(a_max=[1], a_min=[-2], name=At(-2,1), shape=(1,)), Exponentiate(base=10, coeff=-1, name=Ex(10,-1))])
		class_weight: UnorderedDiscrete([None, 'balanced'])

	2. "lgbm classifier dist2" ::

		silent: True
		lambda_l2: 0.001
		boosting_type: UnorderedDiscrete(['gbdt', 'dart'])
		min_child_samples: OrderedDiscrete([1, 5, 7, 10, 15, 20, 35, 50, 100, 200, 500, 1000])
		num_leaves: OrderedDiscrete([2, 4, 7, 10, 15, 20, 25, 30, 35, 40, 50, 65, 80, 100, 125, 150, 200, 250])
		colsample_bytree: OrderedDiscrete([0.7, 0.9, 1.0])
		subsample: Array((1,), [ArctanBound(a_max=[1], a_min=[0.3], name=At(0.3,1), shape=(1,))])
		learning_rate: OrderedDiscrete([0.01, 0.05, 0.1])
		n_estimators: OrderedDiscrete([5, 20, 35, 50, 75, 100, 150, 200, 350, 500, 750, 1000])
		class_weight: UnorderedDiscrete([None, 'balanced'])


"linear svm classifier"
***********************

  Base Class Documenation: :class:`sklearn.svm.LinearSVC`

  Param Distributions

	0. "base linear svc" ::

		max_iter: 5000

	1. "linear svc dist" ::

		max_iter: 5000
		C: Log({a_min},{a_max},{width})
		class_weight: UnorderedDiscrete([None, 'balanced'])


"logistic"
**********

  Base Class Documenation: :class:`sklearn.linear_model.LogisticRegression`

  Param Distributions

	0. "base logistic" ::

		max_iter: 5000
		multi_class: auto
		penalty: none
		class_weight: None
		solver: lbfgs


"mlp classifier"
****************

  Base Class Documenation: :class:`sklearn.neural_network.MLPClassifier`

  Param Distributions

	0. "base mlp" ::

		defaults only

	1. "mlp dist 1 layer" ::

		hidden_layer_sizes: Array((1,), [ArctanBound(a_max=[200], a_min=[2], name=At(2,200), shape=(1,))])
		activation: UnorderedDiscrete(['identity', 'logistic', 'tanh', 'relu'])
		alpha: Log({a_min},{a_max},{width})
		batch_size: Array((1,), [ArctanBound(a_max=[200], a_min=[2], name=At(2,200), shape=(1,))])
		learning_rate: UnorderedDiscrete(['constant', 'invscaling', 'adaptive'])
		learning_rate_init: Log({a_min},{a_max},{width})
		max_iter: Array((1,), [ArctanBound(a_max=[500], a_min=[100], name=At(100,500), shape=(1,))])
		beta_1: Array((1,), [ArctanBound(a_max=[0.95], a_min=[0.1], name=At(0.1,0.95), shape=(1,))])
		beta_2: Array((1,), [ArctanBound(a_max=[0.95], a_min=[0.1], name=At(0.1,0.95), shape=(1,))])

	2. "mlp dist es 1 layer" ::

		hidden_layer_sizes: Array((1,), [ArctanBound(a_max=[200], a_min=[2], name=At(2,200), shape=(1,))])
		activation: UnorderedDiscrete(['identity', 'logistic', 'tanh', 'relu'])
		alpha: Log({a_min},{a_max},{width})
		batch_size: Array((1,), [ArctanBound(a_max=[200], a_min=[2], name=At(2,200), shape=(1,))])
		learning_rate: UnorderedDiscrete(['constant', 'invscaling', 'adaptive'])
		learning_rate_init: Log({a_min},{a_max},{width})
		max_iter: Array((1,), [ArctanBound(a_max=[500], a_min=[100], name=At(100,500), shape=(1,))])
		beta_1: Array((1,), [ArctanBound(a_max=[0.95], a_min=[0.1], name=At(0.1,0.95), shape=(1,))])
		beta_2: Array((1,), [ArctanBound(a_max=[0.95], a_min=[0.1], name=At(0.1,0.95), shape=(1,))])
		early_stopping: True
		n_iter_no_change: Array((1,), [ArctanBound(a_max=[50], a_min=[5], name=At(5,50), shape=(1,))])


"random forest classifier"
**************************

  Base Class Documenation: :class:`sklearn.ensemble.RandomForestClassifier`

  Param Distributions

	0. "base rf" ::

		n_estimators: 100

	1. "rf classifier dist" ::

		n_estimators: Array((1,), [ArctanBound(a_max=[500], a_min=[3], name=At(3,500), shape=(1,))])
		max_depth: Array((1,), [ArctanBound(a_max=[200], a_min=[2], name=At(2,200), shape=(1,))])
		max_features: Array((1,), [ArctanBound(a_max=[1], a_min=[0], name=At(0,1), shape=(1,))])
		min_samples_split: Array((1,), [ArctanBound(a_max=[1], a_min=[0], name=At(0,1), shape=(1,))])
		bootstrap: True
		class_weight: UnorderedDiscrete([None, 'balanced'])


"ridge logistic"
****************

  Base Class Documenation: :class:`sklearn.linear_model.LogisticRegression`

  Param Distributions

	0. "base ridge" ::

		max_iter: 5000
		penalty: l2
		solver: saga

	1. "ridge C" ::

		max_iter: 5000
		solver: saga
		C: Log({a_min},{a_max},{width})
		class_weight: UnorderedDiscrete([None, 'balanced'])

	2. "ridge C extra" ::

		max_iter: Array((1,), [ArctanBound(a_max=[10000], a_min=[1000], name=At(1000,10000), shape=(1,))])
		solver: saga
		C: Log({a_min},{a_max},{width})
		class_weight: UnorderedDiscrete([None, 'balanced'])
		tol: Log({a_min},{a_max},{width})


"sgd classifier"
****************

  Base Class Documenation: :class:`sklearn.linear_model.SGDClassifier`

  Param Distributions

	0. "base sgd" ::

		loss: hinge

	1. "sgd classifier" ::

		loss: UnorderedDiscrete(['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'])
		penalty: UnorderedDiscrete(['l2', 'l1', 'elasticnet'])
		alpha: Log({a_min},{a_max},{width})
		l1_ratio: Array((1,), [ArctanBound(a_max=[1], a_min=[0], name=At(0,1), shape=(1,))])
		max_iter: 5000
		learning_rate: UnorderedDiscrete(['optimal', 'invscaling', 'adaptive', 'constant'])
		eta0: Log({a_min},{a_max},{width})
		power_t: Array((1,), [ArctanBound(a_max=[0.9], a_min=[0.1], name=At(0.1,0.9), shape=(1,))])
		early_stopping: UnorderedDiscrete([False, True])
		validation_fraction: Array((1,), [ArctanBound(a_max=[0.5], a_min=[0.05], name=At(0.05,0.5), shape=(1,))])
		n_iter_no_change: Array((1,), [ArctanBound(a_max=[20], a_min=[2], name=At(2,20), shape=(1,))])
		class_weight: UnorderedDiscrete([None, 'balanced'])


"svm classifier"
****************

  Base Class Documenation: :class:`sklearn.svm.SVC`

  Param Distributions

	0. "base svm classifier" ::

		kernel: rbf
		gamma: scale
		probability: True

	1. "svm classifier dist" ::

		kernel: rbf
		gamma: Log({a_min},{a_max},{width})
		C: Log({a_min},{a_max},{width})
		probability: True
		class_weight: UnorderedDiscrete([None, 'balanced'])



multilabel
==========
"dt classifier"
***************

  Base Class Documenation: :class:`sklearn.tree.DecisionTreeClassifier`

  Param Distributions

	0. "base dt" ::

		defaults only

	1. "dt classifier dist" ::

		max_depth: Array((1,), [ArctanBound(a_max=[30], a_min=[1], name=At(1,30), shape=(1,))])
		min_samples_split: Array((1,), [ArctanBound(a_max=[50], a_min=[2], name=At(2,50), shape=(1,))])
		class_weight: UnorderedDiscrete([None, 'balanced'])


"knn classifier"
****************

  Base Class Documenation: :class:`sklearn.neighbors.KNeighborsClassifier`

  Param Distributions

	0. "base knn" ::

		n_neighbors: 5

	1. "knn dist" ::

		weights: UnorderedDiscrete(['uniform', 'distance'])
		n_neighbors: Array((1,), [ArctanBound(a_max=[25], a_min=[2], name=At(2,25), shape=(1,))])


"mlp classifier"
****************

  Base Class Documenation: :class:`sklearn.neural_network.MLPClassifier`

  Param Distributions

	0. "base mlp" ::

		defaults only

	1. "mlp dist 1 layer" ::

		hidden_layer_sizes: Array((1,), [ArctanBound(a_max=[200], a_min=[2], name=At(2,200), shape=(1,))])
		activation: UnorderedDiscrete(['identity', 'logistic', 'tanh', 'relu'])
		alpha: Log({a_min},{a_max},{width})
		batch_size: Array((1,), [ArctanBound(a_max=[200], a_min=[2], name=At(2,200), shape=(1,))])
		learning_rate: UnorderedDiscrete(['constant', 'invscaling', 'adaptive'])
		learning_rate_init: Log({a_min},{a_max},{width})
		max_iter: Array((1,), [ArctanBound(a_max=[500], a_min=[100], name=At(100,500), shape=(1,))])
		beta_1: Array((1,), [ArctanBound(a_max=[0.95], a_min=[0.1], name=At(0.1,0.95), shape=(1,))])
		beta_2: Array((1,), [ArctanBound(a_max=[0.95], a_min=[0.1], name=At(0.1,0.95), shape=(1,))])

	2. "mlp dist es 1 layer" ::

		hidden_layer_sizes: Array((1,), [ArctanBound(a_max=[200], a_min=[2], name=At(2,200), shape=(1,))])
		activation: UnorderedDiscrete(['identity', 'logistic', 'tanh', 'relu'])
		alpha: Log({a_min},{a_max},{width})
		batch_size: Array((1,), [ArctanBound(a_max=[200], a_min=[2], name=At(2,200), shape=(1,))])
		learning_rate: UnorderedDiscrete(['constant', 'invscaling', 'adaptive'])
		learning_rate_init: Log({a_min},{a_max},{width})
		max_iter: Array((1,), [ArctanBound(a_max=[500], a_min=[100], name=At(100,500), shape=(1,))])
		beta_1: Array((1,), [ArctanBound(a_max=[0.95], a_min=[0.1], name=At(0.1,0.95), shape=(1,))])
		beta_2: Array((1,), [ArctanBound(a_max=[0.95], a_min=[0.1], name=At(0.1,0.95), shape=(1,))])
		early_stopping: True
		n_iter_no_change: Array((1,), [ArctanBound(a_max=[50], a_min=[5], name=At(5,50), shape=(1,))])


"random forest classifier"
**************************

  Base Class Documenation: :class:`sklearn.ensemble.RandomForestClassifier`

  Param Distributions

	0. "base rf" ::

		n_estimators: 100

	1. "rf classifier dist" ::

		n_estimators: Array((1,), [ArctanBound(a_max=[500], a_min=[3], name=At(3,500), shape=(1,))])
		max_depth: Array((1,), [ArctanBound(a_max=[200], a_min=[2], name=At(2,200), shape=(1,))])
		max_features: Array((1,), [ArctanBound(a_max=[1], a_min=[0], name=At(0,1), shape=(1,))])
		min_samples_split: Array((1,), [ArctanBound(a_max=[1], a_min=[0], name=At(0,1), shape=(1,))])
		bootstrap: True
		class_weight: UnorderedDiscrete([None, 'balanced'])



.. _Metrics:
 
*******
Metrics
*******

Different availible choices for the `metric` parameter are shown below.
`metric` is accepted by :func:`Evaluate <ABCD_ML.ABCD_ML.ABCD_ML.Evaluate>` and :func:`Test <ABCD_ML.ABCD_ML.ABCD_ML.Test>`.
The exact str indicator for each `metric` is represented bythe sub-heading (within "")
The avaliable metrics are further broken down by which can work with different problem_types.
Additionally, a link to the original models documentation is shown.
Note: When supplying the metric as a str indicator you donot need to include the prepended "multiclass"

binary
======
"accuracy"
**********

  Base Func Documenation: :func:`sklearn.metrics.accuracy_score`

"balanced accuracy"
*******************

  Base Func Documenation: :func:`sklearn.metrics.balanced_accuracy_score`

"f1"
****

  Base Func Documenation: :func:`sklearn.metrics.f1_score`

"jaccard"
*********

  Base Func Documenation: :func:`sklearn.metrics.jaccard_score`

"log"
*****

  Base Func Documenation: :func:`sklearn.metrics.log_loss`

"macro average precision"
*************************

  Base Func Documenation: :func:`sklearn.metrics.average_precision_score`

"macro roc auc"
***************

  Base Func Documenation: :func:`sklearn.metrics.roc_auc_score`

"matthews"
**********

  Base Func Documenation: :func:`sklearn.metrics.matthews_corrcoef`

"neg brier"
***********

  Base Func Documenation: :func:`sklearn.metrics.brier_score_loss`

"neg hamming"
*************

  Base Func Documenation: :func:`sklearn.metrics.hamming_loss`

"precision"
***********

  Base Func Documenation: :func:`sklearn.metrics.precision_score`

"recall"
********

  Base Func Documenation: :func:`sklearn.metrics.recall_score`


regression
==========
"explained variance"
********************

  Base Func Documenation: :func:`sklearn.metrics.explained_variance_score`

"neg max error"
***************

  Base Func Documenation: :func:`sklearn.metrics.max_error`

"neg mean absolute error"
*************************

  Base Func Documenation: :func:`sklearn.metrics.mean_absolute_error`

"neg mean squared error"
************************

  Base Func Documenation: :func:`sklearn.metrics.mean_squared_error`

"neg mean squared log error"
****************************

  Base Func Documenation: :func:`sklearn.metrics.mean_squared_log_error`

"neg median absolute error"
***************************

  Base Func Documenation: :func:`sklearn.metrics.median_absolute_error`

"r2"
****

  Base Func Documenation: :func:`sklearn.metrics.r2_score`


categorical
===========
"accuracy"
**********

  Base Func Documenation: :func:`sklearn.metrics.accuracy_score`

"balanced accuracy"
*******************

  Base Func Documenation: :func:`sklearn.metrics.balanced_accuracy_score`

"by class f1"
*************

  Base Func Documenation: :func:`sklearn.metrics.f1_score`

"by class jaccard"
******************

  Base Func Documenation: :func:`sklearn.metrics.jaccard_score`

"by class precision"
********************

  Base Func Documenation: :func:`sklearn.metrics.precision_score`

"by class recall"
*****************

  Base Func Documenation: :func:`sklearn.metrics.recall_score`

"log"
*****

  Base Func Documenation: :func:`sklearn.metrics.log_loss`

"macro f1"
**********

  Base Func Documenation: :func:`sklearn.metrics.f1_score`

"macro jaccard"
***************

  Base Func Documenation: :func:`sklearn.metrics.jaccard_score`

"macro precision"
*****************

  Base Func Documenation: :func:`sklearn.metrics.precision_score`

"macro recall"
**************

  Base Func Documenation: :func:`sklearn.metrics.recall_score`

"matthews"
**********

  Base Func Documenation: :func:`sklearn.metrics.matthews_corrcoef`

"micro f1"
**********

  Base Func Documenation: :func:`sklearn.metrics.f1_score`

"micro jaccard"
***************

  Base Func Documenation: :func:`sklearn.metrics.jaccard_score`

"micro precision"
*****************

  Base Func Documenation: :func:`sklearn.metrics.precision_score`

"micro recall"
**************

  Base Func Documenation: :func:`sklearn.metrics.recall_score`

"multiclass by class average precision"
***************************************

  Base Func Documenation: :func:`sklearn.metrics.average_precision_score`

"multiclass by class roc auc"
*****************************

  Base Func Documenation: :func:`sklearn.metrics.roc_auc_score`

"multiclass macro average precision"
************************************

  Base Func Documenation: :func:`sklearn.metrics.average_precision_score`

"multiclass macro roc auc"
**************************

  Base Func Documenation: :func:`sklearn.metrics.roc_auc_score`

"multiclass micro average precision"
************************************

  Base Func Documenation: :func:`sklearn.metrics.average_precision_score`

"multiclass micro roc auc"
**************************

  Base Func Documenation: :func:`sklearn.metrics.roc_auc_score`

"multiclass samples average precision"
**************************************

  Base Func Documenation: :func:`sklearn.metrics.average_precision_score`

"multiclass samples roc auc"
****************************

  Base Func Documenation: :func:`sklearn.metrics.roc_auc_score`

"multiclass weighted average precision"
***************************************

  Base Func Documenation: :func:`sklearn.metrics.average_precision_score`

"multiclass weighted roc auc"
*****************************

  Base Func Documenation: :func:`sklearn.metrics.roc_auc_score`

"neg hamming"
*************

  Base Func Documenation: :func:`sklearn.metrics.hamming_loss`

"weighted f1"
*************

  Base Func Documenation: :func:`sklearn.metrics.f1_score`

"weighted jaccard"
******************

  Base Func Documenation: :func:`sklearn.metrics.jaccard_score`

"weighted precision"
********************

  Base Func Documenation: :func:`sklearn.metrics.precision_score`

"weighted recall"
*****************

  Base Func Documenation: :func:`sklearn.metrics.recall_score`


multilabel
==========
"accuracy"
**********

  Base Func Documenation: :func:`sklearn.metrics.accuracy_score`

"by class average precision"
****************************

  Base Func Documenation: :func:`sklearn.metrics.average_precision_score`

"by class f1"
*************

  Base Func Documenation: :func:`sklearn.metrics.f1_score`

"by class jaccard"
******************

  Base Func Documenation: :func:`sklearn.metrics.jaccard_score`

"by class precision"
********************

  Base Func Documenation: :func:`sklearn.metrics.precision_score`

"by class recall"
*****************

  Base Func Documenation: :func:`sklearn.metrics.recall_score`

"by class roc auc"
******************

  Base Func Documenation: :func:`sklearn.metrics.roc_auc_score`

"log"
*****

  Base Func Documenation: :func:`sklearn.metrics.log_loss`

"macro average precision"
*************************

  Base Func Documenation: :func:`sklearn.metrics.average_precision_score`

"macro f1"
**********

  Base Func Documenation: :func:`sklearn.metrics.f1_score`

"macro jaccard"
***************

  Base Func Documenation: :func:`sklearn.metrics.jaccard_score`

"macro precision"
*****************

  Base Func Documenation: :func:`sklearn.metrics.precision_score`

"macro recall"
**************

  Base Func Documenation: :func:`sklearn.metrics.recall_score`

"macro roc auc"
***************

  Base Func Documenation: :func:`sklearn.metrics.roc_auc_score`

"micro average precision"
*************************

  Base Func Documenation: :func:`sklearn.metrics.average_precision_score`

"micro f1"
**********

  Base Func Documenation: :func:`sklearn.metrics.f1_score`

"micro jaccard"
***************

  Base Func Documenation: :func:`sklearn.metrics.jaccard_score`

"micro precision"
*****************

  Base Func Documenation: :func:`sklearn.metrics.precision_score`

"micro recall"
**************

  Base Func Documenation: :func:`sklearn.metrics.recall_score`

"micro roc auc"
***************

  Base Func Documenation: :func:`sklearn.metrics.roc_auc_score`

"neg hamming"
*************

  Base Func Documenation: :func:`sklearn.metrics.hamming_loss`

"samples average precision"
***************************

  Base Func Documenation: :func:`sklearn.metrics.average_precision_score`

"samples f1"
************

  Base Func Documenation: :func:`sklearn.metrics.f1_score`

"samples jaccard"
*****************

  Base Func Documenation: :func:`sklearn.metrics.jaccard_score`

"samples precision"
*******************

  Base Func Documenation: :func:`sklearn.metrics.precision_score`

"samples recall"
****************

  Base Func Documenation: :func:`sklearn.metrics.recall_score`

"samples roc auc"
*****************

  Base Func Documenation: :func:`sklearn.metrics.roc_auc_score`

"weighted average precision"
****************************

  Base Func Documenation: :func:`sklearn.metrics.average_precision_score`

"weighted f1"
*************

  Base Func Documenation: :func:`sklearn.metrics.f1_score`

"weighted jaccard"
******************

  Base Func Documenation: :func:`sklearn.metrics.jaccard_score`

"weighted precision"
********************

  Base Func Documenation: :func:`sklearn.metrics.precision_score`

"weighted recall"
*****************

  Base Func Documenation: :func:`sklearn.metrics.recall_score`

"weighted roc auc"
******************

  Base Func Documenation: :func:`sklearn.metrics.roc_auc_score`


.. _Loaders:
 
*******
Loaders
*******

Different availible choices for the `loader` parameter are shown below.
loader is accepted by :func:`Evaluate <ABCD_ML.ABCD_ML.ABCD_ML.Evaluate>` and :func:`Test <ABCD_ML.ABCD_ML.ABCD_ML.Test>`.
The exact str indicator for each `loader` is represented by the sub-heading (within "")
Additionally, a link to the original loaders documentation as well as the implemented parameter distributions are shown.

All Problem Types
=================
"identity"
**********

  Base Class Documenation: :class:`ABCD_ML.pipeline.extensions.Loaders.Identity`

  Param Distributions

	0. "default" ::

		defaults only



.. _Imputers:
 
********
Imputers
********

Different availible choices for the `imputer` parameter are shown below.
imputer is accepted by :func:`Evaluate <ABCD_ML.ABCD_ML.ABCD_ML.Evaluate>` and :func:`Test <ABCD_ML.ABCD_ML.ABCD_ML.Test>`.
The exact str indicator for each `imputer` is represented by the sub-heading (within "")
Additionally, a link to the original imputers documentation as well as the implemented parameter distributions are shown.
Imputers are also special, in that a model can be passed instead of the imputer str. In that case, the model will be used to fill any NaN by column.
For `imputer_scope` of float, or custom column names, only regression type models are valid, and for scope of categorical, only binary / multiclass model are valid!
The sklearn iterative imputer is used when a model is passed.
Also, if a model is passed, then the `imputer_params` argument will then be considered as applied to the base  estimator / model!

All Problem Types
=================
"mean"
******

  Base Class Documenation: :class:`sklearn.impute.SimpleImputer`

  Param Distributions

	0. "mean imp" ::

		strategy: mean


"median"
********

  Base Class Documenation: :class:`sklearn.impute.SimpleImputer`

  Param Distributions

	0. "median imp" ::

		strategy: median


"most frequent"
***************

  Base Class Documenation: :class:`sklearn.impute.SimpleImputer`

  Param Distributions

	0. "most freq imp" ::

		strategy: most_frequent


"constant"
**********

  Base Class Documenation: :class:`sklearn.impute.SimpleImputer`

  Param Distributions

	0. "constant imp" ::

		strategy: constant


"iterative"
***********

  Base Class Documenation: :class:`sklearn.impute.IterativeImputer`

  Param Distributions

	0. "iterative imp" ::

		initial_strategy: mean



.. _Scalers:
 
*******
Scalers
*******

Different availible choices for the `scaler` parameter are shown below.
scaler is accepted by :func:`Evaluate <ABCD_ML.ABCD_ML.ABCD_ML.Evaluate>` and :func:`Test <ABCD_ML.ABCD_ML.ABCD_ML.Test>`.
The exact str indicator for each `scaler` is represented by the sub-heading (within "")
Additionally, a link to the original scalers documentation as well as the implemented parameter distributions are shown.

All Problem Types
=================
"standard"
**********

  Base Class Documenation: :class:`sklearn.preprocessing.StandardScaler`

  Param Distributions

	0. "base standard" ::

		with_mean: True
		with_std: True


"minmax"
********

  Base Class Documenation: :class:`sklearn.preprocessing.MinMaxScaler`

  Param Distributions

	0. "base minmax" ::

		feature_range: (0, 1)


"maxabs"
********

  Base Class Documenation: :class:`sklearn.preprocessing.MaxAbsScaler`

  Param Distributions

	0. "base minmax" ::

		feature_range: (0, 1)


"robust"
********

  Base Class Documenation: :class:`sklearn.preprocessing.RobustScaler`

  Param Distributions

	0. "base robust" ::

		quantile_range: (5, 95)

	1. "robust gs" ::

		quantile_range: UnorderedDiscrete([(1, 99), (3, 97), (5, 95), (10, 90), (15, 85), (20, 80), (25, 75), (30, 70), (35, 65), (40, 60)])


"yeo"
*****

  Base Class Documenation: :class:`sklearn.preprocessing.PowerTransformer`

  Param Distributions

	0. "base yeo" ::

		method: yeo-johnson
		standardize: True


"boxcox"
********

  Base Class Documenation: :class:`sklearn.preprocessing.PowerTransformer`

  Param Distributions

	0. "base boxcox" ::

		method: box-cox
		standardize: True


"winsorize"
***********

  Base Class Documenation: :class:`ABCD_ML.pipeline.extensions.Scalers.Winsorizer`

  Param Distributions

	0. "base winsorize" ::

		quantile_range: (1, 99)

	1. "winsorize gs" ::

		quantile_range: UnorderedDiscrete([(0.1, 99.9), (0.5, 99.5), (1, 99), (1.5, 98.5), (2, 98), (2.5, 97.5), (3, 97), (3.5, 96.5), (4, 96), (4.5, 95.5), (5, 95)])


"quantile norm"
***************

  Base Class Documenation: :class:`sklearn.preprocessing.QuantileTransformer`

  Param Distributions

	0. "base quant norm" ::

		output_distribution: normal


"quantile uniform"
******************

  Base Class Documenation: :class:`sklearn.preprocessing.QuantileTransformer`

  Param Distributions

	0. "base quant uniform" ::

		output_distribution: uniform


"normalize"
***********

  Base Class Documenation: :class:`sklearn.preprocessing.Normalizer`

  Param Distributions

	0. "default" ::

		defaults only



.. _Transformers:
 
************
Transformers
************

Different availible choices for the `transformer` parameter are shown below.
transformer is accepted by :func:`Evaluate <ABCD_ML.ABCD_ML.ABCD_ML.Evaluate>` and :func:`Test <ABCD_ML.ABCD_ML.ABCD_ML.Test>`.
The exact str indicator for each `transformer` is represented by the sub-heading (within "")
Additionally, a link to the original transformers documentation as well as the implemented parameter distributions are shown.

All Problem Types
=================
"pca"
*****

  Base Class Documenation: :class:`sklearn.decomposition.PCA`

  Param Distributions

	0. "default" ::

		defaults only

	1. "pca var search" ::

		n_components: Array((1,), [ArctanBound(a_max=[0.99], a_min=[0.1], name=At(0.1,0.99), shape=(1,))])
		svd_solver: full


"sparse pca"
************

  Base Class Documenation: :class:`sklearn.decomposition.SparsePCA`

  Param Distributions

	0. "default" ::

		defaults only


"mini batch sparse pca"
***********************

  Base Class Documenation: :class:`sklearn.decomposition.MiniBatchSparsePCA`

  Param Distributions

	0. "default" ::

		defaults only


"factor analysis"
*****************

  Base Class Documenation: :class:`sklearn.decomposition.FactorAnalysis`

  Param Distributions

	0. "default" ::

		defaults only


"dictionary learning"
*********************

  Base Class Documenation: :class:`sklearn.decomposition.DictionaryLearning`

  Param Distributions

	0. "default" ::

		defaults only


"mini batch dictionary learning"
********************************

  Base Class Documenation: :class:`sklearn.decomposition.MiniBatchDictionaryLearning`

  Param Distributions

	0. "default" ::

		defaults only


"fast ica"
**********

  Base Class Documenation: :class:`sklearn.decomposition.FastICA`

  Param Distributions

	0. "default" ::

		defaults only


"incremental pca"
*****************

  Base Class Documenation: :class:`sklearn.decomposition.IncrementalPCA`

  Param Distributions

	0. "default" ::

		defaults only


"kernel pca"
************

  Base Class Documenation: :class:`sklearn.decomposition.KernelPCA`

  Param Distributions

	0. "default" ::

		defaults only


"nmf"
*****

  Base Class Documenation: :class:`sklearn.decomposition.NMF`

  Param Distributions

	0. "default" ::

		defaults only



.. _Samplers:
 
********
Samplers
********

Different availible choices for the `sampler` parameter are shown below.
`sampler` is accepted by :func:`Evaluate <ABCD_ML.ABCD_ML.ABCD_ML.Evaluate>` and :func:`Test <ABCD_ML.ABCD_ML.ABCD_ML.Test>`.
The exact str indicator for each `sampler` is represented by the sub-heading (within "")
Additionally, a link to the original samplers documentation as well as the implemented parameter distributions are shown.

All Problem Types
=================
"random over sampler"
*********************

  Base Class Documenation: :class:`imblearn.over_sampling.RandomOverSampler`

  Param Distributions

	0. "base no change sampler" ::

		sampler_type: no change
		regression_bins: 3
		regression_bin_strategy: uniform


"smote"
*******

  Base Class Documenation: :class:`imblearn.over_sampling.SMOTE`

  Param Distributions

	0. "base change sampler" ::

		sampler_type: change
		regression_bins: 3
		regression_bin_strategy: uniform


"adasyn"
********

  Base Class Documenation: :class:`imblearn.over_sampling.ADASYN`

  Param Distributions

	0. "base change sampler" ::

		sampler_type: change
		regression_bins: 3
		regression_bin_strategy: uniform


"borderline smote"
******************

  Base Class Documenation: :class:`imblearn.over_sampling.BorderlineSMOTE`

  Param Distributions

	0. "base change sampler" ::

		sampler_type: change
		regression_bins: 3
		regression_bin_strategy: uniform


"svm smote"
***********

  Base Class Documenation: :class:`imblearn.over_sampling.SVMSMOTE`

  Param Distributions

	0. "base change sampler" ::

		sampler_type: change
		regression_bins: 3
		regression_bin_strategy: uniform


"kmeans smote"
**************

  Base Class Documenation: :class:`imblearn.over_sampling.KMeansSMOTE`

  Param Distributions

	0. "base change sampler" ::

		sampler_type: change
		regression_bins: 3
		regression_bin_strategy: uniform


"smote nc"
**********

  Base Class Documenation: :class:`imblearn.over_sampling.SMOTENC`

  Param Distributions

	0. "base special sampler" ::

		sampler_type: special
		regression_bins: 3
		regression_bin_strategy: uniform


"cluster centroids"
*******************

  Base Class Documenation: :class:`imblearn.under_sampling.ClusterCentroids`

  Param Distributions

	0. "base change sampler" ::

		sampler_type: change
		regression_bins: 3
		regression_bin_strategy: uniform


"random under sampler"
**********************

  Base Class Documenation: :class:`imblearn.under_sampling.RandomUnderSampler`

  Param Distributions

	0. "base no change sampler" ::

		sampler_type: no change
		regression_bins: 3
		regression_bin_strategy: uniform

	1. "rus binary ratio" ::

		sampler_type: no change
		regression_bins: 3
		regression_bin_strategy: uniform
		sampling_strategy: Array((1,), [ArctanBound(a_max=[1], a_min=[0.1], name=At(0.1,1), shape=(1,))])


"near miss"
***********

  Base Class Documenation: :class:`imblearn.under_sampling.NearMiss`

  Param Distributions

	0. "base no change sampler" ::

		sampler_type: no change
		regression_bins: 3
		regression_bin_strategy: uniform


"tomek links"
*************

  Base Class Documenation: :class:`imblearn.under_sampling.TomekLinks`

  Param Distributions

	0. "base no change sampler" ::

		sampler_type: no change
		regression_bins: 3
		regression_bin_strategy: uniform


"enn"
*****

  Base Class Documenation: :class:`imblearn.under_sampling.EditedNearestNeighbours`

  Param Distributions

	0. "base no change sampler" ::

		sampler_type: no change
		regression_bins: 3
		regression_bin_strategy: uniform


"renn"
******

  Base Class Documenation: :class:`imblearn.under_sampling.RepeatedEditedNearestNeighbours`

  Param Distributions

	0. "base no change sampler" ::

		sampler_type: no change
		regression_bins: 3
		regression_bin_strategy: uniform


"all knn"
*********

  Base Class Documenation: :class:`imblearn.under_sampling.AllKNN`

  Param Distributions

	0. "base no change sampler" ::

		sampler_type: no change
		regression_bins: 3
		regression_bin_strategy: uniform


"condensed nn"
**************

  Base Class Documenation: :class:`imblearn.under_sampling.CondensedNearestNeighbour`

  Param Distributions

	0. "base no change sampler" ::

		sampler_type: no change
		regression_bins: 3
		regression_bin_strategy: uniform


"one sided selection"
*********************

  Base Class Documenation: :class:`imblearn.under_sampling.OneSidedSelection`

  Param Distributions

	0. "base no change sampler" ::

		sampler_type: no change
		regression_bins: 3
		regression_bin_strategy: uniform


"neighbourhood cleaning rule"
*****************************

  Base Class Documenation: :class:`imblearn.under_sampling.NeighbourhoodCleaningRule`

  Param Distributions

	0. "base no change sampler" ::

		sampler_type: no change
		regression_bins: 3
		regression_bin_strategy: uniform


"smote enn"
***********

  Base Class Documenation: :class:`imblearn.combine.SMOTEENN`

  Param Distributions

	0. "base change sampler" ::

		sampler_type: change
		regression_bins: 3
		regression_bin_strategy: uniform


"smote tomek"
*************

  Base Class Documenation: :class:`imblearn.combine.SMOTETomek`

  Param Distributions

	0. "base change sampler" ::

		sampler_type: change
		regression_bins: 3
		regression_bin_strategy: uniform



.. _Feat Selectors:
 
**************
Feat Selectors
**************

Different availible choices for the `feat_selector` parameter are shown below.
`feat_selector` is accepted by :func:`Evaluate <ABCD_ML.ABCD_ML.ABCD_ML.Evaluate>` and :func:`Test <ABCD_ML.ABCD_ML.ABCD_ML.Test>`.
The exact str indicator for each `feat_selector` is represented by the sub-heading (within "")
The avaliable feat selectors are further broken down by which can work with different problem_types.
Additionally, a link to the original feat selectors  documentation as well as the implemented parameter distributions are shown.

binary
======
"rfe"
*****

  Base Class Documenation: :class:`sklearn.feature_selection.RFE`

  Param Distributions

	0. "base rfe" ::

		n_features_to_select: None

	1. "rfe num feats dist" ::

		n_features_to_select: Array((1,), [ArctanBound(a_max=[0.99], a_min=[0.01], name=At(0.01,0.99), shape=(1,))])


"selector"
**********

  Base Class Documenation: :class:`ABCD_ML.pipeline.extensions.Feat_Selectors.FeatureSelector`

  Param Distributions

	0. "random" ::

		mask: sets as random features

	1. "searchable" ::

		mask: sets as hyperparameters


"univariate selection classification"
*************************************

  Base Class Documenation: :class:`sklearn.feature_selection.SelectPercentile`

  Param Distributions

	0. "base univar fs classifier" ::

		score_func: <function f_classif at 0x7f5170988a70>
		percentile: 50

	1. "univar fs classifier dist" ::

		score_func: <function f_classif at 0x7f5170988a70>
		percentile: Array((1,), [ArctanBound(a_max=[99], a_min=[1], name=At(1,99), shape=(1,))])


"variance threshold"
********************

  Base Class Documenation: :class:`sklearn.feature_selection.VarianceThreshold`

  Param Distributions

	0. "default" ::

		defaults only



regression
==========
"rfe"
*****

  Base Class Documenation: :class:`sklearn.feature_selection.RFE`

  Param Distributions

	0. "base rfe" ::

		n_features_to_select: None

	1. "rfe num feats dist" ::

		n_features_to_select: Array((1,), [ArctanBound(a_max=[0.99], a_min=[0.01], name=At(0.01,0.99), shape=(1,))])


"selector"
**********

  Base Class Documenation: :class:`ABCD_ML.pipeline.extensions.Feat_Selectors.FeatureSelector`

  Param Distributions

	0. "random" ::

		mask: sets as random features

	1. "searchable" ::

		mask: sets as hyperparameters


"univariate selection regression"
*********************************

  Base Class Documenation: :class:`sklearn.feature_selection.SelectPercentile`

  Param Distributions

	0. "base univar fs regression" ::

		score_func: <function f_regression at 0x7f5170988dd0>
		percentile: 50

	1. "univar fs regression dist" ::

		score_func: <function f_regression at 0x7f5170988dd0>
		percentile: Array((1,), [ArctanBound(a_max=[99], a_min=[1], name=At(1,99), shape=(1,))])


"variance threshold"
********************

  Base Class Documenation: :class:`sklearn.feature_selection.VarianceThreshold`

  Param Distributions

	0. "default" ::

		defaults only



categorical
===========
"rfe"
*****

  Base Class Documenation: :class:`sklearn.feature_selection.RFE`

  Param Distributions

	0. "base rfe" ::

		n_features_to_select: None

	1. "rfe num feats dist" ::

		n_features_to_select: Array((1,), [ArctanBound(a_max=[0.99], a_min=[0.01], name=At(0.01,0.99), shape=(1,))])


"selector"
**********

  Base Class Documenation: :class:`ABCD_ML.pipeline.extensions.Feat_Selectors.FeatureSelector`

  Param Distributions

	0. "random" ::

		mask: sets as random features

	1. "searchable" ::

		mask: sets as hyperparameters


"univariate selection classification"
*************************************

  Base Class Documenation: :class:`sklearn.feature_selection.SelectPercentile`

  Param Distributions

	0. "base univar fs classifier" ::

		score_func: <function f_classif at 0x7f5170988a70>
		percentile: 50

	1. "univar fs classifier dist" ::

		score_func: <function f_classif at 0x7f5170988a70>
		percentile: Array((1,), [ArctanBound(a_max=[99], a_min=[1], name=At(1,99), shape=(1,))])


"variance threshold"
********************

  Base Class Documenation: :class:`sklearn.feature_selection.VarianceThreshold`

  Param Distributions

	0. "default" ::

		defaults only



multilabel
==========
"selector"
**********

  Base Class Documenation: :class:`ABCD_ML.pipeline.extensions.Feat_Selectors.FeatureSelector`

  Param Distributions

	0. "random" ::

		mask: sets as random features

	1. "searchable" ::

		mask: sets as hyperparameters


"variance threshold"
********************

  Base Class Documenation: :class:`sklearn.feature_selection.VarianceThreshold`

  Param Distributions

	0. "default" ::

		defaults only



.. _Ensemble Types:
 
**************
Ensemble Types
**************

Different availible choices for the `ensemble` parameter are shown below.
`ensemble` is accepted by :func:`Evaluate <ABCD_ML.ABCD_ML.ABCD_ML.Evaluate>` and :func:`Test <ABCD_ML.ABCD_ML.ABCD_ML.Test>`.
The exact str indicator for each `ensemble` is represented by the sub-heading (within "")
The avaliable ensemble types are further broken down by which can work with different problem_types.
Additionally, a link to the original ensemble types  documentation as well as the implemented parameter distributions are shown.

binary
======
"adaboost classifier"
*********************

  Base Class Documenation: :class:`sklearn.ensemble.AdaBoostClassifier`

  Param Distributions

	0. "single default" ::

		needs_split: False
		single_estimator: True


"aposteriori"
*************

  Base Class Documenation: :class:`deslib.dcs.a_posteriori.APosteriori`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"apriori"
*********

  Base Class Documenation: :class:`deslib.dcs.a_priori.APriori`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"bagging classifier"
********************

  Base Class Documenation: :class:`sklearn.ensemble.BaggingClassifier`

  Param Distributions

	0. "single default" ::

		needs_split: False
		single_estimator: True


"balanced bagging classifier"
*****************************

  Base Class Documenation: :class:`imblearn.ensemble.BalancedBaggingClassifier`

  Param Distributions

	0. "bb default" ::

		needs_split: False
		single_estimator: True


"des clustering"
****************

  Base Class Documenation: :class:`deslib.des.des_clustering.DESClustering`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"des knn"
*********

  Base Class Documenation: :class:`deslib.des.des_knn.DESKNN`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"deskl"
*******

  Base Class Documenation: :class:`deslib.des.probabilistic.DESKL`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"desmi"
*******

  Base Class Documenation: :class:`deslib.des.des_mi.DESMI`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"desp"
******

  Base Class Documenation: :class:`deslib.des.des_p.DESP`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"exponential"
*************

  Base Class Documenation: :class:`deslib.des.probabilistic.Exponential`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"knop"
******

  Base Class Documenation: :class:`deslib.des.knop.KNOP`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"knorae"
********

  Base Class Documenation: :class:`deslib.des.knora_e.KNORAE`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"knrau"
*******

  Base Class Documenation: :class:`deslib.des.knora_u.KNORAU`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"lca"
*****

  Base Class Documenation: :class:`deslib.dcs.lca.LCA`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"logarithmic"
*************

  Base Class Documenation: :class:`deslib.des.probabilistic.Logarithmic`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"mcb"
*****

  Base Class Documenation: :class:`deslib.dcs.mcb.MCB`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"metades"
*********

  Base Class Documenation: :class:`deslib.des.meta_des.METADES`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"min dif"
*********

  Base Class Documenation: :class:`deslib.des.probabilistic.MinimumDifference`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"mla"
*****

  Base Class Documenation: :class:`deslib.dcs.mla.MLA`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"ola"
*****

  Base Class Documenation: :class:`deslib.dcs.ola.OLA`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"rank"
******

  Base Class Documenation: :class:`deslib.dcs.rank.Rank`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"rrc"
*****

  Base Class Documenation: :class:`deslib.des.probabilistic.RRC`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"single best"
*************

  Base Class Documenation: :class:`deslib.static.single_best.SingleBest`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"stacked"
*********

  Base Class Documenation: :class:`deslib.static.stacked.StackedClassifier`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"stacking classifier"
*********************

  Base Class Documenation: :class:`sklearn.ensemble.StackingClassifier`

  Param Distributions

	0. "stacking default" ::

		needs_split: False
		single_estimator: False
		cv: 3



regression
==========
"adaboost regressor"
********************

  Base Class Documenation: :class:`sklearn.ensemble.AdaBoostRegressor`

  Param Distributions

	0. "single default" ::

		needs_split: False
		single_estimator: True


"bagging regressor"
*******************

  Base Class Documenation: :class:`sklearn.ensemble.BaggingRegressor`

  Param Distributions

	0. "single default" ::

		needs_split: False
		single_estimator: True


"stacking regressor"
********************

  Base Class Documenation: :class:`sklearn.ensemble.StackingRegressor`

  Param Distributions

	0. "stacking default" ::

		needs_split: False
		single_estimator: False
		cv: 3



categorical
===========
"adaboost classifier"
*********************

  Base Class Documenation: :class:`sklearn.ensemble.AdaBoostClassifier`

  Param Distributions

	0. "single default" ::

		needs_split: False
		single_estimator: True


"aposteriori"
*************

  Base Class Documenation: :class:`deslib.dcs.a_posteriori.APosteriori`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"apriori"
*********

  Base Class Documenation: :class:`deslib.dcs.a_priori.APriori`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"bagging classifier"
********************

  Base Class Documenation: :class:`sklearn.ensemble.BaggingClassifier`

  Param Distributions

	0. "single default" ::

		needs_split: False
		single_estimator: True


"balanced bagging classifier"
*****************************

  Base Class Documenation: :class:`imblearn.ensemble.BalancedBaggingClassifier`

  Param Distributions

	0. "bb default" ::

		needs_split: False
		single_estimator: True


"des clustering"
****************

  Base Class Documenation: :class:`deslib.des.des_clustering.DESClustering`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"des knn"
*********

  Base Class Documenation: :class:`deslib.des.des_knn.DESKNN`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"deskl"
*******

  Base Class Documenation: :class:`deslib.des.probabilistic.DESKL`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"desmi"
*******

  Base Class Documenation: :class:`deslib.des.des_mi.DESMI`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"desp"
******

  Base Class Documenation: :class:`deslib.des.des_p.DESP`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"exponential"
*************

  Base Class Documenation: :class:`deslib.des.probabilistic.Exponential`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"knop"
******

  Base Class Documenation: :class:`deslib.des.knop.KNOP`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"knorae"
********

  Base Class Documenation: :class:`deslib.des.knora_e.KNORAE`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"knrau"
*******

  Base Class Documenation: :class:`deslib.des.knora_u.KNORAU`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"lca"
*****

  Base Class Documenation: :class:`deslib.dcs.lca.LCA`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"logarithmic"
*************

  Base Class Documenation: :class:`deslib.des.probabilistic.Logarithmic`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"mcb"
*****

  Base Class Documenation: :class:`deslib.dcs.mcb.MCB`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"metades"
*********

  Base Class Documenation: :class:`deslib.des.meta_des.METADES`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"min dif"
*********

  Base Class Documenation: :class:`deslib.des.probabilistic.MinimumDifference`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"mla"
*****

  Base Class Documenation: :class:`deslib.dcs.mla.MLA`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"ola"
*****

  Base Class Documenation: :class:`deslib.dcs.ola.OLA`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"rank"
******

  Base Class Documenation: :class:`deslib.dcs.rank.Rank`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"rrc"
*****

  Base Class Documenation: :class:`deslib.des.probabilistic.RRC`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"single best"
*************

  Base Class Documenation: :class:`deslib.static.single_best.SingleBest`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"stacked"
*********

  Base Class Documenation: :class:`deslib.static.stacked.StackedClassifier`

  Param Distributions

	0. "des default" ::

		needs_split: True
		single_estimator: False


"stacking classifier"
*********************

  Base Class Documenation: :class:`sklearn.ensemble.StackingClassifier`

  Param Distributions

	0. "stacking default" ::

		needs_split: False
		single_estimator: False
		cv: 3



multilabel
==========

