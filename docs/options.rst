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

	1. "dt classifier rs" ::

		max_depth: Scalar(shape=(1,), transforms=[ArctanBound(a_max=30, a_min=1)])
		min_samples_split: Scalar(shape=(1,), transforms=[ArctanBound(a_max=50, a_min=2)])
		class_weight: OrderedDiscrete(possibilities=[None, 'balanced'])


"elastic net logistic"
**********************

  Base Class Documenation: :class:`sklearn.linear_model.LogisticRegression`

  Param Distributions

	0. "base elastic" ::

		solver: saga
		max_iter: 5000
		multi_class: auto
		penalty: elasticnet
		class_weight: OrderedDiscrete(possibilities=[None, 'balanced'])
		l1_ratio: 0.5

	1. "elastic classifier" ::

		solver: saga
		max_iter: 5000
		multi_class: auto
		penalty: elasticnet
		class_weight: OrderedDiscrete(possibilities=[None, 'balanced'])
		l1_ratio: Scalar(shape=(1,), transforms=[ArctanBound(a_max=1, a_min=0)])
		C: Scalar(shape=(1,), transforms=[ArctanBound(a_max=4, a_min=-4), Exponentiate(base=10, coeff=-1)])


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

	1. "knn rs" ::

		weights: OrderedDiscrete(possibilities=['uniform', 'distance'])
		n_neighbors: Scalar(shape=(1,), transforms=[ArctanBound(a_max=25, a_min=2)])


"lasso logistic"
****************

  Base Class Documenation: :class:`sklearn.linear_model.LogisticRegression`

  Param Distributions

	0. "base lasso" ::

		solver: saga
		max_iter: 5000
		multi_class: auto
		penalty: l1
		class_weight: OrderedDiscrete(possibilities=[None, 'balanced'])

	1. "lasso C" ::

		solver: saga
		max_iter: 5000
		multi_class: auto
		penalty: l1
		class_weight: OrderedDiscrete(possibilities=[None, 'balanced'])
		C: Scalar(shape=(1,), transforms=[ArctanBound(a_max=4, a_min=-4), Exponentiate(base=10, coeff=-1)])


"light gbm classifier"
**********************

  Base Class Documenation: :class:`lightgbm.LGBMClassifier`

  Param Distributions

	0. "base lgbm" ::

		silent: True

	1. "lgbm classifier rs1" ::

		silent: True
		boosting_type: OrderedDiscrete(possibilities=['gbdt', 'dart', 'goss'])
		n_estimators: Scalar(shape=(1,), transforms=[ArctanBound(a_max=500, a_min=3)])
		num_leaves: Scalar(shape=(1,), transforms=[ArctanBound(a_max=80, a_min=6)])
		min_child_samples: Scalar(shape=(1,), transforms=[ArctanBound(a_max=500, a_min=10)])
		min_child_weight: Scalar(shape=(1,), transforms=[ArctanBound(a_max=5, a_min=-4), Exponentiate(base=10, coeff=-1)])
		subsample: Scalar(shape=(1,), transforms=[ArctanBound(a_max=0.95, a_min=0.3)])
		colsample_bytree: Scalar(shape=(1,), transforms=[ArctanBound(a_max=0.95, a_min=0.3)])
		reg_alpha: Scalar(shape=(1,), transforms=[ArctanBound(a_max=1, a_min=-2), Exponentiate(base=10, coeff=-1)])
		reg_lambda: Scalar(shape=(1,), transforms=[ArctanBound(a_max=1, a_min=-2), Exponentiate(base=10, coeff=-1)])
		class_weight: OrderedDiscrete(possibilities=[None, 'balanced'])

	2. "lgbm classifier rs2" ::

		silent: True
		lambda_l2: 0.001
		boosting_type: OrderedDiscrete(possibilities=['gbdt', 'dart'])
		min_child_samples: OrderedDiscrete(possibilities=[1, 5, 7, 10, 15, 20, 35, 50, 100, 200, 500, 1000])
		num_leaves: OrderedDiscrete(possibilities=[2, 4, 7, 10, 15, 20, 25, 30, 35, 40, 50, 65, 80, 100, 125, 150, 200, 250])
		colsample_bytree: OrderedDiscrete(possibilities=[0.7, 0.9, 1.0])
		subsample: Scalar(shape=(1,), transforms=[ArctanBound(a_max=1, a_min=0.3)])
		learning_rate: OrderedDiscrete(possibilities=[0.01, 0.05, 0.1])
		n_estimators: OrderedDiscrete(possibilities=[5, 20, 35, 50, 75, 100, 150, 200, 350, 500, 750, 1000])
		class_weight: OrderedDiscrete(possibilities=[None, 'balanced'])


"logistic"
**********

  Base Class Documenation: :class:`sklearn.linear_model.LogisticRegression`

  Param Distributions

	0. "base logistic" ::

		solver: saga
		max_iter: 5000
		multi_class: auto
		penalty: none
		class_weight: OrderedDiscrete(possibilities=[None, 'balanced'])


"mlp classifier"
****************

  Base Class Documenation: :class:`sklearn.neural_network.MLPClassifier`

  Param Distributions

	0. "base mlp" ::

		defaults only

	1. "mlp rs" ::

		hidden_layer_sizes: Array(shape=(1, 1, 1), transforms=[ArctanBound(a_max=100, a_min=2)])
		activation: OrderedDiscrete(possibilities=['identity', 'logistic', 'tanh', 'relu'])
		alpha: Scalar(shape=(1,), transforms=[ArctanBound(a_max=5, a_min=-2), Exponentiate(base=10, coeff=-1)])
		batch_size: Scalar(shape=(1,), transforms=[ArctanBound(a_max=200, a_min=2)])
		learning_rate: OrderedDiscrete(possibilities=['constant', 'invscaling', 'adaptive'])
		learning_rate_init: Scalar(shape=(1,), transforms=[ArctanBound(a_max=5, a_min=-2), Exponentiate(base=10, coeff=-1)])
		max_iter: Scalar(shape=(1,), transforms=[ArctanBound(a_max=500, a_min=100)])
		beta_1: Scalar(shape=(1,), transforms=[ArctanBound(a_max=0.95, a_min=0.1)])
		beta_2: Scalar(shape=(1,), transforms=[ArctanBound(a_max=0.95, a_min=0.1)])

	2. "mlp rs es" ::

		hidden_layer_sizes: Array(shape=(1, 1, 1), transforms=[ArctanBound(a_max=100, a_min=2)])
		activation: OrderedDiscrete(possibilities=['identity', 'logistic', 'tanh', 'relu'])
		alpha: Scalar(shape=(1,), transforms=[ArctanBound(a_max=5, a_min=-2), Exponentiate(base=10, coeff=-1)])
		batch_size: Scalar(shape=(1,), transforms=[ArctanBound(a_max=200, a_min=2)])
		learning_rate: OrderedDiscrete(possibilities=['constant', 'invscaling', 'adaptive'])
		learning_rate_init: Scalar(shape=(1,), transforms=[ArctanBound(a_max=5, a_min=-2), Exponentiate(base=10, coeff=-1)])
		max_iter: Scalar(shape=(1,), transforms=[ArctanBound(a_max=500, a_min=100)])
		beta_1: Scalar(shape=(1,), transforms=[ArctanBound(a_max=0.95, a_min=0.1)])
		beta_2: Scalar(shape=(1,), transforms=[ArctanBound(a_max=0.95, a_min=0.1)])
		early_stopping: True
		n_iter_no_change: Scalar(shape=(1,), transforms=[ArctanBound(a_max=50, a_min=5)])

	3. "mlp layers search" ::

		hidden_layer_sizes: Array(shape=(1, 1, 1), transforms=[ArctanBound(a_max=100, a_min=2)])


"random forest classifier"
**************************

  Base Class Documenation: :class:`sklearn.ensemble.RandomForestClassifier`

  Param Distributions

	0. "base rf" ::

		n_estimators: 100

	1. "rf classifier rs" ::

		n_estimators: Scalar(shape=(1,), transforms=[ArctanBound(a_max=500, a_min=3)])
		max_depth: Scalar(shape=(1,), transforms=[ArctanBound(a_max=200, a_min=2)])
		max_features: Scalar(shape=(1,), transforms=[ArctanBound(a_max=1, a_min=0)])
		min_samples_split: Scalar(shape=(1,), transforms=[ArctanBound(a_max=1, a_min=0)])
		bootstrap: True
		class_weight: OrderedDiscrete(possibilities=[None, 'balanced'])


"ridge logistic"
****************

  Base Class Documenation: :class:`sklearn.linear_model.LogisticRegression`

  Param Distributions

	0. "base ridge" ::

		solver: saga
		max_iter: 5000
		multi_class: auto
		penalty: l2
		class_weight: OrderedDiscrete(possibilities=[None, 'balanced'])

	1. "ridge C" ::

		solver: saga
		max_iter: 5000
		multi_class: auto
		penalty: l2
		class_weight: OrderedDiscrete(possibilities=[None, 'balanced'])
		C: Scalar(shape=(1,), transforms=[ArctanBound(a_max=4, a_min=-4), Exponentiate(base=10, coeff=-1)])


"svm classifier"
****************

  Base Class Documenation: :class:`sklearn.svm.SVC`

  Param Distributions

	0. "base svm classifier" ::

		kernel: rbf
		gamma: scale
		probability: True

	1. "svm classifier rs" ::

		kernel: rbf
		gamma: Scalar(shape=(1,), transforms=[ArctanBound(a_max=6, a_min=1), Exponentiate(base=10, coeff=-1)])
		C: Scalar(shape=(1,), transforms=[ArctanBound(a_max=4, a_min=-4), Exponentiate(base=10, coeff=-1)])
		probability: True
		class_weight: OrderedDiscrete(possibilities=[None, 'balanced'])


"xgb classifier"
****************

  Base Class Documenation: :class:`xgboost.XGBClassifier`

  Param Distributions

	0. "base xgb" ::

		verbosity: 0

	1. "xgb rs" ::

		verbosity: 0
		max_depth: Scalar(shape=(1,), transforms=[ArctanBound(a_max=50, a_min=2)])
		learning_rate: Scalar(shape=(1,), transforms=[ArctanBound(a_max=0.5, a_min=0.01)])
		n_estimators: Scalar(shape=(1,), transforms=[ArctanBound(a_max=500, a_min=3)])
		min_child_weight: OrderedDiscrete(possibilities=[1, 5, 10, 50])
		subsample: Scalar(shape=(1,), transforms=[ArctanBound(a_max=1, a_min=0.5)])
		colsample_bytree: Scalar(shape=(1,), transforms=[ArctanBound(a_max=0.95, a_min=0.4)])



regression
==========
"dt regressor"
**************

  Base Class Documenation: :class:`sklearn.tree.DecisionTreeRegressor`

  Param Distributions

	0. "base dt" ::

		defaults only

	1. "dt rs" ::

		max_depth: Scalar(shape=(1,), transforms=[ArctanBound(a_max=30, a_min=1)])
		min_samples_split: Scalar(shape=(1,), transforms=[ArctanBound(a_max=50, a_min=2)])


"elastic net regressor"
***********************

  Base Class Documenation: :class:`sklearn.linear_model.ElasticNet`

  Param Distributions

	0. "base elastic net" ::

		max_iter: 5000

	1. "elastic regression" ::

		max_iter: 5000
		alpha: Scalar(shape=(1,), transforms=[ArctanBound(a_max=5, a_min=-2), Exponentiate(base=10, coeff=-1)])
		l1_ratio: Scalar(shape=(1,), transforms=[ArctanBound(a_max=1, a_min=0)])


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

	1. "knn rs" ::

		weights: OrderedDiscrete(possibilities=['uniform', 'distance'])
		n_neighbors: Scalar(shape=(1,), transforms=[ArctanBound(a_max=25, a_min=2)])


"lasso regressor"
*****************

  Base Class Documenation: :class:`sklearn.linear_model.Lasso`

  Param Distributions

	0. "base lasso regressor" ::

		max_iter: 5000

	1. "lasso regressor rs" ::

		alpha: Scalar(shape=(1,), transforms=[ArctanBound(a_max=5, a_min=-4), Exponentiate(base=10, coeff=-1)])


"light gbm regressor"
*********************

  Base Class Documenation: :class:`lightgbm.LGBMRegressor`

  Param Distributions

	0. "base lgbm" ::

		silent: True

	1. "lgbm rs1" ::

		silent: True
		boosting_type: OrderedDiscrete(possibilities=['gbdt', 'dart', 'goss'])
		n_estimators: Scalar(shape=(1,), transforms=[ArctanBound(a_max=500, a_min=3)])
		num_leaves: Scalar(shape=(1,), transforms=[ArctanBound(a_max=80, a_min=6)])
		min_child_samples: Scalar(shape=(1,), transforms=[ArctanBound(a_max=500, a_min=10)])
		min_child_weight: Scalar(shape=(1,), transforms=[ArctanBound(a_max=5, a_min=-4), Exponentiate(base=10, coeff=-1)])
		subsample: Scalar(shape=(1,), transforms=[ArctanBound(a_max=0.95, a_min=0.3)])
		colsample_bytree: Scalar(shape=(1,), transforms=[ArctanBound(a_max=0.95, a_min=0.3)])
		reg_alpha: Scalar(shape=(1,), transforms=[ArctanBound(a_max=1, a_min=-2), Exponentiate(base=10, coeff=-1)])
		reg_lambda: Scalar(shape=(1,), transforms=[ArctanBound(a_max=1, a_min=-2), Exponentiate(base=10, coeff=-1)])

	2. "lgbm rs2" ::

		silent: True
		lambda_l2: 0.001
		boosting_type: OrderedDiscrete(possibilities=['gbdt', 'dart'])
		min_child_samples: OrderedDiscrete(possibilities=[1, 5, 7, 10, 15, 20, 35, 50, 100, 200, 500, 1000])
		num_leaves: OrderedDiscrete(possibilities=[2, 4, 7, 10, 15, 20, 25, 30, 35, 40, 50, 65, 80, 100, 125, 150, 200, 250])
		colsample_bytree: OrderedDiscrete(possibilities=[0.7, 0.9, 1.0])
		subsample: Scalar(shape=(1,), transforms=[ArctanBound(a_max=1, a_min=0.3)])
		learning_rate: OrderedDiscrete(possibilities=[0.01, 0.05, 0.1])
		n_estimators: OrderedDiscrete(possibilities=[5, 20, 35, 50, 75, 100, 150, 200, 350, 500, 750, 1000])


"linear regressor"
******************

  Base Class Documenation: :class:`sklearn.linear_model.LinearRegression`

  Param Distributions

	0. "base linear" ::

		fit_intercept: True


"mlp regressor"
***************

  Base Class Documenation: :class:`sklearn.neural_network.MLPRegressor`

  Param Distributions

	0. "base mlp" ::

		defaults only

	1. "mlp rs" ::

		hidden_layer_sizes: Array(shape=(1, 1, 1), transforms=[ArctanBound(a_max=100, a_min=2)])
		activation: OrderedDiscrete(possibilities=['identity', 'logistic', 'tanh', 'relu'])
		alpha: Scalar(shape=(1,), transforms=[ArctanBound(a_max=5, a_min=-2), Exponentiate(base=10, coeff=-1)])
		batch_size: Scalar(shape=(1,), transforms=[ArctanBound(a_max=200, a_min=2)])
		learning_rate: OrderedDiscrete(possibilities=['constant', 'invscaling', 'adaptive'])
		learning_rate_init: Scalar(shape=(1,), transforms=[ArctanBound(a_max=5, a_min=-2), Exponentiate(base=10, coeff=-1)])
		max_iter: Scalar(shape=(1,), transforms=[ArctanBound(a_max=500, a_min=100)])
		beta_1: Scalar(shape=(1,), transforms=[ArctanBound(a_max=0.95, a_min=0.1)])
		beta_2: Scalar(shape=(1,), transforms=[ArctanBound(a_max=0.95, a_min=0.1)])

	2. "mlp rs es" ::

		hidden_layer_sizes: Array(shape=(1, 1, 1), transforms=[ArctanBound(a_max=100, a_min=2)])
		activation: OrderedDiscrete(possibilities=['identity', 'logistic', 'tanh', 'relu'])
		alpha: Scalar(shape=(1,), transforms=[ArctanBound(a_max=5, a_min=-2), Exponentiate(base=10, coeff=-1)])
		batch_size: Scalar(shape=(1,), transforms=[ArctanBound(a_max=200, a_min=2)])
		learning_rate: OrderedDiscrete(possibilities=['constant', 'invscaling', 'adaptive'])
		learning_rate_init: Scalar(shape=(1,), transforms=[ArctanBound(a_max=5, a_min=-2), Exponentiate(base=10, coeff=-1)])
		max_iter: Scalar(shape=(1,), transforms=[ArctanBound(a_max=500, a_min=100)])
		beta_1: Scalar(shape=(1,), transforms=[ArctanBound(a_max=0.95, a_min=0.1)])
		beta_2: Scalar(shape=(1,), transforms=[ArctanBound(a_max=0.95, a_min=0.1)])
		early_stopping: True
		n_iter_no_change: Scalar(shape=(1,), transforms=[ArctanBound(a_max=50, a_min=5)])

	3. "mlp layers search" ::

		hidden_layer_sizes: Array(shape=(1, 1, 1), transforms=[ArctanBound(a_max=100, a_min=2)])


"random forest regressor"
*************************

  Base Class Documenation: :class:`sklearn.ensemble.RandomForestRegressor`

  Param Distributions

	0. "base rf" ::

		n_estimators: 100

	1. "rf rs" ::

		n_estimators: Scalar(shape=(1,), transforms=[ArctanBound(a_max=500, a_min=3)])
		max_depth: Scalar(shape=(1,), transforms=[ArctanBound(a_max=200, a_min=2)])
		max_features: Scalar(shape=(1,), transforms=[ArctanBound(a_max=1, a_min=0)])
		min_samples_split: Scalar(shape=(1,), transforms=[ArctanBound(a_max=1, a_min=0)])
		bootstrap: True


"ridge regressor"
*****************

  Base Class Documenation: :class:`sklearn.linear_model.ridge.Ridge`

  Param Distributions

	0. "base ridge regressor" ::

		max_iter: 5000

	1. "ridge regressor rs" ::

		alpha: Scalar(shape=(1,), transforms=[ArctanBound(a_max=5, a_min=-4), Exponentiate(base=10, coeff=-1)])


"svm regressor"
***************

  Base Class Documenation: :class:`sklearn.svm.SVR`

  Param Distributions

	0. "base svm" ::

		kernel: rbf
		gamma: scale

	1. "svm rs" ::

		kernel: rbf
		gamma: Scalar(shape=(1,), transforms=[ArctanBound(a_max=6, a_min=1), Exponentiate(base=10, coeff=-1)])
		C: Scalar(shape=(1,), transforms=[ArctanBound(a_max=4, a_min=-4), Exponentiate(base=10, coeff=-1)])


"xgb regressor"
***************

  Base Class Documenation: :class:`xgboost.XGBRegressor`

  Param Distributions

	0. "base xgb" ::

		verbosity: 0

	1. "xgb rs" ::

		verbosity: 0
		max_depth: Scalar(shape=(1,), transforms=[ArctanBound(a_max=50, a_min=2)])
		learning_rate: Scalar(shape=(1,), transforms=[ArctanBound(a_max=0.5, a_min=0.01)])
		n_estimators: Scalar(shape=(1,), transforms=[ArctanBound(a_max=500, a_min=3)])
		min_child_weight: OrderedDiscrete(possibilities=[1, 5, 10, 50])
		subsample: Scalar(shape=(1,), transforms=[ArctanBound(a_max=1, a_min=0.5)])
		colsample_bytree: Scalar(shape=(1,), transforms=[ArctanBound(a_max=0.95, a_min=0.4)])



categorical
===========
"dt classifier"
***************

  Base Class Documenation: :class:`sklearn.tree.DecisionTreeClassifier`

  Param Distributions

	0. "base dt" ::

		defaults only

	1. "dt classifier rs" ::

		max_depth: Scalar(shape=(1,), transforms=[ArctanBound(a_max=30, a_min=1)])
		min_samples_split: Scalar(shape=(1,), transforms=[ArctanBound(a_max=50, a_min=2)])
		class_weight: OrderedDiscrete(possibilities=[None, 'balanced'])


"elastic net logistic"
**********************

  Base Class Documenation: :class:`sklearn.linear_model.LogisticRegression`

  Param Distributions

	0. "base elastic" ::

		solver: saga
		max_iter: 5000
		multi_class: auto
		penalty: elasticnet
		class_weight: OrderedDiscrete(possibilities=[None, 'balanced'])
		l1_ratio: 0.5

	1. "elastic classifier" ::

		solver: saga
		max_iter: 5000
		multi_class: auto
		penalty: elasticnet
		class_weight: OrderedDiscrete(possibilities=[None, 'balanced'])
		l1_ratio: Scalar(shape=(1,), transforms=[ArctanBound(a_max=1, a_min=0)])
		C: Scalar(shape=(1,), transforms=[ArctanBound(a_max=4, a_min=-4), Exponentiate(base=10, coeff=-1)])


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

	1. "knn rs" ::

		weights: OrderedDiscrete(possibilities=['uniform', 'distance'])
		n_neighbors: Scalar(shape=(1,), transforms=[ArctanBound(a_max=25, a_min=2)])


"lasso logistic"
****************

  Base Class Documenation: :class:`sklearn.linear_model.LogisticRegression`

  Param Distributions

	0. "base lasso" ::

		solver: saga
		max_iter: 5000
		multi_class: auto
		penalty: l1
		class_weight: OrderedDiscrete(possibilities=[None, 'balanced'])

	1. "lasso C" ::

		solver: saga
		max_iter: 5000
		multi_class: auto
		penalty: l1
		class_weight: OrderedDiscrete(possibilities=[None, 'balanced'])
		C: Scalar(shape=(1,), transforms=[ArctanBound(a_max=4, a_min=-4), Exponentiate(base=10, coeff=-1)])


"light gbm classifier"
**********************

  Base Class Documenation: :class:`lightgbm.LGBMClassifier`

  Param Distributions

	0. "base lgbm" ::

		silent: True

	1. "lgbm classifier rs1" ::

		silent: True
		boosting_type: OrderedDiscrete(possibilities=['gbdt', 'dart', 'goss'])
		n_estimators: Scalar(shape=(1,), transforms=[ArctanBound(a_max=500, a_min=3)])
		num_leaves: Scalar(shape=(1,), transforms=[ArctanBound(a_max=80, a_min=6)])
		min_child_samples: Scalar(shape=(1,), transforms=[ArctanBound(a_max=500, a_min=10)])
		min_child_weight: Scalar(shape=(1,), transforms=[ArctanBound(a_max=5, a_min=-4), Exponentiate(base=10, coeff=-1)])
		subsample: Scalar(shape=(1,), transforms=[ArctanBound(a_max=0.95, a_min=0.3)])
		colsample_bytree: Scalar(shape=(1,), transforms=[ArctanBound(a_max=0.95, a_min=0.3)])
		reg_alpha: Scalar(shape=(1,), transforms=[ArctanBound(a_max=1, a_min=-2), Exponentiate(base=10, coeff=-1)])
		reg_lambda: Scalar(shape=(1,), transforms=[ArctanBound(a_max=1, a_min=-2), Exponentiate(base=10, coeff=-1)])
		class_weight: OrderedDiscrete(possibilities=[None, 'balanced'])

	2. "lgbm classifier rs2" ::

		silent: True
		lambda_l2: 0.001
		boosting_type: OrderedDiscrete(possibilities=['gbdt', 'dart'])
		min_child_samples: OrderedDiscrete(possibilities=[1, 5, 7, 10, 15, 20, 35, 50, 100, 200, 500, 1000])
		num_leaves: OrderedDiscrete(possibilities=[2, 4, 7, 10, 15, 20, 25, 30, 35, 40, 50, 65, 80, 100, 125, 150, 200, 250])
		colsample_bytree: OrderedDiscrete(possibilities=[0.7, 0.9, 1.0])
		subsample: Scalar(shape=(1,), transforms=[ArctanBound(a_max=1, a_min=0.3)])
		learning_rate: OrderedDiscrete(possibilities=[0.01, 0.05, 0.1])
		n_estimators: OrderedDiscrete(possibilities=[5, 20, 35, 50, 75, 100, 150, 200, 350, 500, 750, 1000])
		class_weight: OrderedDiscrete(possibilities=[None, 'balanced'])


"logistic"
**********

  Base Class Documenation: :class:`sklearn.linear_model.LogisticRegression`

  Param Distributions

	0. "base logistic" ::

		solver: saga
		max_iter: 5000
		multi_class: auto
		penalty: none
		class_weight: OrderedDiscrete(possibilities=[None, 'balanced'])


"mlp classifier"
****************

  Base Class Documenation: :class:`sklearn.neural_network.MLPClassifier`

  Param Distributions

	0. "base mlp" ::

		defaults only

	1. "mlp rs" ::

		hidden_layer_sizes: Array(shape=(1, 1, 1), transforms=[ArctanBound(a_max=100, a_min=2)])
		activation: OrderedDiscrete(possibilities=['identity', 'logistic', 'tanh', 'relu'])
		alpha: Scalar(shape=(1,), transforms=[ArctanBound(a_max=5, a_min=-2), Exponentiate(base=10, coeff=-1)])
		batch_size: Scalar(shape=(1,), transforms=[ArctanBound(a_max=200, a_min=2)])
		learning_rate: OrderedDiscrete(possibilities=['constant', 'invscaling', 'adaptive'])
		learning_rate_init: Scalar(shape=(1,), transforms=[ArctanBound(a_max=5, a_min=-2), Exponentiate(base=10, coeff=-1)])
		max_iter: Scalar(shape=(1,), transforms=[ArctanBound(a_max=500, a_min=100)])
		beta_1: Scalar(shape=(1,), transforms=[ArctanBound(a_max=0.95, a_min=0.1)])
		beta_2: Scalar(shape=(1,), transforms=[ArctanBound(a_max=0.95, a_min=0.1)])

	2. "mlp rs es" ::

		hidden_layer_sizes: Array(shape=(1, 1, 1), transforms=[ArctanBound(a_max=100, a_min=2)])
		activation: OrderedDiscrete(possibilities=['identity', 'logistic', 'tanh', 'relu'])
		alpha: Scalar(shape=(1,), transforms=[ArctanBound(a_max=5, a_min=-2), Exponentiate(base=10, coeff=-1)])
		batch_size: Scalar(shape=(1,), transforms=[ArctanBound(a_max=200, a_min=2)])
		learning_rate: OrderedDiscrete(possibilities=['constant', 'invscaling', 'adaptive'])
		learning_rate_init: Scalar(shape=(1,), transforms=[ArctanBound(a_max=5, a_min=-2), Exponentiate(base=10, coeff=-1)])
		max_iter: Scalar(shape=(1,), transforms=[ArctanBound(a_max=500, a_min=100)])
		beta_1: Scalar(shape=(1,), transforms=[ArctanBound(a_max=0.95, a_min=0.1)])
		beta_2: Scalar(shape=(1,), transforms=[ArctanBound(a_max=0.95, a_min=0.1)])
		early_stopping: True
		n_iter_no_change: Scalar(shape=(1,), transforms=[ArctanBound(a_max=50, a_min=5)])

	3. "mlp layers search" ::

		hidden_layer_sizes: Array(shape=(1, 1, 1), transforms=[ArctanBound(a_max=100, a_min=2)])


"random forest classifier"
**************************

  Base Class Documenation: :class:`sklearn.ensemble.RandomForestClassifier`

  Param Distributions

	0. "base rf" ::

		n_estimators: 100

	1. "rf classifier rs" ::

		n_estimators: Scalar(shape=(1,), transforms=[ArctanBound(a_max=500, a_min=3)])
		max_depth: Scalar(shape=(1,), transforms=[ArctanBound(a_max=200, a_min=2)])
		max_features: Scalar(shape=(1,), transforms=[ArctanBound(a_max=1, a_min=0)])
		min_samples_split: Scalar(shape=(1,), transforms=[ArctanBound(a_max=1, a_min=0)])
		bootstrap: True
		class_weight: OrderedDiscrete(possibilities=[None, 'balanced'])


"ridge logistic"
****************

  Base Class Documenation: :class:`sklearn.linear_model.LogisticRegression`

  Param Distributions

	0. "base ridge" ::

		solver: saga
		max_iter: 5000
		multi_class: auto
		penalty: l2
		class_weight: OrderedDiscrete(possibilities=[None, 'balanced'])

	1. "ridge C" ::

		solver: saga
		max_iter: 5000
		multi_class: auto
		penalty: l2
		class_weight: OrderedDiscrete(possibilities=[None, 'balanced'])
		C: Scalar(shape=(1,), transforms=[ArctanBound(a_max=4, a_min=-4), Exponentiate(base=10, coeff=-1)])


"svm classifier"
****************

  Base Class Documenation: :class:`sklearn.svm.SVC`

  Param Distributions

	0. "base svm classifier" ::

		kernel: rbf
		gamma: scale
		probability: True

	1. "svm classifier rs" ::

		kernel: rbf
		gamma: Scalar(shape=(1,), transforms=[ArctanBound(a_max=6, a_min=1), Exponentiate(base=10, coeff=-1)])
		C: Scalar(shape=(1,), transforms=[ArctanBound(a_max=4, a_min=-4), Exponentiate(base=10, coeff=-1)])
		probability: True
		class_weight: OrderedDiscrete(possibilities=[None, 'balanced'])


"xgb classifier"
****************

  Base Class Documenation: :class:`xgboost.XGBClassifier`

  Param Distributions

	0. "base xgb" ::

		verbosity: 0

	1. "xgb rs" ::

		verbosity: 0
		max_depth: Scalar(shape=(1,), transforms=[ArctanBound(a_max=50, a_min=2)])
		learning_rate: Scalar(shape=(1,), transforms=[ArctanBound(a_max=0.5, a_min=0.01)])
		n_estimators: Scalar(shape=(1,), transforms=[ArctanBound(a_max=500, a_min=3)])
		min_child_weight: OrderedDiscrete(possibilities=[1, 5, 10, 50])
		subsample: Scalar(shape=(1,), transforms=[ArctanBound(a_max=1, a_min=0.5)])
		colsample_bytree: Scalar(shape=(1,), transforms=[ArctanBound(a_max=0.95, a_min=0.4)])



multilabel
==========
"dt classifier"
***************

  Base Class Documenation: :class:`sklearn.tree.DecisionTreeClassifier`

  Param Distributions

	0. "base dt" ::

		defaults only

	1. "dt classifier rs" ::

		max_depth: Scalar(shape=(1,), transforms=[ArctanBound(a_max=30, a_min=1)])
		min_samples_split: Scalar(shape=(1,), transforms=[ArctanBound(a_max=50, a_min=2)])
		class_weight: OrderedDiscrete(possibilities=[None, 'balanced'])


"knn classifier"
****************

  Base Class Documenation: :class:`sklearn.neighbors.KNeighborsClassifier`

  Param Distributions

	0. "base knn" ::

		n_neighbors: 5

	1. "knn rs" ::

		weights: OrderedDiscrete(possibilities=['uniform', 'distance'])
		n_neighbors: Scalar(shape=(1,), transforms=[ArctanBound(a_max=25, a_min=2)])


"mlp classifier"
****************

  Base Class Documenation: :class:`sklearn.neural_network.MLPClassifier`

  Param Distributions

	0. "base mlp" ::

		defaults only

	1. "mlp rs" ::

		hidden_layer_sizes: Array(shape=(1, 1, 1), transforms=[ArctanBound(a_max=100, a_min=2)])
		activation: OrderedDiscrete(possibilities=['identity', 'logistic', 'tanh', 'relu'])
		alpha: Scalar(shape=(1,), transforms=[ArctanBound(a_max=5, a_min=-2), Exponentiate(base=10, coeff=-1)])
		batch_size: Scalar(shape=(1,), transforms=[ArctanBound(a_max=200, a_min=2)])
		learning_rate: OrderedDiscrete(possibilities=['constant', 'invscaling', 'adaptive'])
		learning_rate_init: Scalar(shape=(1,), transforms=[ArctanBound(a_max=5, a_min=-2), Exponentiate(base=10, coeff=-1)])
		max_iter: Scalar(shape=(1,), transforms=[ArctanBound(a_max=500, a_min=100)])
		beta_1: Scalar(shape=(1,), transforms=[ArctanBound(a_max=0.95, a_min=0.1)])
		beta_2: Scalar(shape=(1,), transforms=[ArctanBound(a_max=0.95, a_min=0.1)])

	2. "mlp rs es" ::

		hidden_layer_sizes: Array(shape=(1, 1, 1), transforms=[ArctanBound(a_max=100, a_min=2)])
		activation: OrderedDiscrete(possibilities=['identity', 'logistic', 'tanh', 'relu'])
		alpha: Scalar(shape=(1,), transforms=[ArctanBound(a_max=5, a_min=-2), Exponentiate(base=10, coeff=-1)])
		batch_size: Scalar(shape=(1,), transforms=[ArctanBound(a_max=200, a_min=2)])
		learning_rate: OrderedDiscrete(possibilities=['constant', 'invscaling', 'adaptive'])
		learning_rate_init: Scalar(shape=(1,), transforms=[ArctanBound(a_max=5, a_min=-2), Exponentiate(base=10, coeff=-1)])
		max_iter: Scalar(shape=(1,), transforms=[ArctanBound(a_max=500, a_min=100)])
		beta_1: Scalar(shape=(1,), transforms=[ArctanBound(a_max=0.95, a_min=0.1)])
		beta_2: Scalar(shape=(1,), transforms=[ArctanBound(a_max=0.95, a_min=0.1)])
		early_stopping: True
		n_iter_no_change: Scalar(shape=(1,), transforms=[ArctanBound(a_max=50, a_min=5)])

	3. "mlp layers search" ::

		hidden_layer_sizes: Array(shape=(1, 1, 1), transforms=[ArctanBound(a_max=100, a_min=2)])


"random forest classifier"
**************************

  Base Class Documenation: :class:`sklearn.ensemble.RandomForestClassifier`

  Param Distributions

	0. "base rf" ::

		n_estimators: 100

	1. "rf classifier rs" ::

		n_estimators: Scalar(shape=(1,), transforms=[ArctanBound(a_max=500, a_min=3)])
		max_depth: Scalar(shape=(1,), transforms=[ArctanBound(a_max=200, a_min=2)])
		max_features: Scalar(shape=(1,), transforms=[ArctanBound(a_max=1, a_min=0)])
		min_samples_split: Scalar(shape=(1,), transforms=[ArctanBound(a_max=1, a_min=0)])
		bootstrap: True
		class_weight: OrderedDiscrete(possibilities=[None, 'balanced'])



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

"brier"
*******

  Base Func Documenation: :func:`sklearn.metrics.brier_score_loss`

"f1"
****

  Base Func Documenation: :func:`sklearn.metrics.f1_score`

"hamming"
*********

  Base Func Documenation: :func:`sklearn.metrics.hamming_loss`

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

"max error"
***********

  Base Func Documenation: :func:`sklearn.metrics.max_error`

"mean absolute error"
*********************

  Base Func Documenation: :func:`sklearn.metrics.mean_absolute_error`

"mean squared error"
********************

  Base Func Documenation: :func:`sklearn.metrics.mean_squared_error`

"mean squared log error"
************************

  Base Func Documenation: :func:`sklearn.metrics.mean_squared_log_error`

"median absolute error"
***********************

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

"hamming"
*********

  Base Func Documenation: :func:`sklearn.metrics.hamming_loss`

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

"hamming"
*********

  Base Func Documenation: :func:`sklearn.metrics.hamming_loss`

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


"robust"
********

  Base Class Documenation: :class:`sklearn.preprocessing.RobustScaler`

  Param Distributions

	0. "base robust" ::

		quantile_range: (5, 95)

	1. "robust gs" ::

		quantile_range: OrderedDiscrete(possibilities=[(1, 99), (5, 95), (10, 90), (15, 85), (20, 80), (25, 75), (30, 70), (35, 65), (40, 60)])


"power"
*******

  Base Class Documenation: :class:`sklearn.preprocessing.PowerTransformer`

  Param Distributions

	0. "base power" ::

		method: yeo-johnson
		standardize: True



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

  Base Class Documenation: :class:`ABCD_ML.Feature_Selectors.RFE`

  Param Distributions

	0. "base rfe" ::

		n_features_to_select: None

	1. "rfe num feats rs" ::

		n_features_to_select: Scalar(shape=(1,), transforms=[ArctanBound(a_max=0.99, a_min=0.01)])


"univariate selection classification"
*************************************

  Base Class Documenation: :class:`sklearn.feature_selection.SelectPercentile`

  Param Distributions

	0. "base univar fs classifier" ::

		score_func: [<function f_classif at 0x7f74d9a26950>]
		percentile: [50]

	1. "univar fs classifier rs" ::

		score_func: [<function f_classif at 0x7f74d9a26950>]
		percentile: Scalar(shape=(1,), transforms=[ArctanBound(a_max=99, a_min=1)])


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

  Base Class Documenation: :class:`ABCD_ML.Feature_Selectors.RFE`

  Param Distributions

	0. "base rfe" ::

		n_features_to_select: None

	1. "rfe num feats rs" ::

		n_features_to_select: Scalar(shape=(1,), transforms=[ArctanBound(a_max=0.99, a_min=0.01)])


"univariate selection regression"
*********************************

  Base Class Documenation: :class:`sklearn.feature_selection.SelectPercentile`

  Param Distributions

	0. "base univar fs regression" ::

		score_func: <function f_regression at 0x7f74d9a26ae8>
		percentile: 50

	1. "univar fs regression rs" ::

		score_func: <function f_regression at 0x7f74d9a26ae8>
		percentile: Scalar(shape=(1,), transforms=[ArctanBound(a_max=99, a_min=1)])


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

  Base Class Documenation: :class:`ABCD_ML.Feature_Selectors.RFE`

  Param Distributions

	0. "base rfe" ::

		n_features_to_select: None

	1. "rfe num feats rs" ::

		n_features_to_select: Scalar(shape=(1,), transforms=[ArctanBound(a_max=0.99, a_min=0.01)])


"univariate selection classification"
*************************************

  Base Class Documenation: :class:`sklearn.feature_selection.SelectPercentile`

  Param Distributions

	0. "base univar fs classifier" ::

		score_func: [<function f_classif at 0x7f74d9a26950>]
		percentile: [50]

	1. "univar fs classifier rs" ::

		score_func: [<function f_classif at 0x7f74d9a26950>]
		percentile: Scalar(shape=(1,), transforms=[ArctanBound(a_max=99, a_min=1)])


"variance threshold"
********************

  Base Class Documenation: :class:`sklearn.feature_selection.VarianceThreshold`

  Param Distributions

	0. "default" ::

		defaults only



multilabel
==========
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

  Base Class Documenation: :class:`sklearn.ensemble.bagging.BaggingClassifier`

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



regression
==========
"bagging regressor"
*******************

  Base Class Documenation: :class:`sklearn.ensemble.bagging.BaggingRegressor`

  Param Distributions

	0. "single default" ::

		needs_split: False
		single_estimator: True



categorical
===========
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

  Base Class Documenation: :class:`sklearn.ensemble.bagging.BaggingClassifier`

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



multilabel
==========

