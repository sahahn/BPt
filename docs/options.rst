.. _Models:
 
******
Models
******

Different base obj choices for the :class:`Model<BPt.Model>` are shown below
The exact str indicator, as passed to the `obj` param is represented by the sub-heading (within "")
The avaliable models are further broken down by which can workwith different problem_types.
Additionally, a link to the original models documentation as well as the implemented parameter distributions are shown.

binary
======
"dt classifier"
***************

  Base Class Documenation: :class:`sklearn.tree.DecisionTreeClassifier`

  Param Distributions

	0. "default" ::

		defaults only

	1. "dt classifier dist" ::

		max_depth: ng.p.Scalar(lower=1, upper=30).set_integer_casting()
		min_samples_split: ng.p.Scalar(lower=2, upper=50).set_integer_casting()
		class_weight: ng.p.TransitionChoice([None, 'balanced'])


"elastic net logistic"
**********************

  Base Class Documenation: :class:`sklearn.linear_model.LogisticRegression`

  Param Distributions

	0. "base elastic" ::

		max_iter: 1000
		multi_class: 'auto'
		penalty: 'elasticnet'
		class_weight: None
		solver: 'saga'
		l1_ratio: .5

	1. "elastic classifier" ::

		max_iter: 1000
		multi_class: 'auto'
		penalty: 'elasticnet'
		class_weight: ng.p.TransitionChoice([None, 'balanced'])
		solver: 'saga'
		l1_ratio: ng.p.Scalar(lower=.01, upper=1)
		C: ng.p.Log(lower=1e-5, upper=1e5)

	2. "elastic clf v2" ::

		max_iter: 1000
		multi_class: 'auto'
		penalty: 'elasticnet'
		class_weight: ng.p.TransitionChoice([None, 'balanced'])
		solver: 'saga'
		l1_ratio: ng.p.Scalar(lower=.01, upper=1)
		C: ng.p.Log(lower=1e-2, upper=1e5)

	3. "elastic classifier extra" ::

		max_iter: ng.p.Scalar(lower=1000, upper=10000).set_integer_casting()
		multi_class: 'auto'
		penalty: 'elasticnet'
		class_weight: ng.p.TransitionChoice([None, 'balanced'])
		solver: 'saga'
		l1_ratio: ng.p.Scalar(lower=.01, upper=1)
		C: ng.p.Log(lower=1e-5, upper=1e5)
		tol: ng.p.Log(lower=1e-6, upper=.01)


"et classifier"
***************

  Base Class Documenation: :class:`sklearn.ensemble.ExtraTreesClassifier`

  Param Distributions

	0. "default" ::

		defaults only


"gaussian nb"
*************

  Base Class Documenation: :class:`sklearn.naive_bayes.GaussianNB`

  Param Distributions

	0. "base gnb" ::

		var_smoothing: 1e-9


"gb classifier"
***************

  Base Class Documenation: :class:`sklearn.ensemble.GradientBoostingClassifier`

  Param Distributions

	0. "default" ::

		defaults only


"gp classifier"
***************

  Base Class Documenation: :class:`sklearn.gaussian_process.GaussianProcessClassifier`

  Param Distributions

	0. "base gp classifier" ::

		n_restarts_optimizer: 5


"hgb classifier"
****************

  Base Class Documenation: :class:`sklearn.ensemble.gradient_boosting.HistGradientBoostingClassifier`

  Param Distributions

	0. "default" ::

		defaults only


"knn classifier"
****************

  Base Class Documenation: :class:`sklearn.neighbors.KNeighborsClassifier`

  Param Distributions

	0. "base knn" ::

		n_neighbors: 5

	1. "knn dist" ::

		weights: ng.p.TransitionChoice(['uniform', 'distance'])
		n_neighbors: ng.p.Scalar(lower=2, upper=25).set_integer_casting()


"lasso logistic"
****************

  Base Class Documenation: :class:`sklearn.linear_model.LogisticRegression`

  Param Distributions

	0. "base lasso" ::

		max_iter: 1000
		multi_class: 'auto'
		penalty: 'l1'
		class_weight: None
		solver: 'liblinear'

	1. "lasso C" ::

		max_iter: 1000
		multi_class: 'auto'
		penalty: 'l1'
		class_weight: ng.p.TransitionChoice([None, 'balanced'])
		solver: 'liblinear'
		C: ng.p.Log(lower=1e-5, upper=1e3)

	2. "lasso C extra" ::

		max_iter: ng.p.Scalar(lower=1000, upper=10000).set_integer_casting()
		multi_class: 'auto'
		penalty: 'l1'
		class_weight: ng.p.TransitionChoice([None, 'balanced'])
		solver: 'liblinear'
		C: ng.p.Log(lower=1e-5, upper=1e3)
		tol: ng.p.Log(lower=1e-6, upper=.01)


"light gbm classifier"
**********************

  Base Class Documenation: :class:`lightgbm.LGBMClassifier`

  Param Distributions

	0. "base lgbm" ::

		silent: True

	1. "lgbm classifier dist1" ::

		silent: True
		boosting_type: ng.p.TransitionChoice(['gbdt', 'dart', 'goss'])
		n_estimators: ng.p.Scalar(init=100, lower=3, upper=500).set_integer_casting()
		num_leaves: ng.p.Scalar(init=20, lower=6, upper=80).set_integer_casting()
		min_child_samples: ng.p.Scalar(lower=10, upper=500).set_integer_casting()
		min_child_weight: ng.p.Log(lower=1e-5, upper=1e4)
		subsample: ng.p.Scalar(lower=.3, upper=.95)
		colsample_bytree: ng.p.Scalar(lower=.3, upper=.95)
		reg_alpha: ng.p.TransitionChoice([0, ng.p.Log(lower=1e-5, upper=1)])
		reg_lambda: ng.p.TransitionChoice([0, ng.p.Log(lower=1e-5, upper=1)])
		class_weight: ng.p.TransitionChoice([None, 'balanced'])

	2. "lgbm classifier dist2" ::

		silent: True
		lambda_l2: 0.001
		boosting_type: ng.p.TransitionChoice(['gbdt', 'dart'])
		min_child_samples: ng.p.TransitionChoice([1, 5, 7, 10, 15, 20, 35, 50, 100, 200, 500, 1000])
		num_leaves: ng.p.TransitionChoice([2, 4, 7, 10, 15, 20, 25, 30, 35, 40, 50, 65, 80, 100, 125, 150, 200, 250])
		colsample_bytree: ng.p.TransitionChoice([0.7, 0.9, 1.0])
		subsample: ng.p.Scalar(lower=.3, upper=1)
		learning_rate: ng.p.TransitionChoice([0.01, 0.05, 0.1])
		n_estimators: ng.p.TransitionChoice([5, 20, 35, 50, 75, 100, 150, 200, 350, 500, 750, 1000])
		class_weight: ng.p.TransitionChoice([None, 'balanced'])


"linear svm classifier"
***********************

  Base Class Documenation: :class:`sklearn.svm.LinearSVC`

  Param Distributions

	0. "base linear svc" ::

		max_iter: 1000

	1. "linear svc dist" ::

		max_iter: 1000
		C: ng.p.Log(lower=1e-4, upper=1e4)
		class_weight: ng.p.TransitionChoice([None, 'balanced'])


"logistic"
**********

  Base Class Documenation: :class:`sklearn.linear_model.LogisticRegression`

  Param Distributions

	0. "base logistic" ::

		max_iter: 1000
		multi_class: 'auto'
		penalty: 'none'
		class_weight: None
		solver: 'lbfgs'


"mlp classifier"
****************

  Base Class Documenation: :class:`BPt.extensions.MLP.MLPClassifier_Wrapper`

  Param Distributions

	0. "default" ::

		defaults only

	1. "mlp dist 3 layer" ::

		hidden_layer_sizes: ng.p.Array(init=(100, 100, 100)).set_mutation(sigma=50).set_bounds(lower=1, upper=300).set_integer_casting()
		activation: ng.p.TransitionChoice(['identity', 'logistic', 'tanh', 'relu'])
		alpha: ng.p.Log(lower=1e-5, upper=1e2)
		batch_size: ng.p.TransitionChoice(['auto', ng.p.Scalar(init=200, lower=50, upper=400).set_integer_casting()])
		learning_rate: ng.p.TransitionChoice(['constant', 'invscaling', 'adaptive'])
		learning_rate_init: ng.p.Log(lower=1e-5, upper=1e2)
		max_iter: ng.p.Scalar(init=200, lower=100, upper=1000).set_integer_casting()
		beta_1: ng.p.Scalar(init=.9, lower=.1, upper=.99)
		beta_2: ng.p.Scalar(init=.999, lower=.1, upper=.9999)

	2. "mlp dist es 3 layer" ::

		hidden_layer_sizes: ng.p.Scalar(init=100, lower=2, upper=300).set_integer_casting()
		activation: ng.p.TransitionChoice(['identity', 'logistic', 'tanh', 'relu'])
		alpha: ng.p.Log(lower=1e-5, upper=1e2)
		batch_size: ng.p.TransitionChoice(['auto', ng.p.Scalar(init=200, lower=50, upper=400).set_integer_casting()])
		learning_rate: ng.p.TransitionChoice(['constant', 'invscaling', 'adaptive'])
		learning_rate_init: ng.p.Log(lower=1e-5, upper=1e2)
		max_iter: ng.p.Scalar(init=200, lower=100, upper=1000).set_integer_casting()
		beta_1: ng.p.Scalar(init=.9, lower=.1, upper=.99)
		beta_2: ng.p.Scalar(init=.999, lower=.1, upper=.9999)
		early_stopping: True
		n_iter_no_change: ng.p.Scalar(lower=5, upper=50)

	3. "mlp dist 2 layer" ::

		hidden_layer_sizes: ng.p.Array(init=(100, 100)).set_mutation(sigma=50).set_bounds(lower=1, upper=300).set_integer_casting()
		activation: ng.p.TransitionChoice(['identity', 'logistic', 'tanh', 'relu'])
		alpha: ng.p.Log(lower=1e-5, upper=1e2)
		batch_size: ng.p.TransitionChoice(['auto', ng.p.Scalar(init=200, lower=50, upper=400).set_integer_casting()])
		learning_rate: ng.p.TransitionChoice(['constant', 'invscaling', 'adaptive'])
		learning_rate_init: ng.p.Log(lower=1e-5, upper=1e2)
		max_iter: ng.p.Scalar(init=200, lower=100, upper=1000).set_integer_casting()
		beta_1: ng.p.Scalar(init=.9, lower=.1, upper=.99)
		beta_2: ng.p.Scalar(init=.999, lower=.1, upper=.9999)

	4. "mlp dist es 2 layer" ::

		hidden_layer_sizes: ng.p.Scalar(init=100, lower=2, upper=300).set_integer_casting()
		activation: ng.p.TransitionChoice(['identity', 'logistic', 'tanh', 'relu'])
		alpha: ng.p.Log(lower=1e-5, upper=1e2)
		batch_size: ng.p.TransitionChoice(['auto', ng.p.Scalar(init=200, lower=50, upper=400).set_integer_casting()])
		learning_rate: ng.p.TransitionChoice(['constant', 'invscaling', 'adaptive'])
		learning_rate_init: ng.p.Log(lower=1e-5, upper=1e2)
		max_iter: ng.p.Scalar(init=200, lower=100, upper=1000).set_integer_casting()
		beta_1: ng.p.Scalar(init=.9, lower=.1, upper=.99)
		beta_2: ng.p.Scalar(init=.999, lower=.1, upper=.9999)
		early_stopping: True
		n_iter_no_change: ng.p.Scalar(lower=5, upper=50)

	5. "mlp dist 1 layer" ::

		hidden_layer_sizes: ng.p.Scalar(init=100, lower=2, upper=300).set_integer_casting()
		activation: ng.p.TransitionChoice(['identity', 'logistic', 'tanh', 'relu'])
		alpha: ng.p.Log(lower=1e-5, upper=1e2)
		batch_size: ng.p.TransitionChoice(['auto', ng.p.Scalar(init=200, lower=50, upper=400).set_integer_casting()])
		learning_rate: ng.p.TransitionChoice(['constant', 'invscaling', 'adaptive'])
		learning_rate_init: ng.p.Log(lower=1e-5, upper=1e2)
		max_iter: ng.p.Scalar(init=200, lower=100, upper=1000).set_integer_casting()
		beta_1: ng.p.Scalar(init=.9, lower=.1, upper=.99)
		beta_2: ng.p.Scalar(init=.999, lower=.1, upper=.9999)

	6. "mlp dist es 1 layer" ::

		hidden_layer_sizes: ng.p.Scalar(init=100, lower=2, upper=300).set_integer_casting()
		activation: ng.p.TransitionChoice(['identity', 'logistic', 'tanh', 'relu'])
		alpha: ng.p.Log(lower=1e-5, upper=1e2)
		batch_size: ng.p.TransitionChoice(['auto', ng.p.Scalar(init=200, lower=50, upper=400).set_integer_casting()])
		learning_rate: ng.p.TransitionChoice(['constant', 'invscaling', 'adaptive'])
		learning_rate_init: ng.p.Log(lower=1e-5, upper=1e2)
		max_iter: ng.p.Scalar(init=200, lower=100, upper=1000).set_integer_casting()
		beta_1: ng.p.Scalar(init=.9, lower=.1, upper=.99)
		beta_2: ng.p.Scalar(init=.999, lower=.1, upper=.9999)
		early_stopping: True
		n_iter_no_change: ng.p.Scalar(lower=5, upper=50)


"pa classifier"
***************

  Base Class Documenation: :class:`sklearn.linear_model.PassiveAggressiveClassifier`

  Param Distributions

	0. "default" ::

		defaults only


"random forest classifier"
**************************

  Base Class Documenation: :class:`sklearn.ensemble.RandomForestClassifier`

  Param Distributions

	0. "base rf regressor" ::

		n_estimators: 100

	1. "rf classifier dist" ::

		n_estimators: ng.p.Scalar(init=100, lower=3, upper=500).set_integer_casting()
		max_depth: ng.p.TransitionChoice([None, ng.p.Scalar(init=25, lower=2, upper=200).set_integer_casting()])
		max_features: ng.p.Scalar(lower=.1, upper=1.0)
		min_samples_split: ng.p.Scalar(lower=.1, upper=1.0)
		bootstrap: True
		class_weight: ng.p.TransitionChoice([None, 'balanced'])


"ridge logistic"
****************

  Base Class Documenation: :class:`sklearn.linear_model.LogisticRegression`

  Param Distributions

	0. "base ridge" ::

		max_iter: 1000
		penalty: 'l2'
		solver: 'saga'

	1. "ridge C" ::

		max_iter: 1000
		solver: 'saga'
		C: ng.p.Log(lower=1e-5, upper=1e3)
		class_weight: ng.p.TransitionChoice([None, 'balanced'])

	2. "ridge C extra" ::

		max_iter: ng.p.Scalar(lower=1000, upper=10000).set_integer_casting()
		solver: 'saga'
		C: ng.p.Log(lower=1e-5, upper=1e3)
		class_weight: ng.p.TransitionChoice([None, 'balanced'])
		tol: ng.p.Log(lower=1e-6, upper=.01)


"sgd classifier"
****************

  Base Class Documenation: :class:`sklearn.linear_model.SGDClassifier`

  Param Distributions

	0. "base sgd" ::

		loss: 'hinge'

	1. "sgd classifier" ::

		loss: ng.p.TransitionChoice(['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'])
		penalty: ng.p.TransitionChoice(['l2', 'l1', 'elasticnet'])
		alpha: ng.p.Log(lower=1e-5, upper=1e2)
		l1_ratio: ng.p.Scalar(lower=0, upper=1)
		max_iter: 1000
		learning_rate: ng.p.TransitionChoice(['optimal', 'invscaling', 'adaptive', 'constant'])
		eta0: ng.p.Log(lower=1e-6, upper=1e3)
		power_t: ng.p.Scalar(lower=.1, upper=.9)
		early_stopping: ng.p.TransitionChoice([False, True])
		validation_fraction: ng.p.Scalar(lower=.05, upper=.5)
		n_iter_no_change: ng.p.TransitionChoice(np.arange(2, 20))
		class_weight: ng.p.TransitionChoice([None, 'balanced'])


"svm classifier"
****************

  Base Class Documenation: :class:`sklearn.svm.SVC`

  Param Distributions

	0. "base svm classifier" ::

		kernel: 'rbf'
		gamma: 'scale'
		probability: True

	1. "svm classifier dist" ::

		kernel: 'rbf'
		gamma: ng.p.Log(lower=1e-6, upper=1)
		C: ng.p.Log(lower=1e-4, upper=1e4)
		probability: True
		class_weight: ng.p.TransitionChoice([None, 'balanced'])


"xgb classifier"
****************

  Base Class Documenation: :class:`xgboost.XGBClassifier`

  Param Distributions

	0. "base xgb classifier" ::

		verbosity: 0
		objective: 'binary:logistic'

	1. "xgb classifier dist1" ::

		verbosity: 0
		objective: 'binary:logistic'
		n_estimators: ng.p.Scalar(init=100, lower=3, upper=500).set_integer_casting()
		min_child_weight: ng.p.Log(lower=1e-5, upper=1e4)
		subsample: ng.p.Scalar(lower=.3, upper=.95)
		colsample_bytree: ng.p.Scalar(lower=.3, upper=.95)
		reg_alpha: ng.p.TransitionChoice([0, ng.p.Log(lower=1e-5, upper=1)])
		reg_lambda: ng.p.TransitionChoice([0, ng.p.Log(lower=1e-5, upper=1)])

	2. "xgb classifier dist2" ::

		verbosity: 0
		objective: 'binary:logistic'
		max_depth: ng.p.TransitionChoice([None, ng.p.Scalar(init=25, lower=2, upper=200).set_integer_casting()])
		learning_rate: ng.p.Scalar(lower=.01, upper=.5)
		n_estimators: ng.p.Scalar(lower=3, upper=500).set_integer_casting()
		min_child_weight: ng.p.TransitionChoice([1, 5, 10, 50])
		subsample: ng.p.Scalar(lower=.5, upper=1)
		colsample_bytree: ng.p.Scalar(lower=.4, upper=.95)

	3. "xgb classifier dist3" ::

		verbosity: 0
		objective: 'binary:logistic'
		learning_rare: ng.p.Scalar(lower=.005, upper=.3)
		min_child_weight: ng.p.Scalar(lower=.5, upper=10)
		max_depth: ng.p.TransitionChoice(np.arange(3, 10))
		subsample: ng.p.Scalar(lower=.5, upper=1)
		colsample_bytree: ng.p.Scalar(lower=.5, upper=1)
		reg_alpha: ng.p.Log(lower=.00001, upper=1)



regression
==========
"ard regressor"
***************

  Base Class Documenation: :class:`sklearn.linear_model.ARDRegression`

  Param Distributions

	0. "default" ::

		defaults only


"bayesian ridge regressor"
**************************

  Base Class Documenation: :class:`sklearn.linear_model.BayesianRidge`

  Param Distributions

	0. "default" ::

		defaults only


"dt regressor"
**************

  Base Class Documenation: :class:`sklearn.tree.DecisionTreeRegressor`

  Param Distributions

	0. "default" ::

		defaults only

	1. "dt dist" ::

		max_depth: ng.p.Scalar(lower=1, upper=30).set_integer_casting()
		min_samples_split: ng.p.Scalar(lower=2, upper=50).set_integer_casting()


"elastic net regressor"
***********************

  Base Class Documenation: :class:`sklearn.linear_model.ElasticNet`

  Param Distributions

	0. "base elastic net" ::

		max_iter: 1000

	1. "elastic regression" ::

		max_iter: 1000
		alpha: ng.p.Log(lower=1e-5, upper=1e5)
		l1_ratio: ng.p.Scalar(lower=.01, upper=1)

	2. "elastic regression extra" ::

		max_iter: ng.p.Scalar(lower=1000, upper=10000).set_integer_casting()
		alpha: ng.p.Log(lower=1e-5, upper=1e5)
		l1_ratio: ng.p.Scalar(lower=.01, upper=1)
		tol: ng.p.Log(lower=1e-6, upper=.01)


"et regressor"
**************

  Base Class Documenation: :class:`sklearn.ensemble.ExtraTreesRegressor`

  Param Distributions

	0. "default" ::

		defaults only


"gb regressor"
**************

  Base Class Documenation: :class:`sklearn.ensemble.GradientBoostingRegressor`

  Param Distributions

	0. "default" ::

		defaults only


"gp regressor"
**************

  Base Class Documenation: :class:`sklearn.gaussian_process.GaussianProcessRegressor`

  Param Distributions

	0. "base gp regressor" ::

		n_restarts_optimizer: 5
		normalize_y: True


"hgb regressor"
***************

  Base Class Documenation: :class:`sklearn.ensemble.gradient_boosting.HistGradientBoostingRegressor`

  Param Distributions

	0. "default" ::

		defaults only


"knn regressor"
***************

  Base Class Documenation: :class:`sklearn.neighbors.KNeighborsRegressor`

  Param Distributions

	0. "base knn regression" ::

		n_neighbors: 5

	1. "knn dist regression" ::

		weights: ng.p.TransitionChoice(['uniform', 'distance'])
		n_neighbors: ng.p.Scalar(lower=2, upper=25).set_integer_casting()


"lasso regressor"
*****************

  Base Class Documenation: :class:`sklearn.linear_model.Lasso`

  Param Distributions

	0. "base lasso regressor" ::

		max_iter: 1000

	1. "lasso regressor dist" ::

		max_iter: 1000
		alpha: ng.p.Log(lower=1e-5, upper=1e5)


"light gbm regressor"
*********************

  Base Class Documenation: :class:`lightgbm.LGBMRegressor`

  Param Distributions

	0. "base lgbm" ::

		silent: True

	1. "lgbm dist1" ::

		silent: True
		boosting_type: ng.p.TransitionChoice(['gbdt', 'dart', 'goss'])
		n_estimators: ng.p.Scalar(init=100, lower=3, upper=500).set_integer_casting()
		num_leaves: ng.p.Scalar(init=20, lower=6, upper=80).set_integer_casting()
		min_child_samples: ng.p.Scalar(lower=10, upper=500).set_integer_casting()
		min_child_weight: ng.p.Log(lower=1e-5, upper=1e4)
		subsample: ng.p.Scalar(lower=.3, upper=.95)
		colsample_bytree: ng.p.Scalar(lower=.3, upper=.95)
		reg_alpha: ng.p.TransitionChoice([0, ng.p.Log(lower=1e-5, upper=1)])
		reg_lambda: ng.p.TransitionChoice([0, ng.p.Log(lower=1e-5, upper=1)])

	2. "lgbm dist2" ::

		silent: True
		lambda_l2: 0.001
		boosting_type: ng.p.TransitionChoice(['gbdt', 'dart'])
		min_child_samples: ng.p.TransitionChoice([1, 5, 7, 10, 15, 20, 35, 50, 100, 200, 500, 1000])
		num_leaves: ng.p.TransitionChoice([2, 4, 7, 10, 15, 20, 25, 30, 35, 40, 50, 65, 80, 100, 125, 150, 200, 250])
		colsample_bytree: ng.p.TransitionChoice([0.7, 0.9, 1.0])
		subsample: ng.p.Scalar(lower=.3, upper=1)
		learning_rate: ng.p.TransitionChoice([0.01, 0.05, 0.1])
		n_estimators: ng.p.TransitionChoice([5, 20, 35, 50, 75, 100, 150, 200, 350, 500, 750, 1000])


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

		loss: 'epsilon_insensitive'
		max_iter: 1000

	1. "linear svr dist" ::

		loss: 'epsilon_insensitive'
		max_iter: 1000
		C: ng.p.Log(lower=1e-4, upper=1e4)


"mlp regressor"
***************

  Base Class Documenation: :class:`BPt.extensions.MLP.MLPRegressor_Wrapper`

  Param Distributions

	0. "default" ::

		defaults only

	1. "mlp dist 3 layer" ::

		hidden_layer_sizes: ng.p.Array(init=(100, 100, 100)).set_mutation(sigma=50).set_bounds(lower=1, upper=300).set_integer_casting()
		activation: ng.p.TransitionChoice(['identity', 'logistic', 'tanh', 'relu'])
		alpha: ng.p.Log(lower=1e-5, upper=1e2)
		batch_size: ng.p.TransitionChoice(['auto', ng.p.Scalar(init=200, lower=50, upper=400).set_integer_casting()])
		learning_rate: ng.p.TransitionChoice(['constant', 'invscaling', 'adaptive'])
		learning_rate_init: ng.p.Log(lower=1e-5, upper=1e2)
		max_iter: ng.p.Scalar(init=200, lower=100, upper=1000).set_integer_casting()
		beta_1: ng.p.Scalar(init=.9, lower=.1, upper=.99)
		beta_2: ng.p.Scalar(init=.999, lower=.1, upper=.9999)

	2. "mlp dist es 3 layer" ::

		hidden_layer_sizes: ng.p.Scalar(init=100, lower=2, upper=300).set_integer_casting()
		activation: ng.p.TransitionChoice(['identity', 'logistic', 'tanh', 'relu'])
		alpha: ng.p.Log(lower=1e-5, upper=1e2)
		batch_size: ng.p.TransitionChoice(['auto', ng.p.Scalar(init=200, lower=50, upper=400).set_integer_casting()])
		learning_rate: ng.p.TransitionChoice(['constant', 'invscaling', 'adaptive'])
		learning_rate_init: ng.p.Log(lower=1e-5, upper=1e2)
		max_iter: ng.p.Scalar(init=200, lower=100, upper=1000).set_integer_casting()
		beta_1: ng.p.Scalar(init=.9, lower=.1, upper=.99)
		beta_2: ng.p.Scalar(init=.999, lower=.1, upper=.9999)
		early_stopping: True
		n_iter_no_change: ng.p.Scalar(lower=5, upper=50)

	3. "mlp dist 2 layer" ::

		hidden_layer_sizes: ng.p.Array(init=(100, 100)).set_mutation(sigma=50).set_bounds(lower=1, upper=300).set_integer_casting()
		activation: ng.p.TransitionChoice(['identity', 'logistic', 'tanh', 'relu'])
		alpha: ng.p.Log(lower=1e-5, upper=1e2)
		batch_size: ng.p.TransitionChoice(['auto', ng.p.Scalar(init=200, lower=50, upper=400).set_integer_casting()])
		learning_rate: ng.p.TransitionChoice(['constant', 'invscaling', 'adaptive'])
		learning_rate_init: ng.p.Log(lower=1e-5, upper=1e2)
		max_iter: ng.p.Scalar(init=200, lower=100, upper=1000).set_integer_casting()
		beta_1: ng.p.Scalar(init=.9, lower=.1, upper=.99)
		beta_2: ng.p.Scalar(init=.999, lower=.1, upper=.9999)

	4. "mlp dist es 2 layer" ::

		hidden_layer_sizes: ng.p.Scalar(init=100, lower=2, upper=300).set_integer_casting()
		activation: ng.p.TransitionChoice(['identity', 'logistic', 'tanh', 'relu'])
		alpha: ng.p.Log(lower=1e-5, upper=1e2)
		batch_size: ng.p.TransitionChoice(['auto', ng.p.Scalar(init=200, lower=50, upper=400).set_integer_casting()])
		learning_rate: ng.p.TransitionChoice(['constant', 'invscaling', 'adaptive'])
		learning_rate_init: ng.p.Log(lower=1e-5, upper=1e2)
		max_iter: ng.p.Scalar(init=200, lower=100, upper=1000).set_integer_casting()
		beta_1: ng.p.Scalar(init=.9, lower=.1, upper=.99)
		beta_2: ng.p.Scalar(init=.999, lower=.1, upper=.9999)
		early_stopping: True
		n_iter_no_change: ng.p.Scalar(lower=5, upper=50)

	5. "mlp dist 1 layer" ::

		hidden_layer_sizes: ng.p.Scalar(init=100, lower=2, upper=300).set_integer_casting()
		activation: ng.p.TransitionChoice(['identity', 'logistic', 'tanh', 'relu'])
		alpha: ng.p.Log(lower=1e-5, upper=1e2)
		batch_size: ng.p.TransitionChoice(['auto', ng.p.Scalar(init=200, lower=50, upper=400).set_integer_casting()])
		learning_rate: ng.p.TransitionChoice(['constant', 'invscaling', 'adaptive'])
		learning_rate_init: ng.p.Log(lower=1e-5, upper=1e2)
		max_iter: ng.p.Scalar(init=200, lower=100, upper=1000).set_integer_casting()
		beta_1: ng.p.Scalar(init=.9, lower=.1, upper=.99)
		beta_2: ng.p.Scalar(init=.999, lower=.1, upper=.9999)

	6. "mlp dist es 1 layer" ::

		hidden_layer_sizes: ng.p.Scalar(init=100, lower=2, upper=300).set_integer_casting()
		activation: ng.p.TransitionChoice(['identity', 'logistic', 'tanh', 'relu'])
		alpha: ng.p.Log(lower=1e-5, upper=1e2)
		batch_size: ng.p.TransitionChoice(['auto', ng.p.Scalar(init=200, lower=50, upper=400).set_integer_casting()])
		learning_rate: ng.p.TransitionChoice(['constant', 'invscaling', 'adaptive'])
		learning_rate_init: ng.p.Log(lower=1e-5, upper=1e2)
		max_iter: ng.p.Scalar(init=200, lower=100, upper=1000).set_integer_casting()
		beta_1: ng.p.Scalar(init=.9, lower=.1, upper=.99)
		beta_2: ng.p.Scalar(init=.999, lower=.1, upper=.9999)
		early_stopping: True
		n_iter_no_change: ng.p.Scalar(lower=5, upper=50)


"random forest regressor"
*************************

  Base Class Documenation: :class:`sklearn.ensemble.RandomForestRegressor`

  Param Distributions

	0. "base rf" ::

		n_estimators: 100

	1. "rf dist" ::

		n_estimators: ng.p.Scalar(init=100, lower=3, upper=500).set_integer_casting()
		max_depth: ng.p.TransitionChoice([None, ng.p.Scalar(init=25, lower=2, upper=200).set_integer_casting()])
		max_features: ng.p.Scalar(lower=.1, upper=1.0)
		min_samples_split: ng.p.Scalar(lower=.1, upper=1.0)
		bootstrap: True


"ridge regressor"
*****************

  Base Class Documenation: :class:`sklearn.linear_model.Ridge`

  Param Distributions

	0. "base ridge regressor" ::

		max_iter: 1000
		solver: 'lsqr'

	1. "ridge regressor dist" ::

		max_iter: 1000
		solver: 'lsqr'
		alpha: ng.p.Log(lower=1e-3, upper=1e5)


"svm regressor"
***************

  Base Class Documenation: :class:`sklearn.svm.SVR`

  Param Distributions

	0. "base svm" ::

		kernel: 'rbf'
		gamma: 'scale'

	1. "svm dist" ::

		kernel: 'rbf'
		gamma: ng.p.Log(lower=1e-6, upper=1)
		C: ng.p.Log(lower=1e-4, upper=1e4)


"tweedie regressor"
*******************

  Base Class Documenation: :class:`sklearn.linear_model.glm.TweedieRegressor`

  Param Distributions

	0. "default" ::

		defaults only


"xgb regressor"
***************

  Base Class Documenation: :class:`xgboost.XGBRegressor`

  Param Distributions

	0. "base xgb" ::

		verbosity: 0
		objective: 'reg:squarederror'

	1. "xgb dist1" ::

		verbosity: 0
		objective: 'reg:squarederror'
		n_estimators: ng.p.Scalar(init=100, lower=3, upper=500).set_integer_casting()
		min_child_weight: ng.p.Log(lower=1e-5, upper=1e4)
		subsample: ng.p.Scalar(lower=.3, upper=.95)
		colsample_bytree: ng.p.Scalar(lower=.3, upper=.95)
		reg_alpha: ng.p.TransitionChoice([0, ng.p.Log(lower=1e-5, upper=1)])
		reg_lambda: ng.p.TransitionChoice([0, ng.p.Log(lower=1e-5, upper=1)])

	2. "xgb dist2" ::

		verbosity: 0
		objective: 'reg:squarederror'
		max_depth: ng.p.TransitionChoice([None, ng.p.Scalar(init=25, lower=2, upper=200).set_integer_casting()])
		learning_rate: ng.p.Scalar(lower=.01, upper=.5)
		n_estimators: ng.p.Scalar(lower=3, upper=500).set_integer_casting()
		min_child_weight: ng.p.TransitionChoice([1, 5, 10, 50])
		subsample: ng.p.Scalar(lower=.5, upper=1)
		colsample_bytree: ng.p.Scalar(lower=.4, upper=.95)

	3. "xgb dist3" ::

		verbosity: 0
		objective: 'reg:squarederror'
		learning_rare: ng.p.Scalar(lower=.005, upper=.3)
		min_child_weight: ng.p.Scalar(lower=.5, upper=10)
		max_depth: ng.p.TransitionChoice(np.arange(3, 10))
		subsample: ng.p.Scalar(lower=.5, upper=1)
		colsample_bytree: ng.p.Scalar(lower=.5, upper=1)
		reg_alpha: ng.p.Log(lower=.00001, upper=1)



categorical
===========
"dt classifier"
***************

  Base Class Documenation: :class:`sklearn.tree.DecisionTreeClassifier`

  Param Distributions

	0. "default" ::

		defaults only

	1. "dt classifier dist" ::

		max_depth: ng.p.Scalar(lower=1, upper=30).set_integer_casting()
		min_samples_split: ng.p.Scalar(lower=2, upper=50).set_integer_casting()
		class_weight: ng.p.TransitionChoice([None, 'balanced'])


"elastic net logistic"
**********************

  Base Class Documenation: :class:`sklearn.linear_model.LogisticRegression`

  Param Distributions

	0. "base elastic" ::

		max_iter: 1000
		multi_class: 'auto'
		penalty: 'elasticnet'
		class_weight: None
		solver: 'saga'
		l1_ratio: .5

	1. "elastic classifier" ::

		max_iter: 1000
		multi_class: 'auto'
		penalty: 'elasticnet'
		class_weight: ng.p.TransitionChoice([None, 'balanced'])
		solver: 'saga'
		l1_ratio: ng.p.Scalar(lower=.01, upper=1)
		C: ng.p.Log(lower=1e-5, upper=1e5)

	2. "elastic clf v2" ::

		max_iter: 1000
		multi_class: 'auto'
		penalty: 'elasticnet'
		class_weight: ng.p.TransitionChoice([None, 'balanced'])
		solver: 'saga'
		l1_ratio: ng.p.Scalar(lower=.01, upper=1)
		C: ng.p.Log(lower=1e-2, upper=1e5)

	3. "elastic classifier extra" ::

		max_iter: ng.p.Scalar(lower=1000, upper=10000).set_integer_casting()
		multi_class: 'auto'
		penalty: 'elasticnet'
		class_weight: ng.p.TransitionChoice([None, 'balanced'])
		solver: 'saga'
		l1_ratio: ng.p.Scalar(lower=.01, upper=1)
		C: ng.p.Log(lower=1e-5, upper=1e5)
		tol: ng.p.Log(lower=1e-6, upper=.01)


"et classifier"
***************

  Base Class Documenation: :class:`sklearn.ensemble.ExtraTreesClassifier`

  Param Distributions

	0. "default" ::

		defaults only


"gaussian nb"
*************

  Base Class Documenation: :class:`sklearn.naive_bayes.GaussianNB`

  Param Distributions

	0. "base gnb" ::

		var_smoothing: 1e-9


"gb classifier"
***************

  Base Class Documenation: :class:`sklearn.ensemble.GradientBoostingClassifier`

  Param Distributions

	0. "default" ::

		defaults only


"gp classifier"
***************

  Base Class Documenation: :class:`sklearn.gaussian_process.GaussianProcessClassifier`

  Param Distributions

	0. "base gp classifier" ::

		n_restarts_optimizer: 5


"hgb classifier"
****************

  Base Class Documenation: :class:`sklearn.ensemble.gradient_boosting.HistGradientBoostingClassifier`

  Param Distributions

	0. "default" ::

		defaults only


"knn classifier"
****************

  Base Class Documenation: :class:`sklearn.neighbors.KNeighborsClassifier`

  Param Distributions

	0. "base knn" ::

		n_neighbors: 5

	1. "knn dist" ::

		weights: ng.p.TransitionChoice(['uniform', 'distance'])
		n_neighbors: ng.p.Scalar(lower=2, upper=25).set_integer_casting()


"lasso logistic"
****************

  Base Class Documenation: :class:`sklearn.linear_model.LogisticRegression`

  Param Distributions

	0. "base lasso" ::

		max_iter: 1000
		multi_class: 'auto'
		penalty: 'l1'
		class_weight: None
		solver: 'liblinear'

	1. "lasso C" ::

		max_iter: 1000
		multi_class: 'auto'
		penalty: 'l1'
		class_weight: ng.p.TransitionChoice([None, 'balanced'])
		solver: 'liblinear'
		C: ng.p.Log(lower=1e-5, upper=1e3)

	2. "lasso C extra" ::

		max_iter: ng.p.Scalar(lower=1000, upper=10000).set_integer_casting()
		multi_class: 'auto'
		penalty: 'l1'
		class_weight: ng.p.TransitionChoice([None, 'balanced'])
		solver: 'liblinear'
		C: ng.p.Log(lower=1e-5, upper=1e3)
		tol: ng.p.Log(lower=1e-6, upper=.01)


"light gbm classifier"
**********************

  Base Class Documenation: :class:`lightgbm.LGBMClassifier`

  Param Distributions

	0. "base lgbm" ::

		silent: True

	1. "lgbm classifier dist1" ::

		silent: True
		boosting_type: ng.p.TransitionChoice(['gbdt', 'dart', 'goss'])
		n_estimators: ng.p.Scalar(init=100, lower=3, upper=500).set_integer_casting()
		num_leaves: ng.p.Scalar(init=20, lower=6, upper=80).set_integer_casting()
		min_child_samples: ng.p.Scalar(lower=10, upper=500).set_integer_casting()
		min_child_weight: ng.p.Log(lower=1e-5, upper=1e4)
		subsample: ng.p.Scalar(lower=.3, upper=.95)
		colsample_bytree: ng.p.Scalar(lower=.3, upper=.95)
		reg_alpha: ng.p.TransitionChoice([0, ng.p.Log(lower=1e-5, upper=1)])
		reg_lambda: ng.p.TransitionChoice([0, ng.p.Log(lower=1e-5, upper=1)])
		class_weight: ng.p.TransitionChoice([None, 'balanced'])

	2. "lgbm classifier dist2" ::

		silent: True
		lambda_l2: 0.001
		boosting_type: ng.p.TransitionChoice(['gbdt', 'dart'])
		min_child_samples: ng.p.TransitionChoice([1, 5, 7, 10, 15, 20, 35, 50, 100, 200, 500, 1000])
		num_leaves: ng.p.TransitionChoice([2, 4, 7, 10, 15, 20, 25, 30, 35, 40, 50, 65, 80, 100, 125, 150, 200, 250])
		colsample_bytree: ng.p.TransitionChoice([0.7, 0.9, 1.0])
		subsample: ng.p.Scalar(lower=.3, upper=1)
		learning_rate: ng.p.TransitionChoice([0.01, 0.05, 0.1])
		n_estimators: ng.p.TransitionChoice([5, 20, 35, 50, 75, 100, 150, 200, 350, 500, 750, 1000])
		class_weight: ng.p.TransitionChoice([None, 'balanced'])


"linear svm classifier"
***********************

  Base Class Documenation: :class:`sklearn.svm.LinearSVC`

  Param Distributions

	0. "base linear svc" ::

		max_iter: 1000

	1. "linear svc dist" ::

		max_iter: 1000
		C: ng.p.Log(lower=1e-4, upper=1e4)
		class_weight: ng.p.TransitionChoice([None, 'balanced'])


"logistic"
**********

  Base Class Documenation: :class:`sklearn.linear_model.LogisticRegression`

  Param Distributions

	0. "base logistic" ::

		max_iter: 1000
		multi_class: 'auto'
		penalty: 'none'
		class_weight: None
		solver: 'lbfgs'


"mlp classifier"
****************

  Base Class Documenation: :class:`BPt.extensions.MLP.MLPClassifier_Wrapper`

  Param Distributions

	0. "default" ::

		defaults only

	1. "mlp dist 3 layer" ::

		hidden_layer_sizes: ng.p.Array(init=(100, 100, 100)).set_mutation(sigma=50).set_bounds(lower=1, upper=300).set_integer_casting()
		activation: ng.p.TransitionChoice(['identity', 'logistic', 'tanh', 'relu'])
		alpha: ng.p.Log(lower=1e-5, upper=1e2)
		batch_size: ng.p.TransitionChoice(['auto', ng.p.Scalar(init=200, lower=50, upper=400).set_integer_casting()])
		learning_rate: ng.p.TransitionChoice(['constant', 'invscaling', 'adaptive'])
		learning_rate_init: ng.p.Log(lower=1e-5, upper=1e2)
		max_iter: ng.p.Scalar(init=200, lower=100, upper=1000).set_integer_casting()
		beta_1: ng.p.Scalar(init=.9, lower=.1, upper=.99)
		beta_2: ng.p.Scalar(init=.999, lower=.1, upper=.9999)

	2. "mlp dist es 3 layer" ::

		hidden_layer_sizes: ng.p.Scalar(init=100, lower=2, upper=300).set_integer_casting()
		activation: ng.p.TransitionChoice(['identity', 'logistic', 'tanh', 'relu'])
		alpha: ng.p.Log(lower=1e-5, upper=1e2)
		batch_size: ng.p.TransitionChoice(['auto', ng.p.Scalar(init=200, lower=50, upper=400).set_integer_casting()])
		learning_rate: ng.p.TransitionChoice(['constant', 'invscaling', 'adaptive'])
		learning_rate_init: ng.p.Log(lower=1e-5, upper=1e2)
		max_iter: ng.p.Scalar(init=200, lower=100, upper=1000).set_integer_casting()
		beta_1: ng.p.Scalar(init=.9, lower=.1, upper=.99)
		beta_2: ng.p.Scalar(init=.999, lower=.1, upper=.9999)
		early_stopping: True
		n_iter_no_change: ng.p.Scalar(lower=5, upper=50)

	3. "mlp dist 2 layer" ::

		hidden_layer_sizes: ng.p.Array(init=(100, 100)).set_mutation(sigma=50).set_bounds(lower=1, upper=300).set_integer_casting()
		activation: ng.p.TransitionChoice(['identity', 'logistic', 'tanh', 'relu'])
		alpha: ng.p.Log(lower=1e-5, upper=1e2)
		batch_size: ng.p.TransitionChoice(['auto', ng.p.Scalar(init=200, lower=50, upper=400).set_integer_casting()])
		learning_rate: ng.p.TransitionChoice(['constant', 'invscaling', 'adaptive'])
		learning_rate_init: ng.p.Log(lower=1e-5, upper=1e2)
		max_iter: ng.p.Scalar(init=200, lower=100, upper=1000).set_integer_casting()
		beta_1: ng.p.Scalar(init=.9, lower=.1, upper=.99)
		beta_2: ng.p.Scalar(init=.999, lower=.1, upper=.9999)

	4. "mlp dist es 2 layer" ::

		hidden_layer_sizes: ng.p.Scalar(init=100, lower=2, upper=300).set_integer_casting()
		activation: ng.p.TransitionChoice(['identity', 'logistic', 'tanh', 'relu'])
		alpha: ng.p.Log(lower=1e-5, upper=1e2)
		batch_size: ng.p.TransitionChoice(['auto', ng.p.Scalar(init=200, lower=50, upper=400).set_integer_casting()])
		learning_rate: ng.p.TransitionChoice(['constant', 'invscaling', 'adaptive'])
		learning_rate_init: ng.p.Log(lower=1e-5, upper=1e2)
		max_iter: ng.p.Scalar(init=200, lower=100, upper=1000).set_integer_casting()
		beta_1: ng.p.Scalar(init=.9, lower=.1, upper=.99)
		beta_2: ng.p.Scalar(init=.999, lower=.1, upper=.9999)
		early_stopping: True
		n_iter_no_change: ng.p.Scalar(lower=5, upper=50)

	5. "mlp dist 1 layer" ::

		hidden_layer_sizes: ng.p.Scalar(init=100, lower=2, upper=300).set_integer_casting()
		activation: ng.p.TransitionChoice(['identity', 'logistic', 'tanh', 'relu'])
		alpha: ng.p.Log(lower=1e-5, upper=1e2)
		batch_size: ng.p.TransitionChoice(['auto', ng.p.Scalar(init=200, lower=50, upper=400).set_integer_casting()])
		learning_rate: ng.p.TransitionChoice(['constant', 'invscaling', 'adaptive'])
		learning_rate_init: ng.p.Log(lower=1e-5, upper=1e2)
		max_iter: ng.p.Scalar(init=200, lower=100, upper=1000).set_integer_casting()
		beta_1: ng.p.Scalar(init=.9, lower=.1, upper=.99)
		beta_2: ng.p.Scalar(init=.999, lower=.1, upper=.9999)

	6. "mlp dist es 1 layer" ::

		hidden_layer_sizes: ng.p.Scalar(init=100, lower=2, upper=300).set_integer_casting()
		activation: ng.p.TransitionChoice(['identity', 'logistic', 'tanh', 'relu'])
		alpha: ng.p.Log(lower=1e-5, upper=1e2)
		batch_size: ng.p.TransitionChoice(['auto', ng.p.Scalar(init=200, lower=50, upper=400).set_integer_casting()])
		learning_rate: ng.p.TransitionChoice(['constant', 'invscaling', 'adaptive'])
		learning_rate_init: ng.p.Log(lower=1e-5, upper=1e2)
		max_iter: ng.p.Scalar(init=200, lower=100, upper=1000).set_integer_casting()
		beta_1: ng.p.Scalar(init=.9, lower=.1, upper=.99)
		beta_2: ng.p.Scalar(init=.999, lower=.1, upper=.9999)
		early_stopping: True
		n_iter_no_change: ng.p.Scalar(lower=5, upper=50)


"pa classifier"
***************

  Base Class Documenation: :class:`sklearn.linear_model.PassiveAggressiveClassifier`

  Param Distributions

	0. "default" ::

		defaults only


"random forest classifier"
**************************

  Base Class Documenation: :class:`sklearn.ensemble.RandomForestClassifier`

  Param Distributions

	0. "base rf regressor" ::

		n_estimators: 100

	1. "rf classifier dist" ::

		n_estimators: ng.p.Scalar(init=100, lower=3, upper=500).set_integer_casting()
		max_depth: ng.p.TransitionChoice([None, ng.p.Scalar(init=25, lower=2, upper=200).set_integer_casting()])
		max_features: ng.p.Scalar(lower=.1, upper=1.0)
		min_samples_split: ng.p.Scalar(lower=.1, upper=1.0)
		bootstrap: True
		class_weight: ng.p.TransitionChoice([None, 'balanced'])


"ridge logistic"
****************

  Base Class Documenation: :class:`sklearn.linear_model.LogisticRegression`

  Param Distributions

	0. "base ridge" ::

		max_iter: 1000
		penalty: 'l2'
		solver: 'saga'

	1. "ridge C" ::

		max_iter: 1000
		solver: 'saga'
		C: ng.p.Log(lower=1e-5, upper=1e3)
		class_weight: ng.p.TransitionChoice([None, 'balanced'])

	2. "ridge C extra" ::

		max_iter: ng.p.Scalar(lower=1000, upper=10000).set_integer_casting()
		solver: 'saga'
		C: ng.p.Log(lower=1e-5, upper=1e3)
		class_weight: ng.p.TransitionChoice([None, 'balanced'])
		tol: ng.p.Log(lower=1e-6, upper=.01)


"sgd classifier"
****************

  Base Class Documenation: :class:`sklearn.linear_model.SGDClassifier`

  Param Distributions

	0. "base sgd" ::

		loss: 'hinge'

	1. "sgd classifier" ::

		loss: ng.p.TransitionChoice(['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'])
		penalty: ng.p.TransitionChoice(['l2', 'l1', 'elasticnet'])
		alpha: ng.p.Log(lower=1e-5, upper=1e2)
		l1_ratio: ng.p.Scalar(lower=0, upper=1)
		max_iter: 1000
		learning_rate: ng.p.TransitionChoice(['optimal', 'invscaling', 'adaptive', 'constant'])
		eta0: ng.p.Log(lower=1e-6, upper=1e3)
		power_t: ng.p.Scalar(lower=.1, upper=.9)
		early_stopping: ng.p.TransitionChoice([False, True])
		validation_fraction: ng.p.Scalar(lower=.05, upper=.5)
		n_iter_no_change: ng.p.TransitionChoice(np.arange(2, 20))
		class_weight: ng.p.TransitionChoice([None, 'balanced'])


"svm classifier"
****************

  Base Class Documenation: :class:`sklearn.svm.SVC`

  Param Distributions

	0. "base svm classifier" ::

		kernel: 'rbf'
		gamma: 'scale'
		probability: True

	1. "svm classifier dist" ::

		kernel: 'rbf'
		gamma: ng.p.Log(lower=1e-6, upper=1)
		C: ng.p.Log(lower=1e-4, upper=1e4)
		probability: True
		class_weight: ng.p.TransitionChoice([None, 'balanced'])


"xgb classifier"
****************

  Base Class Documenation: :class:`xgboost.XGBClassifier`

  Param Distributions

	0. "base xgb classifier" ::

		verbosity: 0
		objective: 'binary:logistic'

	1. "xgb classifier dist1" ::

		verbosity: 0
		objective: 'binary:logistic'
		n_estimators: ng.p.Scalar(init=100, lower=3, upper=500).set_integer_casting()
		min_child_weight: ng.p.Log(lower=1e-5, upper=1e4)
		subsample: ng.p.Scalar(lower=.3, upper=.95)
		colsample_bytree: ng.p.Scalar(lower=.3, upper=.95)
		reg_alpha: ng.p.TransitionChoice([0, ng.p.Log(lower=1e-5, upper=1)])
		reg_lambda: ng.p.TransitionChoice([0, ng.p.Log(lower=1e-5, upper=1)])

	2. "xgb classifier dist2" ::

		verbosity: 0
		objective: 'binary:logistic'
		max_depth: ng.p.TransitionChoice([None, ng.p.Scalar(init=25, lower=2, upper=200).set_integer_casting()])
		learning_rate: ng.p.Scalar(lower=.01, upper=.5)
		n_estimators: ng.p.Scalar(lower=3, upper=500).set_integer_casting()
		min_child_weight: ng.p.TransitionChoice([1, 5, 10, 50])
		subsample: ng.p.Scalar(lower=.5, upper=1)
		colsample_bytree: ng.p.Scalar(lower=.4, upper=.95)

	3. "xgb classifier dist3" ::

		verbosity: 0
		objective: 'binary:logistic'
		learning_rare: ng.p.Scalar(lower=.005, upper=.3)
		min_child_weight: ng.p.Scalar(lower=.5, upper=10)
		max_depth: ng.p.TransitionChoice(np.arange(3, 10))
		subsample: ng.p.Scalar(lower=.5, upper=1)
		colsample_bytree: ng.p.Scalar(lower=.5, upper=1)
		reg_alpha: ng.p.Log(lower=.00001, upper=1)



.. _Scorers:
 
*******
Scorers
*******

Different availible choices for the `scorer` parameter are shown below.
`scorer` is accepted by :class:`Problem_Spec<BPt.Problem_Spec>`, :class:`Param_Search<BPt.Param_Search>` and :class:`Feat_Importance<BPt.Feat_Importance>`
The str indicator for each `scorer` is represented bythe sub-heading (within "")
The avaliable scorers are further broken down by which can work with different problem_types.
Additionally, a link to the original models documentation is shown.

binary
======
"accuracy"
**********

  Base Func Documenation: :func:`sklearn.metrics.accuracy_score`

"roc_auc"
*********

  Base Func Documenation: :func:`sklearn.metrics.roc_auc_score`

"roc_auc_ovr"
*************

  Base Func Documenation: :func:`sklearn.metrics.roc_auc_score`

"roc_auc_ovo"
*************

  Base Func Documenation: :func:`sklearn.metrics.roc_auc_score`

"roc_auc_ovr_weighted"
**********************

  Base Func Documenation: :func:`sklearn.metrics.roc_auc_score`

"roc_auc_ovo_weighted"
**********************

  Base Func Documenation: :func:`sklearn.metrics.roc_auc_score`

"balanced_accuracy"
*******************

  Base Func Documenation: :func:`sklearn.metrics.balanced_accuracy_score`

"average_precision"
*******************

  Base Func Documenation: :func:`sklearn.metrics.average_precision_score`

"neg_log_loss"
**************

  Base Func Documenation: :func:`sklearn.metrics.log_loss`

"neg_brier_score"
*****************

  Base Func Documenation: :func:`sklearn.metrics.brier_score_loss`

"precision"
***********

  Base Func Documenation: :func:`sklearn.metrics.precision_score`

"precision_macro"
*****************

  Base Func Documenation: :func:`sklearn.metrics.precision_score`

"precision_micro"
*****************

  Base Func Documenation: :func:`sklearn.metrics.precision_score`

"precision_samples"
*******************

  Base Func Documenation: :func:`sklearn.metrics.precision_score`

"precision_weighted"
********************

  Base Func Documenation: :func:`sklearn.metrics.precision_score`

"recall"
********

  Base Func Documenation: :func:`sklearn.metrics.recall_score`

"recall_macro"
**************

  Base Func Documenation: :func:`sklearn.metrics.recall_score`

"recall_micro"
**************

  Base Func Documenation: :func:`sklearn.metrics.recall_score`

"recall_samples"
****************

  Base Func Documenation: :func:`sklearn.metrics.recall_score`

"recall_weighted"
*****************

  Base Func Documenation: :func:`sklearn.metrics.recall_score`

"f1"
****

  Base Func Documenation: :func:`sklearn.metrics.f1_score`

"f1_macro"
**********

  Base Func Documenation: :func:`sklearn.metrics.f1_score`

"f1_micro"
**********

  Base Func Documenation: :func:`sklearn.metrics.f1_score`

"f1_samples"
************

  Base Func Documenation: :func:`sklearn.metrics.f1_score`

"f1_weighted"
*************

  Base Func Documenation: :func:`sklearn.metrics.f1_score`

"jaccard"
*********

  Base Func Documenation: :func:`sklearn.metrics.jaccard_score`

"jaccard_macro"
***************

  Base Func Documenation: :func:`sklearn.metrics.jaccard_score`

"jaccard_micro"
***************

  Base Func Documenation: :func:`sklearn.metrics.jaccard_score`

"jaccard_samples"
*****************

  Base Func Documenation: :func:`sklearn.metrics.jaccard_score`

"jaccard_weighted"
******************

  Base Func Documenation: :func:`sklearn.metrics.jaccard_score`

"neg_hamming"
*************

  Base Func Documenation: :func:`sklearn.metrics.hamming_loss`

"matthews"
**********

  Base Func Documenation: :func:`sklearn.metrics.matthews_corrcoef`

"default"
*********

  Base Func Documenation: :func:`sklearn.metrics.roc_auc_score`


regression
==========
"explained_variance"
********************

  Base Func Documenation: :func:`sklearn.metrics.explained_variance_score`

"explained_variance score"
**************************

  Base Func Documenation: :func:`sklearn.metrics.explained_variance_score`

"r2"
****

  Base Func Documenation: :func:`sklearn.metrics.r2_score`

"max_error"
***********

  Base Func Documenation: :func:`sklearn.metrics.max_error`

"neg_median_absolute_error"
***************************

  Base Func Documenation: :func:`sklearn.metrics.median_absolute_error`

"median_absolute_error"
***********************

  Base Func Documenation: :func:`sklearn.metrics.median_absolute_error`

"neg_mean_absolute_error"
*************************

  Base Func Documenation: :func:`sklearn.metrics.mean_absolute_error`

"mean_absolute_error"
*********************

  Base Func Documenation: :func:`sklearn.metrics.mean_absolute_error`

"neg_mean_squared_error"
************************

  Base Func Documenation: :func:`sklearn.metrics.mean_squared_error`

"mean_squared_error"
********************

  Base Func Documenation: :func:`sklearn.metrics.mean_squared_error`

"neg_mean_squared_log_error"
****************************

  Base Func Documenation: :func:`sklearn.metrics.mean_squared_log_error`

"mean_squared_log_error"
************************

  Base Func Documenation: :func:`sklearn.metrics.mean_squared_log_error`

"neg_root_mean_squared_error"
*****************************

  Base Func Documenation: :func:`sklearn.metrics.mean_squared_error`

"root_mean_squared_error"
*************************

  Base Func Documenation: :func:`sklearn.metrics.mean_squared_error`

"neg_mean_poisson_deviance"
***************************

  Base Func Documenation: :func:`sklearn.metrics.mean_poisson_deviance`

"mean_poisson_deviance"
***********************

  Base Func Documenation: :func:`sklearn.metrics.mean_poisson_deviance`

"neg_mean_gamma_deviance"
*************************

  Base Func Documenation: :func:`sklearn.metrics.mean_gamma_deviance`

"mean_gamma_deviance"
*********************

  Base Func Documenation: :func:`sklearn.metrics.mean_gamma_deviance`

"default"
*********

  Base Func Documenation: :func:`sklearn.metrics.r2_score`


categorical
===========
"accuracy"
**********

  Base Func Documenation: :func:`sklearn.metrics.accuracy_score`

"roc_auc"
*********

  Base Func Documenation: :func:`sklearn.metrics.roc_auc_score`

"roc_auc_ovr"
*************

  Base Func Documenation: :func:`sklearn.metrics.roc_auc_score`

"roc_auc_ovo"
*************

  Base Func Documenation: :func:`sklearn.metrics.roc_auc_score`

"roc_auc_ovr_weighted"
**********************

  Base Func Documenation: :func:`sklearn.metrics.roc_auc_score`

"roc_auc_ovo_weighted"
**********************

  Base Func Documenation: :func:`sklearn.metrics.roc_auc_score`

"balanced_accuracy"
*******************

  Base Func Documenation: :func:`sklearn.metrics.balanced_accuracy_score`

"average_precision"
*******************

  Base Func Documenation: :func:`sklearn.metrics.average_precision_score`

"neg_log_loss"
**************

  Base Func Documenation: :func:`sklearn.metrics.log_loss`

"neg_brier_score"
*****************

  Base Func Documenation: :func:`sklearn.metrics.brier_score_loss`

"precision"
***********

  Base Func Documenation: :func:`sklearn.metrics.precision_score`

"precision_macro"
*****************

  Base Func Documenation: :func:`sklearn.metrics.precision_score`

"precision_micro"
*****************

  Base Func Documenation: :func:`sklearn.metrics.precision_score`

"precision_samples"
*******************

  Base Func Documenation: :func:`sklearn.metrics.precision_score`

"precision_weighted"
********************

  Base Func Documenation: :func:`sklearn.metrics.precision_score`

"recall"
********

  Base Func Documenation: :func:`sklearn.metrics.recall_score`

"recall_macro"
**************

  Base Func Documenation: :func:`sklearn.metrics.recall_score`

"recall_micro"
**************

  Base Func Documenation: :func:`sklearn.metrics.recall_score`

"recall_samples"
****************

  Base Func Documenation: :func:`sklearn.metrics.recall_score`

"recall_weighted"
*****************

  Base Func Documenation: :func:`sklearn.metrics.recall_score`

"f1"
****

  Base Func Documenation: :func:`sklearn.metrics.f1_score`

"f1_macro"
**********

  Base Func Documenation: :func:`sklearn.metrics.f1_score`

"f1_micro"
**********

  Base Func Documenation: :func:`sklearn.metrics.f1_score`

"f1_samples"
************

  Base Func Documenation: :func:`sklearn.metrics.f1_score`

"f1_weighted"
*************

  Base Func Documenation: :func:`sklearn.metrics.f1_score`

"jaccard"
*********

  Base Func Documenation: :func:`sklearn.metrics.jaccard_score`

"jaccard_macro"
***************

  Base Func Documenation: :func:`sklearn.metrics.jaccard_score`

"jaccard_micro"
***************

  Base Func Documenation: :func:`sklearn.metrics.jaccard_score`

"jaccard_samples"
*****************

  Base Func Documenation: :func:`sklearn.metrics.jaccard_score`

"jaccard_weighted"
******************

  Base Func Documenation: :func:`sklearn.metrics.jaccard_score`

"neg_hamming"
*************

  Base Func Documenation: :func:`sklearn.metrics.hamming_loss`

"matthews"
**********

  Base Func Documenation: :func:`sklearn.metrics.matthews_corrcoef`

"default"
*********

  Base Func Documenation: :func:`sklearn.metrics.roc_auc_score`


.. _Loaders:
 
*******
Loaders
*******

Different base obj choices for the :class:`Loader<BPt.Loader>` are shown below
The exact str indicator, as passed to the `obj` param is represented by the sub-heading (within "")
Additionally, a link to the original models documentation as well as the implemented parameter distributions are shown.

All Problem Types
=================
"identity"
**********

  Base Class Documenation: :class:`BPt.extensions.Loaders.Identity`

  Param Distributions

	0. "default" ::

		defaults only


"surface rois"
**************

  Base Class Documenation: :class:`BPt.extensions.Loaders.SurfLabels`

  Param Distributions

	0. "default" ::

		defaults only


"volume rois"
*************

  Base Class Documenation: :class:`nilearn.input_data.nifti_labels_masker.NiftiLabelsMasker`

  Param Distributions

	0. "default" ::

		defaults only


"connectivity"
**************

  Base Class Documenation: :class:`BPt.extensions.Loaders.Connectivity`

  Param Distributions

	0. "default" ::

		defaults only



.. _Imputers:
 
********
Imputers
********

Different base obj choices for the :class:`Imputer<BPt.Imputer>` are shown below
The exact str indicator, as passed to the `obj` param is represented by the sub-heading (within "")
Additionally, a link to the original models documentation as well as the implemented parameter distributions are shown.
Note that if the iterative imputer is requested, base_model must also be passed.

All Problem Types
=================
"mean"
******

  Base Class Documenation: :class:`sklearn.impute.SimpleImputer`

  Param Distributions

	0. "mean imp" ::

		strategy: 'mean'


"median"
********

  Base Class Documenation: :class:`sklearn.impute.SimpleImputer`

  Param Distributions

	0. "median imp" ::

		strategy: 'median'


"most frequent"
***************

  Base Class Documenation: :class:`sklearn.impute.SimpleImputer`

  Param Distributions

	0. "most freq imp" ::

		strategy: 'most_frequent'


"constant"
**********

  Base Class Documenation: :class:`sklearn.impute.SimpleImputer`

  Param Distributions

	0. "constant imp" ::

		strategy: 'constant'


"iterative"
***********

  Base Class Documenation: :class:`sklearn.impute.IterativeImputer`

  Param Distributions

	0. "iterative imp" ::

		initial_strategy: 'mean'
		skip_complete: True



.. _Scalers:
 
*******
Scalers
*******

Different base obj choices for the :class:`Scaler<BPt.Scaler>` are shown below
The exact str indicator, as passed to the `obj` param is represented by the sub-heading (within "")
Additionally, a link to the original models documentation as well as the implemented parameter distributions are shown.

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

	0. "default" ::

		defaults only


"robust"
********

  Base Class Documenation: :class:`sklearn.preprocessing.RobustScaler`

  Param Distributions

	0. "base robust" ::

		quantile_range: (5, 95)

	1. "robust gs" ::

		quantile_range: ng.p.TransitionChoice([(x, 100-x) for x in np.arange(1, 40)])


"yeo"
*****

  Base Class Documenation: :class:`sklearn.preprocessing.PowerTransformer`

  Param Distributions

	0. "base yeo" ::

		method: 'yeo-johnson'
		standardize: True


"boxcox"
********

  Base Class Documenation: :class:`sklearn.preprocessing.PowerTransformer`

  Param Distributions

	0. "base boxcox" ::

		method: 'box-cox'
		standardize: True


"winsorize"
***********

  Base Class Documenation: :class:`BPt.extensions.Scalers.Winsorizer`

  Param Distributions

	0. "base winsorize" ::

		quantile_range: (1, 99)

	1. "winsorize gs" ::

		quantile_range: ng.p.TransitionChoice([(x, 100-x) for x in np.arange(1, 40)])


"quantile norm"
***************

  Base Class Documenation: :class:`sklearn.preprocessing.QuantileTransformer`

  Param Distributions

	0. "base quant norm" ::

		output_distribution: 'normal'


"quantile uniform"
******************

  Base Class Documenation: :class:`sklearn.preprocessing.QuantileTransformer`

  Param Distributions

	0. "base quant uniform" ::

		output_distribution: 'uniform'


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

Different base obj choices for the :class:`Transformer<BPt.Transformer>` are shown below
The exact str indicator, as passed to the `obj` param is represented by the sub-heading (within "")
Additionally, a link to the original models documentation as well as the implemented parameter distributions are shown.

All Problem Types
=================
"pca"
*****

  Base Class Documenation: :class:`sklearn.decomposition.PCA`

  Param Distributions

	0. "default" ::

		defaults only

	1. "pca var search" ::

		n_components: ng.p.Scalar(init=.75, lower=.1, upper=.99)
		svd_solver: 'full'


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


"truncated svd"
***************

  Base Class Documenation: :class:`sklearn.decomposition.TruncatedSVD`

  Param Distributions

	0. "default" ::

		defaults only


"one hot encoder"
*****************

  Base Class Documenation: :class:`sklearn.preprocessing.OneHotEncoder`

  Param Distributions

	0. "ohe" ::

		sparse: False
		handle_unknown: 'ignore'


"backward difference encoder"
*****************************

  Base Class Documenation: :class:`category_encoders.backward_difference.BackwardDifferenceEncoder`

  Param Distributions

	0. "default" ::

		defaults only


"binary encoder"
****************

  Base Class Documenation: :class:`category_encoders.binary.BinaryEncoder`

  Param Distributions

	0. "default" ::

		defaults only


"cat boost encoder"
*******************

  Base Class Documenation: :class:`category_encoders.cat_boost.CatBoostEncoder`

  Param Distributions

	0. "default" ::

		defaults only


"helmert encoder"
*****************

  Base Class Documenation: :class:`category_encoders.helmert.HelmertEncoder`

  Param Distributions

	0. "default" ::

		defaults only


"james stein encoder"
*********************

  Base Class Documenation: :class:`category_encoders.james_stein.JamesSteinEncoder`

  Param Distributions

	0. "default" ::

		defaults only


"leave one out encoder"
***********************

  Base Class Documenation: :class:`category_encoders.leave_one_out.LeaveOneOutEncoder`

  Param Distributions

	0. "default" ::

		defaults only


"m estimate encoder"
********************

  Base Class Documenation: :class:`category_encoders.m_estimate.MEstimateEncoder`

  Param Distributions

	0. "default" ::

		defaults only


"polynomial encoder"
********************

  Base Class Documenation: :class:`category_encoders.polynomial.PolynomialEncoder`

  Param Distributions

	0. "default" ::

		defaults only


"sum encoder"
*************

  Base Class Documenation: :class:`category_encoders.sum_coding.SumEncoder`

  Param Distributions

	0. "default" ::

		defaults only


"target encoder"
****************

  Base Class Documenation: :class:`category_encoders.target_encoder.TargetEncoder`

  Param Distributions

	0. "default" ::

		defaults only


"woe encoder"
*************

  Base Class Documenation: :class:`category_encoders.woe.WOEEncoder`

  Param Distributions

	0. "default" ::

		defaults only



.. _Feat Selectors:
 
**************
Feat Selectors
**************

Different base obj choices for the :class:`Feat_Selector<BPt.Feat_Selector>` are shown below
The exact str indicator, as passed to the `obj` param is represented by the sub-heading (within "")
The avaliable feat selectors are further broken down by which can workwith different problem_types.
Additionally, a link to the original models documentation as well as the implemented parameter distributions are shown.

binary
======
"rfe"
*****

  Base Class Documenation: :class:`sklearn.feature_selection.RFE`

  Param Distributions

	0. "base rfe" ::

		n_features_to_select: None

	1. "rfe num feats dist" ::

		n_features_to_select: ng.p.Scalar(init=.5, lower=.1, upper=.99)


"selector"
**********

  Base Class Documenation: :class:`BPt.extensions.Feat_Selectors.FeatureSelector`

  Param Distributions

	0. "random" ::

		mask: 'sets as random features'

	1. "searchable" ::

		mask: 'sets as hyperparameters'


"univariate selection c"
************************

  Base Class Documenation: :class:`sklearn.feature_selection.SelectPercentile`

  Param Distributions

	0. "base univar fs classifier" ::

		score_func: f_classif
		percentile: 50

	1. "univar fs classifier dist" ::

		score_func: f_classif
		percentile: ng.p.Scalar(init=50, lower=1, upper=99)

	2. "univar fs classifier dist2" ::

		score_func: f_classif
		percentile: ng.p.Scalar(init=75, lower=50, upper=99)


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

		n_features_to_select: ng.p.Scalar(init=.5, lower=.1, upper=.99)


"selector"
**********

  Base Class Documenation: :class:`BPt.extensions.Feat_Selectors.FeatureSelector`

  Param Distributions

	0. "random" ::

		mask: 'sets as random features'

	1. "searchable" ::

		mask: 'sets as hyperparameters'


"univariate selection r"
************************

  Base Class Documenation: :class:`sklearn.feature_selection.SelectPercentile`

  Param Distributions

	0. "base univar fs regression" ::

		score_func: f_regression
		percentile: 50

	1. "univar fs regression dist" ::

		score_func: f_regression
		percentile: ng.p.Scalar(init=50, lower=1, upper=99)

	2. "univar fs regression dist2" ::

		score_func: f_regression
		percentile: ng.p.Scalar(init=75, lower=50, upper=99)


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

		n_features_to_select: ng.p.Scalar(init=.5, lower=.1, upper=.99)


"selector"
**********

  Base Class Documenation: :class:`BPt.extensions.Feat_Selectors.FeatureSelector`

  Param Distributions

	0. "random" ::

		mask: 'sets as random features'

	1. "searchable" ::

		mask: 'sets as hyperparameters'


"univariate selection c"
************************

  Base Class Documenation: :class:`sklearn.feature_selection.SelectPercentile`

  Param Distributions

	0. "base univar fs classifier" ::

		score_func: f_classif
		percentile: 50

	1. "univar fs classifier dist" ::

		score_func: f_classif
		percentile: ng.p.Scalar(init=50, lower=1, upper=99)

	2. "univar fs classifier dist2" ::

		score_func: f_classif
		percentile: ng.p.Scalar(init=75, lower=50, upper=99)


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

Different base obj choices for the :class:`Ensemble<BPt.Ensemble>` are shown below
The exact str indicator, as passed to the `obj` param is represented by the sub-heading (within "")
The avaliable ensembles are further broken down by which can workwith different problem_types.
Additionally, a link to the original models documentation as well as the implemented parameter distributions are shown.
Also note that ensemble require a few extra params! I.e., in general, all DESlib based ensemble need needs_split = True

binary
======
"adaboost classifier"
*********************

  Base Class Documenation: :class:`sklearn.ensemble.AdaBoostClassifier`

  Param Distributions

	0. "default" ::

		defaults only


"aposteriori"
*************

  Base Class Documenation: :class:`deslib.dcs.a_posteriori.APosteriori`

  Param Distributions

	0. "default" ::

		defaults only


"apriori"
*********

  Base Class Documenation: :class:`deslib.dcs.a_priori.APriori`

  Param Distributions

	0. "default" ::

		defaults only


"bagging classifier"
********************

  Base Class Documenation: :class:`sklearn.ensemble.BaggingClassifier`

  Param Distributions

	0. "default" ::

		defaults only


"balanced bagging classifier"
*****************************

  Base Class Documenation: :class:`imblearn.ensemble.BalancedBaggingClassifier`

  Param Distributions

	0. "default" ::

		defaults only


"des clustering"
****************

  Base Class Documenation: :class:`deslib.des.des_clustering.DESClustering`

  Param Distributions

	0. "default" ::

		defaults only


"des knn"
*********

  Base Class Documenation: :class:`deslib.des.des_knn.DESKNN`

  Param Distributions

	0. "default" ::

		defaults only


"deskl"
*******

  Base Class Documenation: :class:`deslib.des.probabilistic.DESKL`

  Param Distributions

	0. "default" ::

		defaults only


"desmi"
*******

  Base Class Documenation: :class:`deslib.des.des_mi.DESMI`

  Param Distributions

	0. "default" ::

		defaults only


"desp"
******

  Base Class Documenation: :class:`deslib.des.des_p.DESP`

  Param Distributions

	0. "default" ::

		defaults only


"exponential"
*************

  Base Class Documenation: :class:`deslib.des.probabilistic.Exponential`

  Param Distributions

	0. "default" ::

		defaults only


"knop"
******

  Base Class Documenation: :class:`deslib.des.knop.KNOP`

  Param Distributions

	0. "default" ::

		defaults only


"knorae"
********

  Base Class Documenation: :class:`deslib.des.knora_e.KNORAE`

  Param Distributions

	0. "default" ::

		defaults only


"knrau"
*******

  Base Class Documenation: :class:`deslib.des.knora_u.KNORAU`

  Param Distributions

	0. "default" ::

		defaults only


"lca"
*****

  Base Class Documenation: :class:`deslib.dcs.lca.LCA`

  Param Distributions

	0. "default" ::

		defaults only


"logarithmic"
*************

  Base Class Documenation: :class:`deslib.des.probabilistic.Logarithmic`

  Param Distributions

	0. "default" ::

		defaults only


"mcb"
*****

  Base Class Documenation: :class:`deslib.dcs.mcb.MCB`

  Param Distributions

	0. "default" ::

		defaults only


"metades"
*********

  Base Class Documenation: :class:`deslib.des.meta_des.METADES`

  Param Distributions

	0. "default" ::

		defaults only


"min dif"
*********

  Base Class Documenation: :class:`deslib.des.probabilistic.MinimumDifference`

  Param Distributions

	0. "default" ::

		defaults only


"mla"
*****

  Base Class Documenation: :class:`deslib.dcs.mla.MLA`

  Param Distributions

	0. "default" ::

		defaults only


"ola"
*****

  Base Class Documenation: :class:`deslib.dcs.ola.OLA`

  Param Distributions

	0. "default" ::

		defaults only


"rank"
******

  Base Class Documenation: :class:`deslib.dcs.rank.Rank`

  Param Distributions

	0. "default" ::

		defaults only


"rrc"
*****

  Base Class Documenation: :class:`deslib.des.probabilistic.RRC`

  Param Distributions

	0. "default" ::

		defaults only


"single best"
*************

  Base Class Documenation: :class:`deslib.static.single_best.SingleBest`

  Param Distributions

	0. "default" ::

		defaults only


"stacked"
*********

  Base Class Documenation: :class:`deslib.static.stacked.StackedClassifier`

  Param Distributions

	0. "default" ::

		defaults only


"stacking classifier"
*********************

  Base Class Documenation: :class:`BPt.pipeline.Ensembles.BPtStackingClassifier`

  Param Distributions

	0. "default" ::

		defaults only


"voting classifier"
*******************

  Base Class Documenation: :class:`BPt.pipeline.Ensembles.BPtVotingClassifier`

  Param Distributions

	0. "voting classifier" ::

		voting: 'soft'



regression
==========
"adaboost regressor"
********************

  Base Class Documenation: :class:`sklearn.ensemble.AdaBoostRegressor`

  Param Distributions

	0. "default" ::

		defaults only


"bagging regressor"
*******************

  Base Class Documenation: :class:`sklearn.ensemble.BaggingRegressor`

  Param Distributions

	0. "default" ::

		defaults only


"stacking regressor"
********************

  Base Class Documenation: :class:`BPt.pipeline.Ensembles.BPtStackingRegressor`

  Param Distributions

	0. "default" ::

		defaults only


"voting regressor"
******************

  Base Class Documenation: :class:`BPt.pipeline.Ensembles.BPtVotingRegressor`

  Param Distributions

	0. "default" ::

		defaults only



categorical
===========
"adaboost classifier"
*********************

  Base Class Documenation: :class:`sklearn.ensemble.AdaBoostClassifier`

  Param Distributions

	0. "default" ::

		defaults only


"aposteriori"
*************

  Base Class Documenation: :class:`deslib.dcs.a_posteriori.APosteriori`

  Param Distributions

	0. "default" ::

		defaults only


"apriori"
*********

  Base Class Documenation: :class:`deslib.dcs.a_priori.APriori`

  Param Distributions

	0. "default" ::

		defaults only


"bagging classifier"
********************

  Base Class Documenation: :class:`sklearn.ensemble.BaggingClassifier`

  Param Distributions

	0. "default" ::

		defaults only


"balanced bagging classifier"
*****************************

  Base Class Documenation: :class:`imblearn.ensemble.BalancedBaggingClassifier`

  Param Distributions

	0. "default" ::

		defaults only


"des clustering"
****************

  Base Class Documenation: :class:`deslib.des.des_clustering.DESClustering`

  Param Distributions

	0. "default" ::

		defaults only


"des knn"
*********

  Base Class Documenation: :class:`deslib.des.des_knn.DESKNN`

  Param Distributions

	0. "default" ::

		defaults only


"deskl"
*******

  Base Class Documenation: :class:`deslib.des.probabilistic.DESKL`

  Param Distributions

	0. "default" ::

		defaults only


"desmi"
*******

  Base Class Documenation: :class:`deslib.des.des_mi.DESMI`

  Param Distributions

	0. "default" ::

		defaults only


"desp"
******

  Base Class Documenation: :class:`deslib.des.des_p.DESP`

  Param Distributions

	0. "default" ::

		defaults only


"exponential"
*************

  Base Class Documenation: :class:`deslib.des.probabilistic.Exponential`

  Param Distributions

	0. "default" ::

		defaults only


"knop"
******

  Base Class Documenation: :class:`deslib.des.knop.KNOP`

  Param Distributions

	0. "default" ::

		defaults only


"knorae"
********

  Base Class Documenation: :class:`deslib.des.knora_e.KNORAE`

  Param Distributions

	0. "default" ::

		defaults only


"knrau"
*******

  Base Class Documenation: :class:`deslib.des.knora_u.KNORAU`

  Param Distributions

	0. "default" ::

		defaults only


"lca"
*****

  Base Class Documenation: :class:`deslib.dcs.lca.LCA`

  Param Distributions

	0. "default" ::

		defaults only


"logarithmic"
*************

  Base Class Documenation: :class:`deslib.des.probabilistic.Logarithmic`

  Param Distributions

	0. "default" ::

		defaults only


"mcb"
*****

  Base Class Documenation: :class:`deslib.dcs.mcb.MCB`

  Param Distributions

	0. "default" ::

		defaults only


"metades"
*********

  Base Class Documenation: :class:`deslib.des.meta_des.METADES`

  Param Distributions

	0. "default" ::

		defaults only


"min dif"
*********

  Base Class Documenation: :class:`deslib.des.probabilistic.MinimumDifference`

  Param Distributions

	0. "default" ::

		defaults only


"mla"
*****

  Base Class Documenation: :class:`deslib.dcs.mla.MLA`

  Param Distributions

	0. "default" ::

		defaults only


"ola"
*****

  Base Class Documenation: :class:`deslib.dcs.ola.OLA`

  Param Distributions

	0. "default" ::

		defaults only


"rank"
******

  Base Class Documenation: :class:`deslib.dcs.rank.Rank`

  Param Distributions

	0. "default" ::

		defaults only


"rrc"
*****

  Base Class Documenation: :class:`deslib.des.probabilistic.RRC`

  Param Distributions

	0. "default" ::

		defaults only


"single best"
*************

  Base Class Documenation: :class:`deslib.static.single_best.SingleBest`

  Param Distributions

	0. "default" ::

		defaults only


"stacked"
*********

  Base Class Documenation: :class:`deslib.static.stacked.StackedClassifier`

  Param Distributions

	0. "default" ::

		defaults only


"stacking classifier"
*********************

  Base Class Documenation: :class:`BPt.pipeline.Ensembles.BPtStackingClassifier`

  Param Distributions

	0. "default" ::

		defaults only


"voting classifier"
*******************

  Base Class Documenation: :class:`BPt.pipeline.Ensembles.BPtVotingClassifier`

  Param Distributions

	0. "voting classifier" ::

		voting: 'soft'



