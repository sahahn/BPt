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

	0. "base dt" ::

		defaults only

	1. "dt classifier dist" ::

		max_depth: Scalar{int,Cl(1,30)}[sigma=Log{exp=1.2}]:16
		min_samples_split: Scalar{int,Cl(2,50)}[sigma=Log{exp=1.2}]:26
		class_weight: TransitionChoice(choices=Tuple(None,balanced),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None


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
		class_weight: TransitionChoice(choices=Tuple(None,balanced),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None
		solver: saga
		l1_ratio: Scalar{Cl(0.01,1)}[sigma=Log{exp=1.2}]:0.505
		C: Log{exp=46.415888336127786,Cl(1e-05,100000)}:1.0

	2. "elastic clf v2" ::

		max_iter: 5000
		multi_class: auto
		penalty: elasticnet
		class_weight: TransitionChoice(choices=Tuple(None,balanced),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None
		solver: saga
		l1_ratio: Scalar{Cl(0.01,1)}[sigma=Log{exp=1.2}]:0.505
		C: Log{exp=14.677992676220699,Cl(0.01,100000)}:31.622776601683793

	3. "elastic classifier extra" ::

		max_iter: Scalar{int,Cl(1000,10000)}[sigma=Log{exp=1.2}]:5500
		multi_class: auto
		penalty: elasticnet
		class_weight: TransitionChoice(choices=Tuple(None,balanced),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None
		solver: saga
		l1_ratio: Scalar{Cl(0.01,1)}[sigma=Log{exp=1.2}]:0.505
		C: Log{exp=46.415888336127786,Cl(1e-05,100000)}:1.0
		tol: Log{exp=4.641588833612779,Cl(1e-06,0.01)}:0.0001


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

		var_smoothing: 1e-09


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

		weights: TransitionChoice(choices=Tuple(uniform,distance),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):uniform
		n_neighbors: Scalar{int,Cl(2,25)}[sigma=Log{exp=1.2}]:14


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
		class_weight: TransitionChoice(choices=Tuple(None,balanced),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None
		solver: liblinear
		C: Log{exp=21.544346900318843,Cl(1e-05,1000)}:0.1

	2. "lasso C extra" ::

		max_iter: Scalar{int,Cl(1000,10000)}[sigma=Log{exp=1.2}]:5500
		multi_class: auto
		penalty: l1
		class_weight: TransitionChoice(choices=Tuple(None,balanced),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None
		solver: liblinear
		C: Log{exp=21.544346900318843,Cl(1e-05,1000)}:0.1
		tol: Log{exp=4.641588833612779,Cl(1e-06,0.01)}:0.0001


"light gbm classifier"
**********************

  Base Class Documenation: :class:`lightgbm.LGBMClassifier`

  Param Distributions

	0. "base lgbm" ::

		silent: True

	1. "lgbm classifier dist1" ::

		silent: True
		boosting_type: TransitionChoice(choices=Tuple(gbdt,dart,goss),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):dart
		n_estimators: Scalar{int,Cl(3,500)}[sigma=Log{exp=1.2}]:100
		num_leaves: Scalar{int,Cl(6,80)}[sigma=Log{exp=1.2}]:20
		min_child_samples: Scalar{int,Cl(10,500)}[sigma=Log{exp=1.2}]:255
		min_child_weight: Log{exp=31.62277660168379,Cl(1e-05,10000)}:0.31622776601683794
		subsample: Scalar{Cl(0.3,0.95)}[sigma=Log{exp=1.2}]:0.625
		colsample_bytree: Scalar{Cl(0.3,0.95)}[sigma=Log{exp=1.2}]:0.625
		reg_alpha: TransitionChoice(choices=Tuple(0,Log{exp=6.812920690579612,Cl(1e-05,1)}),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):0
		reg_lambda: TransitionChoice(choices=Tuple(0,Log{exp=6.812920690579612,Cl(1e-05,1)}),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):0
		class_weight: TransitionChoice(choices=Tuple(None,balanced),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None

	2. "lgbm classifier dist2" ::

		silent: True
		lambda_l2: 0.001
		boosting_type: TransitionChoice(choices=Tuple(gbdt,dart),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):gbdt
		min_child_samples: TransitionChoice(choices=Tuple(1,5,7,10,15,20,35,50,100,200,500,1000),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):35
		num_leaves: TransitionChoice(choices=Tuple(2,4,7,10,15,20,25,30,35,40,50,65,80,100,125,150,200,250),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):40
		colsample_bytree: TransitionChoice(choices=Tuple(0.7,0.9,1.0),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):0.9
		subsample: Scalar{Cl(0.3,1)}[sigma=Log{exp=1.2}]:0.65
		learning_rate: TransitionChoice(choices=Tuple(0.01,0.05,0.1),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):0.05
		n_estimators: TransitionChoice(choices=Tuple(5,20,35,50,75,100,150,200,350,500,750,1000),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):150
		class_weight: TransitionChoice(choices=Tuple(None,balanced),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None


"linear svm classifier"
***********************

  Base Class Documenation: :class:`sklearn.svm.LinearSVC`

  Param Distributions

	0. "base linear svc" ::

		max_iter: 5000

	1. "linear svc dist" ::

		max_iter: 5000
		C: Log{exp=21.544346900318843,Cl(0.0001,10000)}:1.0
		class_weight: TransitionChoice(choices=Tuple(None,balanced),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None


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

  Base Class Documenation: :class:`BPt.extensions.MLP.MLPClassifier_Wrapper`

  Param Distributions

	0. "base mlp" ::

		defaults only

	1. "mlp dist 3 layer" ::

		hidden_layer_sizes: Array{int,Cl(1,300)}[sigma=50]:[100 100 100]
		activation: TransitionChoice(choices=Tuple(identity,logistic,tanh,relu),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):tanh
		alpha: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		batch_size: TransitionChoice(choices=Tuple(auto,Scalar{int,Cl(50,400)}[sigma=Log{exp=1.2}]),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):auto
		learning_rate: TransitionChoice(choices=Tuple(constant,invscaling,adaptive),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):invscaling
		learning_rate_init: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		max_iter: Scalar{int,Cl(100,1000)}[sigma=Log{exp=1.2}]:200
		beta_1: Scalar{Cl(0.1,0.99)}[sigma=Log{exp=1.2}]:0.9
		beta_2: Scalar{Cl(0.1,0.9999)}[sigma=Log{exp=1.2}]:0.999

	2. "mlp dist es 3 layer" ::

		hidden_layer_sizes: Scalar{int,Cl(2,300)}[sigma=Log{exp=1.2}]:100
		activation: TransitionChoice(choices=Tuple(identity,logistic,tanh,relu),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):tanh
		alpha: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		batch_size: TransitionChoice(choices=Tuple(auto,Scalar{int,Cl(50,400)}[sigma=Log{exp=1.2}]),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):auto
		learning_rate: TransitionChoice(choices=Tuple(constant,invscaling,adaptive),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):invscaling
		learning_rate_init: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		max_iter: Scalar{int,Cl(100,1000)}[sigma=Log{exp=1.2}]:200
		beta_1: Scalar{Cl(0.1,0.99)}[sigma=Log{exp=1.2}]:0.9
		beta_2: Scalar{Cl(0.1,0.9999)}[sigma=Log{exp=1.2}]:0.999
		early_stopping: True
		n_iter_no_change: Scalar{Cl(5,50)}[sigma=Log{exp=1.2}]:27.5

	3. "mlp dist 2 layer" ::

		hidden_layer_sizes: Array{int,Cl(1,300)}[sigma=50]:[100 100]
		activation: TransitionChoice(choices=Tuple(identity,logistic,tanh,relu),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):tanh
		alpha: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		batch_size: TransitionChoice(choices=Tuple(auto,Scalar{int,Cl(50,400)}[sigma=Log{exp=1.2}]),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):auto
		learning_rate: TransitionChoice(choices=Tuple(constant,invscaling,adaptive),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):invscaling
		learning_rate_init: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		max_iter: Scalar{int,Cl(100,1000)}[sigma=Log{exp=1.2}]:200
		beta_1: Scalar{Cl(0.1,0.99)}[sigma=Log{exp=1.2}]:0.9
		beta_2: Scalar{Cl(0.1,0.9999)}[sigma=Log{exp=1.2}]:0.999

	4. "mlp dist es 2 layer" ::

		hidden_layer_sizes: Scalar{int,Cl(2,300)}[sigma=Log{exp=1.2}]:100
		activation: TransitionChoice(choices=Tuple(identity,logistic,tanh,relu),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):tanh
		alpha: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		batch_size: TransitionChoice(choices=Tuple(auto,Scalar{int,Cl(50,400)}[sigma=Log{exp=1.2}]),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):auto
		learning_rate: TransitionChoice(choices=Tuple(constant,invscaling,adaptive),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):invscaling
		learning_rate_init: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		max_iter: Scalar{int,Cl(100,1000)}[sigma=Log{exp=1.2}]:200
		beta_1: Scalar{Cl(0.1,0.99)}[sigma=Log{exp=1.2}]:0.9
		beta_2: Scalar{Cl(0.1,0.9999)}[sigma=Log{exp=1.2}]:0.999
		early_stopping: True
		n_iter_no_change: Scalar{Cl(5,50)}[sigma=Log{exp=1.2}]:27.5

	5. "mlp dist 1 layer" ::

		hidden_layer_sizes: Scalar{int,Cl(2,300)}[sigma=Log{exp=1.2}]:100
		activation: TransitionChoice(choices=Tuple(identity,logistic,tanh,relu),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):tanh
		alpha: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		batch_size: TransitionChoice(choices=Tuple(auto,Scalar{int,Cl(50,400)}[sigma=Log{exp=1.2}]),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):auto
		learning_rate: TransitionChoice(choices=Tuple(constant,invscaling,adaptive),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):invscaling
		learning_rate_init: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		max_iter: Scalar{int,Cl(100,1000)}[sigma=Log{exp=1.2}]:200
		beta_1: Scalar{Cl(0.1,0.99)}[sigma=Log{exp=1.2}]:0.9
		beta_2: Scalar{Cl(0.1,0.9999)}[sigma=Log{exp=1.2}]:0.999

	6. "mlp dist es 1 layer" ::

		hidden_layer_sizes: Scalar{int,Cl(2,300)}[sigma=Log{exp=1.2}]:100
		activation: TransitionChoice(choices=Tuple(identity,logistic,tanh,relu),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):tanh
		alpha: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		batch_size: TransitionChoice(choices=Tuple(auto,Scalar{int,Cl(50,400)}[sigma=Log{exp=1.2}]),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):auto
		learning_rate: TransitionChoice(choices=Tuple(constant,invscaling,adaptive),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):invscaling
		learning_rate_init: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		max_iter: Scalar{int,Cl(100,1000)}[sigma=Log{exp=1.2}]:200
		beta_1: Scalar{Cl(0.1,0.99)}[sigma=Log{exp=1.2}]:0.9
		beta_2: Scalar{Cl(0.1,0.9999)}[sigma=Log{exp=1.2}]:0.999
		early_stopping: True
		n_iter_no_change: Scalar{Cl(5,50)}[sigma=Log{exp=1.2}]:27.5


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

	0. "base rf" ::

		n_estimators: 100

	1. "rf classifier dist" ::

		n_estimators: Scalar{int,Cl(3,500)}[sigma=Log{exp=1.2}]:100
		max_depth: TransitionChoice(choices=Tuple(None,Scalar{int,Cl(2,200)}[sigma=Log{exp=1.2}]),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None
		max_features: Scalar{Cl(0,1)}[sigma=Log{exp=1.2}]:0.5
		min_samples_split: Scalar{Cl(0,1)}[sigma=Log{exp=1.2}]:0.5
		bootstrap: True
		class_weight: TransitionChoice(choices=Tuple(None,balanced),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None


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
		C: Log{exp=21.544346900318843,Cl(1e-05,1000)}:0.1
		class_weight: TransitionChoice(choices=Tuple(None,balanced),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None

	2. "ridge C extra" ::

		max_iter: Scalar{int,Cl(1000,10000)}[sigma=Log{exp=1.2}]:5500
		solver: saga
		C: Log{exp=21.544346900318843,Cl(1e-05,1000)}:0.1
		class_weight: TransitionChoice(choices=Tuple(None,balanced),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None
		tol: Log{exp=4.641588833612779,Cl(1e-06,0.01)}:0.0001


"sgd classifier"
****************

  Base Class Documenation: :class:`sklearn.linear_model.SGDClassifier`

  Param Distributions

	0. "base sgd" ::

		loss: hinge

	1. "sgd classifier" ::

		loss: TransitionChoice(choices=Tuple(hinge,log,modified_huber,squared_hinge,perceptron),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):modified_huber
		penalty: TransitionChoice(choices=Tuple(l2,l1,elasticnet),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):l1
		alpha: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		l1_ratio: Scalar{Cl(0,1)}[sigma=Log{exp=1.2}]:0.5
		max_iter: 5000
		learning_rate: TransitionChoice(choices=Tuple(optimal,invscaling,adaptive,constant),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):adaptive
		eta0: Log{exp=31.62277660168379,Cl(1e-06,1000)}:0.03162277660168379
		power_t: Scalar{Cl(0.1,0.9)}[sigma=Log{exp=1.2}]:0.5
		early_stopping: TransitionChoice(choices=Tuple(False,True),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):False
		validation_fraction: Scalar{Cl(0.05,0.5)}[sigma=Log{exp=1.2}]:0.275
		n_iter_no_change: TransitionChoice(choices=Tuple(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):11
		class_weight: TransitionChoice(choices=Tuple(None,balanced),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None


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
		gamma: Log{exp=9.999999999999998,Cl(1e-06,1)}:0.001
		C: Log{exp=21.544346900318843,Cl(0.0001,10000)}:1.0
		probability: True
		class_weight: TransitionChoice(choices=Tuple(None,balanced),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None


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
		n_estimators: Scalar{int,Cl(3,500)}[sigma=Log{exp=1.2}]:100
		min_child_weight: Log{exp=31.62277660168379,Cl(1e-05,10000)}:0.31622776601683794
		subsample: Scalar{Cl(0.3,0.95)}[sigma=Log{exp=1.2}]:0.625
		colsample_bytree: Scalar{Cl(0.3,0.95)}[sigma=Log{exp=1.2}]:0.625
		reg_alpha: ng.p.TransitionChoice([0, ng.p.Log(lower=1e-5, upper=1)])
		reg_lambda: ng.p.TransitionChoice([0, ng.p.Log(lower=1e-5, upper=1)])

	2. "xgb classifier dist2" ::

		verbosity: 0
		objective: binary:logistic
		max_depth: TransitionChoice(choices=Tuple(None,Scalar{int,Cl(2,200)}[sigma=Log{exp=1.2}]),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None
		learning_rate: Scalar{Cl(0.01,0.5)}[sigma=Log{exp=1.2}]:0.255
		n_estimators: Scalar{int,Cl(3,500)}[sigma=Log{exp=1.2}]:252
		min_child_weight: TransitionChoice(choices=Tuple(1,5,10,50),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):10
		subsample: Scalar{Cl(0.5,1)}[sigma=Log{exp=1.2}]:0.75
		colsample_bytree: Scalar{Cl(0.4,0.95)}[sigma=Log{exp=1.2}]:0.675

	3. "xgb classifier dist3" ::

		verbosity: 0
		objective: binary:logistic
		learning_rare: Scalar{Cl(0.005,0.3)}[sigma=Log{exp=1.2}]:0.1525
		min_child_weight: Scalar{Cl(0.5,10)}[sigma=Log{exp=1.2}]:5.25
		max_depth: TransitionChoice(choices=Tuple(3,4,5,6,7,8,9),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):6
		subsample: Scalar{Cl(0.5,1)}[sigma=Log{exp=1.2}]:0.75
		colsample_bytree: Scalar{Cl(0.5,1)}[sigma=Log{exp=1.2}]:0.75
		reg_alpha: Log{exp=6.812920690579612,Cl(1e-05,1)}:0.0031622776601683794



regression
==========
"dt regressor"
**************

  Base Class Documenation: :class:`sklearn.tree.DecisionTreeRegressor`

  Param Distributions

	0. "base dt" ::

		defaults only

	1. "dt dist" ::

		max_depth: Scalar{int,Cl(1,30)}[sigma=Log{exp=1.2}]:16
		min_samples_split: Scalar{int,Cl(2,50)}[sigma=Log{exp=1.2}]:26


"elastic net regressor"
***********************

  Base Class Documenation: :class:`sklearn.linear_model.ElasticNet`

  Param Distributions

	0. "base elastic net" ::

		max_iter: 5000

	1. "elastic regression" ::

		max_iter: 5000
		alpha: Log{exp=46.415888336127786,Cl(1e-05,100000)}:1.0
		l1_ratio: Scalar{Cl(0.01,1)}[sigma=Log{exp=1.2}]:0.505

	2. "elastic regression extra" ::

		max_iter: Scalar{int,Cl(1000,10000)}[sigma=Log{exp=1.2}]:5500
		alpha: Log{exp=46.415888336127786,Cl(1e-05,100000)}:1.0
		l1_ratio: Scalar{Cl(0.01,1)}[sigma=Log{exp=1.2}]:0.505
		tol: Log{exp=4.641588833612779,Cl(1e-06,0.01)}:0.0001


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

	0. "base knn" ::

		n_neighbors: 5

	1. "knn dist" ::

		weights: TransitionChoice(choices=Tuple(uniform,distance),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):uniform
		n_neighbors: Scalar{int,Cl(2,25)}[sigma=Log{exp=1.2}]:14


"lasso regressor"
*****************

  Base Class Documenation: :class:`sklearn.linear_model.Lasso`

  Param Distributions

	0. "base lasso regressor" ::

		max_iter: 5000

	1. "lasso regressor dist" ::

		max_iter: 5000
		alpha: Log{exp=46.415888336127786,Cl(1e-05,100000)}:1.0


"light gbm regressor"
*********************

  Base Class Documenation: :class:`lightgbm.LGBMRegressor`

  Param Distributions

	0. "base lgbm" ::

		silent: True

	1. "lgbm dist1" ::

		silent: True
		boosting_type: TransitionChoice(choices=Tuple(gbdt,dart,goss),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):dart
		n_estimators: Scalar{int,Cl(3,500)}[sigma=Log{exp=1.2}]:100
		num_leaves: Scalar{int,Cl(6,80)}[sigma=Log{exp=1.2}]:20
		min_child_samples: Scalar{int,Cl(10,500)}[sigma=Log{exp=1.2}]:255
		min_child_weight: Log{exp=31.62277660168379,Cl(1e-05,10000)}:0.31622776601683794
		subsample: Scalar{Cl(0.3,0.95)}[sigma=Log{exp=1.2}]:0.625
		colsample_bytree: Scalar{Cl(0.3,0.95)}[sigma=Log{exp=1.2}]:0.625
		reg_alpha: TransitionChoice(choices=Tuple(0,Log{exp=6.812920690579612,Cl(1e-05,1)}),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):0
		reg_lambda: TransitionChoice(choices=Tuple(0,Log{exp=6.812920690579612,Cl(1e-05,1)}),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):0

	2. "lgbm dist2" ::

		silent: True
		lambda_l2: 0.001
		boosting_type: TransitionChoice(choices=Tuple(gbdt,dart),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):gbdt
		min_child_samples: TransitionChoice(choices=Tuple(1,5,7,10,15,20,35,50,100,200,500,1000),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):35
		num_leaves: TransitionChoice(choices=Tuple(2,4,7,10,15,20,25,30,35,40,50,65,80,100,125,150,200,250),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):40
		colsample_bytree: TransitionChoice(choices=Tuple(0.7,0.9,1.0),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):0.9
		subsample: Scalar{Cl(0.3,1)}[sigma=Log{exp=1.2}]:0.65
		learning_rate: TransitionChoice(choices=Tuple(0.01,0.05,0.1),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):0.05
		n_estimators: TransitionChoice(choices=Tuple(5,20,35,50,75,100,150,200,350,500,750,1000),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):150


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
		C: Log{exp=21.544346900318843,Cl(0.0001,10000)}:1.0


"mlp regressor"
***************

  Base Class Documenation: :class:`BPt.extensions.MLP.MLPRegressor_Wrapper`

  Param Distributions

	0. "base mlp" ::

		defaults only

	1. "mlp dist 3 layer" ::

		hidden_layer_sizes: Array{int,Cl(1,300)}[sigma=50]:[100 100 100]
		activation: TransitionChoice(choices=Tuple(identity,logistic,tanh,relu),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):tanh
		alpha: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		batch_size: TransitionChoice(choices=Tuple(auto,Scalar{int,Cl(50,400)}[sigma=Log{exp=1.2}]),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):auto
		learning_rate: TransitionChoice(choices=Tuple(constant,invscaling,adaptive),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):invscaling
		learning_rate_init: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		max_iter: Scalar{int,Cl(100,1000)}[sigma=Log{exp=1.2}]:200
		beta_1: Scalar{Cl(0.1,0.99)}[sigma=Log{exp=1.2}]:0.9
		beta_2: Scalar{Cl(0.1,0.9999)}[sigma=Log{exp=1.2}]:0.999

	2. "mlp dist es 3 layer" ::

		hidden_layer_sizes: Scalar{int,Cl(2,300)}[sigma=Log{exp=1.2}]:100
		activation: TransitionChoice(choices=Tuple(identity,logistic,tanh,relu),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):tanh
		alpha: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		batch_size: TransitionChoice(choices=Tuple(auto,Scalar{int,Cl(50,400)}[sigma=Log{exp=1.2}]),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):auto
		learning_rate: TransitionChoice(choices=Tuple(constant,invscaling,adaptive),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):invscaling
		learning_rate_init: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		max_iter: Scalar{int,Cl(100,1000)}[sigma=Log{exp=1.2}]:200
		beta_1: Scalar{Cl(0.1,0.99)}[sigma=Log{exp=1.2}]:0.9
		beta_2: Scalar{Cl(0.1,0.9999)}[sigma=Log{exp=1.2}]:0.999
		early_stopping: True
		n_iter_no_change: Scalar{Cl(5,50)}[sigma=Log{exp=1.2}]:27.5

	3. "mlp dist 2 layer" ::

		hidden_layer_sizes: Array{int,Cl(1,300)}[sigma=50]:[100 100]
		activation: TransitionChoice(choices=Tuple(identity,logistic,tanh,relu),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):tanh
		alpha: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		batch_size: TransitionChoice(choices=Tuple(auto,Scalar{int,Cl(50,400)}[sigma=Log{exp=1.2}]),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):auto
		learning_rate: TransitionChoice(choices=Tuple(constant,invscaling,adaptive),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):invscaling
		learning_rate_init: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		max_iter: Scalar{int,Cl(100,1000)}[sigma=Log{exp=1.2}]:200
		beta_1: Scalar{Cl(0.1,0.99)}[sigma=Log{exp=1.2}]:0.9
		beta_2: Scalar{Cl(0.1,0.9999)}[sigma=Log{exp=1.2}]:0.999

	4. "mlp dist es 2 layer" ::

		hidden_layer_sizes: Scalar{int,Cl(2,300)}[sigma=Log{exp=1.2}]:100
		activation: TransitionChoice(choices=Tuple(identity,logistic,tanh,relu),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):tanh
		alpha: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		batch_size: TransitionChoice(choices=Tuple(auto,Scalar{int,Cl(50,400)}[sigma=Log{exp=1.2}]),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):auto
		learning_rate: TransitionChoice(choices=Tuple(constant,invscaling,adaptive),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):invscaling
		learning_rate_init: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		max_iter: Scalar{int,Cl(100,1000)}[sigma=Log{exp=1.2}]:200
		beta_1: Scalar{Cl(0.1,0.99)}[sigma=Log{exp=1.2}]:0.9
		beta_2: Scalar{Cl(0.1,0.9999)}[sigma=Log{exp=1.2}]:0.999
		early_stopping: True
		n_iter_no_change: Scalar{Cl(5,50)}[sigma=Log{exp=1.2}]:27.5

	5. "mlp dist 1 layer" ::

		hidden_layer_sizes: Scalar{int,Cl(2,300)}[sigma=Log{exp=1.2}]:100
		activation: TransitionChoice(choices=Tuple(identity,logistic,tanh,relu),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):tanh
		alpha: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		batch_size: TransitionChoice(choices=Tuple(auto,Scalar{int,Cl(50,400)}[sigma=Log{exp=1.2}]),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):auto
		learning_rate: TransitionChoice(choices=Tuple(constant,invscaling,adaptive),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):invscaling
		learning_rate_init: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		max_iter: Scalar{int,Cl(100,1000)}[sigma=Log{exp=1.2}]:200
		beta_1: Scalar{Cl(0.1,0.99)}[sigma=Log{exp=1.2}]:0.9
		beta_2: Scalar{Cl(0.1,0.9999)}[sigma=Log{exp=1.2}]:0.999

	6. "mlp dist es 1 layer" ::

		hidden_layer_sizes: Scalar{int,Cl(2,300)}[sigma=Log{exp=1.2}]:100
		activation: TransitionChoice(choices=Tuple(identity,logistic,tanh,relu),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):tanh
		alpha: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		batch_size: TransitionChoice(choices=Tuple(auto,Scalar{int,Cl(50,400)}[sigma=Log{exp=1.2}]),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):auto
		learning_rate: TransitionChoice(choices=Tuple(constant,invscaling,adaptive),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):invscaling
		learning_rate_init: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		max_iter: Scalar{int,Cl(100,1000)}[sigma=Log{exp=1.2}]:200
		beta_1: Scalar{Cl(0.1,0.99)}[sigma=Log{exp=1.2}]:0.9
		beta_2: Scalar{Cl(0.1,0.9999)}[sigma=Log{exp=1.2}]:0.999
		early_stopping: True
		n_iter_no_change: Scalar{Cl(5,50)}[sigma=Log{exp=1.2}]:27.5


"random forest regressor"
*************************

  Base Class Documenation: :class:`sklearn.ensemble.RandomForestRegressor`

  Param Distributions

	0. "base rf" ::

		n_estimators: 100

	1. "rf dist" ::

		n_estimators: Scalar{int,Cl(3,500)}[sigma=Log{exp=1.2}]:100
		max_depth: TransitionChoice(choices=Tuple(None,Scalar{int,Cl(2,200)}[sigma=Log{exp=1.2}]),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None
		max_features: Scalar{Cl(0,1)}[sigma=Log{exp=1.2}]:0.5
		min_samples_split: Scalar{Cl(0,1)}[sigma=Log{exp=1.2}]:0.5
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
		alpha: Log{exp=21.544346900318843,Cl(0.001,100000)}:10.0


"svm regressor"
***************

  Base Class Documenation: :class:`sklearn.svm.SVR`

  Param Distributions

	0. "base svm" ::

		kernel: rbf
		gamma: scale

	1. "svm dist" ::

		kernel: rbf
		gamma: Log{exp=9.999999999999998,Cl(1e-06,1)}:0.001
		C: Log{exp=21.544346900318843,Cl(0.0001,10000)}:1.0


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
		n_estimators: Scalar{int,Cl(3,500)}[sigma=Log{exp=1.2}]:100
		min_child_weight: Log{exp=31.62277660168379,Cl(1e-05,10000)}:0.31622776601683794
		subsample: Scalar{Cl(0.3,0.95)}[sigma=Log{exp=1.2}]:0.625
		colsample_bytree: Scalar{Cl(0.3,0.95)}[sigma=Log{exp=1.2}]:0.625
		reg_alpha: ng.p.TransitionChoice([0, ng.p.Log(lower=1e-5, upper=1)])
		reg_lambda: ng.p.TransitionChoice([0, ng.p.Log(lower=1e-5, upper=1)])

	2. "xgb dist2" ::

		verbosity: 0
		objective: reg:squarederror
		max_depth: TransitionChoice(choices=Tuple(None,Scalar{int,Cl(2,200)}[sigma=Log{exp=1.2}]),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None
		learning_rate: Scalar{Cl(0.01,0.5)}[sigma=Log{exp=1.2}]:0.255
		n_estimators: Scalar{int,Cl(3,500)}[sigma=Log{exp=1.2}]:252
		min_child_weight: TransitionChoice(choices=Tuple(1,5,10,50),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):10
		subsample: Scalar{Cl(0.5,1)}[sigma=Log{exp=1.2}]:0.75
		colsample_bytree: Scalar{Cl(0.4,0.95)}[sigma=Log{exp=1.2}]:0.675

	3. "xgb dist3" ::

		verbosity: 0
		objective: reg:squarederror
		learning_rare: Scalar{Cl(0.005,0.3)}[sigma=Log{exp=1.2}]:0.1525
		min_child_weight: Scalar{Cl(0.5,10)}[sigma=Log{exp=1.2}]:5.25
		max_depth: TransitionChoice(choices=Tuple(3,4,5,6,7,8,9),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):6
		subsample: Scalar{Cl(0.5,1)}[sigma=Log{exp=1.2}]:0.75
		colsample_bytree: Scalar{Cl(0.5,1)}[sigma=Log{exp=1.2}]:0.75
		reg_alpha: Log{exp=6.812920690579612,Cl(1e-05,1)}:0.0031622776601683794



categorical
===========
"dt classifier"
***************

  Base Class Documenation: :class:`sklearn.tree.DecisionTreeClassifier`

  Param Distributions

	0. "base dt" ::

		defaults only

	1. "dt classifier dist" ::

		max_depth: Scalar{int,Cl(1,30)}[sigma=Log{exp=1.2}]:16
		min_samples_split: Scalar{int,Cl(2,50)}[sigma=Log{exp=1.2}]:26
		class_weight: TransitionChoice(choices=Tuple(None,balanced),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None


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
		class_weight: TransitionChoice(choices=Tuple(None,balanced),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None
		solver: saga
		l1_ratio: Scalar{Cl(0.01,1)}[sigma=Log{exp=1.2}]:0.505
		C: Log{exp=46.415888336127786,Cl(1e-05,100000)}:1.0

	2. "elastic clf v2" ::

		max_iter: 5000
		multi_class: auto
		penalty: elasticnet
		class_weight: TransitionChoice(choices=Tuple(None,balanced),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None
		solver: saga
		l1_ratio: Scalar{Cl(0.01,1)}[sigma=Log{exp=1.2}]:0.505
		C: Log{exp=14.677992676220699,Cl(0.01,100000)}:31.622776601683793

	3. "elastic classifier extra" ::

		max_iter: Scalar{int,Cl(1000,10000)}[sigma=Log{exp=1.2}]:5500
		multi_class: auto
		penalty: elasticnet
		class_weight: TransitionChoice(choices=Tuple(None,balanced),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None
		solver: saga
		l1_ratio: Scalar{Cl(0.01,1)}[sigma=Log{exp=1.2}]:0.505
		C: Log{exp=46.415888336127786,Cl(1e-05,100000)}:1.0
		tol: Log{exp=4.641588833612779,Cl(1e-06,0.01)}:0.0001


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

		var_smoothing: 1e-09


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

		weights: TransitionChoice(choices=Tuple(uniform,distance),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):uniform
		n_neighbors: Scalar{int,Cl(2,25)}[sigma=Log{exp=1.2}]:14


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
		class_weight: TransitionChoice(choices=Tuple(None,balanced),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None
		solver: liblinear
		C: Log{exp=21.544346900318843,Cl(1e-05,1000)}:0.1

	2. "lasso C extra" ::

		max_iter: Scalar{int,Cl(1000,10000)}[sigma=Log{exp=1.2}]:5500
		multi_class: auto
		penalty: l1
		class_weight: TransitionChoice(choices=Tuple(None,balanced),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None
		solver: liblinear
		C: Log{exp=21.544346900318843,Cl(1e-05,1000)}:0.1
		tol: Log{exp=4.641588833612779,Cl(1e-06,0.01)}:0.0001


"light gbm classifier"
**********************

  Base Class Documenation: :class:`lightgbm.LGBMClassifier`

  Param Distributions

	0. "base lgbm" ::

		silent: True

	1. "lgbm classifier dist1" ::

		silent: True
		boosting_type: TransitionChoice(choices=Tuple(gbdt,dart,goss),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):dart
		n_estimators: Scalar{int,Cl(3,500)}[sigma=Log{exp=1.2}]:100
		num_leaves: Scalar{int,Cl(6,80)}[sigma=Log{exp=1.2}]:20
		min_child_samples: Scalar{int,Cl(10,500)}[sigma=Log{exp=1.2}]:255
		min_child_weight: Log{exp=31.62277660168379,Cl(1e-05,10000)}:0.31622776601683794
		subsample: Scalar{Cl(0.3,0.95)}[sigma=Log{exp=1.2}]:0.625
		colsample_bytree: Scalar{Cl(0.3,0.95)}[sigma=Log{exp=1.2}]:0.625
		reg_alpha: TransitionChoice(choices=Tuple(0,Log{exp=6.812920690579612,Cl(1e-05,1)}),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):0
		reg_lambda: TransitionChoice(choices=Tuple(0,Log{exp=6.812920690579612,Cl(1e-05,1)}),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):0
		class_weight: TransitionChoice(choices=Tuple(None,balanced),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None

	2. "lgbm classifier dist2" ::

		silent: True
		lambda_l2: 0.001
		boosting_type: TransitionChoice(choices=Tuple(gbdt,dart),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):gbdt
		min_child_samples: TransitionChoice(choices=Tuple(1,5,7,10,15,20,35,50,100,200,500,1000),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):35
		num_leaves: TransitionChoice(choices=Tuple(2,4,7,10,15,20,25,30,35,40,50,65,80,100,125,150,200,250),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):40
		colsample_bytree: TransitionChoice(choices=Tuple(0.7,0.9,1.0),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):0.9
		subsample: Scalar{Cl(0.3,1)}[sigma=Log{exp=1.2}]:0.65
		learning_rate: TransitionChoice(choices=Tuple(0.01,0.05,0.1),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):0.05
		n_estimators: TransitionChoice(choices=Tuple(5,20,35,50,75,100,150,200,350,500,750,1000),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):150
		class_weight: TransitionChoice(choices=Tuple(None,balanced),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None


"linear svm classifier"
***********************

  Base Class Documenation: :class:`sklearn.svm.LinearSVC`

  Param Distributions

	0. "base linear svc" ::

		max_iter: 5000

	1. "linear svc dist" ::

		max_iter: 5000
		C: Log{exp=21.544346900318843,Cl(0.0001,10000)}:1.0
		class_weight: TransitionChoice(choices=Tuple(None,balanced),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None


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

  Base Class Documenation: :class:`ABCD_ML.extensions.MLP.MLPClassifier_Wrapper`

  Param Distributions

	0. "base mlp" ::

		defaults only

	1. "mlp dist 3 layer" ::

		hidden_layer_sizes: Array{int,Cl(1,300)}[sigma=50]:[100 100 100]
		activation: TransitionChoice(choices=Tuple(identity,logistic,tanh,relu),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):tanh
		alpha: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		batch_size: TransitionChoice(choices=Tuple(auto,Scalar{int,Cl(50,400)}[sigma=Log{exp=1.2}]),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):auto
		learning_rate: TransitionChoice(choices=Tuple(constant,invscaling,adaptive),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):invscaling
		learning_rate_init: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		max_iter: Scalar{int,Cl(100,1000)}[sigma=Log{exp=1.2}]:200
		beta_1: Scalar{Cl(0.1,0.99)}[sigma=Log{exp=1.2}]:0.9
		beta_2: Scalar{Cl(0.1,0.9999)}[sigma=Log{exp=1.2}]:0.999

	2. "mlp dist es 3 layer" ::

		hidden_layer_sizes: Scalar{int,Cl(2,300)}[sigma=Log{exp=1.2}]:100
		activation: TransitionChoice(choices=Tuple(identity,logistic,tanh,relu),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):tanh
		alpha: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		batch_size: TransitionChoice(choices=Tuple(auto,Scalar{int,Cl(50,400)}[sigma=Log{exp=1.2}]),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):auto
		learning_rate: TransitionChoice(choices=Tuple(constant,invscaling,adaptive),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):invscaling
		learning_rate_init: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		max_iter: Scalar{int,Cl(100,1000)}[sigma=Log{exp=1.2}]:200
		beta_1: Scalar{Cl(0.1,0.99)}[sigma=Log{exp=1.2}]:0.9
		beta_2: Scalar{Cl(0.1,0.9999)}[sigma=Log{exp=1.2}]:0.999
		early_stopping: True
		n_iter_no_change: Scalar{Cl(5,50)}[sigma=Log{exp=1.2}]:27.5

	3. "mlp dist 2 layer" ::

		hidden_layer_sizes: Array{int,Cl(1,300)}[sigma=50]:[100 100]
		activation: TransitionChoice(choices=Tuple(identity,logistic,tanh,relu),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):tanh
		alpha: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		batch_size: TransitionChoice(choices=Tuple(auto,Scalar{int,Cl(50,400)}[sigma=Log{exp=1.2}]),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):auto
		learning_rate: TransitionChoice(choices=Tuple(constant,invscaling,adaptive),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):invscaling
		learning_rate_init: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		max_iter: Scalar{int,Cl(100,1000)}[sigma=Log{exp=1.2}]:200
		beta_1: Scalar{Cl(0.1,0.99)}[sigma=Log{exp=1.2}]:0.9
		beta_2: Scalar{Cl(0.1,0.9999)}[sigma=Log{exp=1.2}]:0.999

	4. "mlp dist es 2 layer" ::

		hidden_layer_sizes: Scalar{int,Cl(2,300)}[sigma=Log{exp=1.2}]:100
		activation: TransitionChoice(choices=Tuple(identity,logistic,tanh,relu),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):tanh
		alpha: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		batch_size: TransitionChoice(choices=Tuple(auto,Scalar{int,Cl(50,400)}[sigma=Log{exp=1.2}]),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):auto
		learning_rate: TransitionChoice(choices=Tuple(constant,invscaling,adaptive),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):invscaling
		learning_rate_init: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		max_iter: Scalar{int,Cl(100,1000)}[sigma=Log{exp=1.2}]:200
		beta_1: Scalar{Cl(0.1,0.99)}[sigma=Log{exp=1.2}]:0.9
		beta_2: Scalar{Cl(0.1,0.9999)}[sigma=Log{exp=1.2}]:0.999
		early_stopping: True
		n_iter_no_change: Scalar{Cl(5,50)}[sigma=Log{exp=1.2}]:27.5

	5. "mlp dist 1 layer" ::

		hidden_layer_sizes: Scalar{int,Cl(2,300)}[sigma=Log{exp=1.2}]:100
		activation: TransitionChoice(choices=Tuple(identity,logistic,tanh,relu),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):tanh
		alpha: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		batch_size: TransitionChoice(choices=Tuple(auto,Scalar{int,Cl(50,400)}[sigma=Log{exp=1.2}]),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):auto
		learning_rate: TransitionChoice(choices=Tuple(constant,invscaling,adaptive),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):invscaling
		learning_rate_init: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		max_iter: Scalar{int,Cl(100,1000)}[sigma=Log{exp=1.2}]:200
		beta_1: Scalar{Cl(0.1,0.99)}[sigma=Log{exp=1.2}]:0.9
		beta_2: Scalar{Cl(0.1,0.9999)}[sigma=Log{exp=1.2}]:0.999

	6. "mlp dist es 1 layer" ::

		hidden_layer_sizes: Scalar{int,Cl(2,300)}[sigma=Log{exp=1.2}]:100
		activation: TransitionChoice(choices=Tuple(identity,logistic,tanh,relu),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):tanh
		alpha: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		batch_size: TransitionChoice(choices=Tuple(auto,Scalar{int,Cl(50,400)}[sigma=Log{exp=1.2}]),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):auto
		learning_rate: TransitionChoice(choices=Tuple(constant,invscaling,adaptive),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):invscaling
		learning_rate_init: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		max_iter: Scalar{int,Cl(100,1000)}[sigma=Log{exp=1.2}]:200
		beta_1: Scalar{Cl(0.1,0.99)}[sigma=Log{exp=1.2}]:0.9
		beta_2: Scalar{Cl(0.1,0.9999)}[sigma=Log{exp=1.2}]:0.999
		early_stopping: True
		n_iter_no_change: Scalar{Cl(5,50)}[sigma=Log{exp=1.2}]:27.5


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

	0. "base rf" ::

		n_estimators: 100

	1. "rf classifier dist" ::

		n_estimators: Scalar{int,Cl(3,500)}[sigma=Log{exp=1.2}]:100
		max_depth: TransitionChoice(choices=Tuple(None,Scalar{int,Cl(2,200)}[sigma=Log{exp=1.2}]),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None
		max_features: Scalar{Cl(0,1)}[sigma=Log{exp=1.2}]:0.5
		min_samples_split: Scalar{Cl(0,1)}[sigma=Log{exp=1.2}]:0.5
		bootstrap: True
		class_weight: TransitionChoice(choices=Tuple(None,balanced),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None


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
		C: Log{exp=21.544346900318843,Cl(1e-05,1000)}:0.1
		class_weight: TransitionChoice(choices=Tuple(None,balanced),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None

	2. "ridge C extra" ::

		max_iter: Scalar{int,Cl(1000,10000)}[sigma=Log{exp=1.2}]:5500
		solver: saga
		C: Log{exp=21.544346900318843,Cl(1e-05,1000)}:0.1
		class_weight: TransitionChoice(choices=Tuple(None,balanced),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None
		tol: Log{exp=4.641588833612779,Cl(1e-06,0.01)}:0.0001


"sgd classifier"
****************

  Base Class Documenation: :class:`sklearn.linear_model.SGDClassifier`

  Param Distributions

	0. "base sgd" ::

		loss: hinge

	1. "sgd classifier" ::

		loss: TransitionChoice(choices=Tuple(hinge,log,modified_huber,squared_hinge,perceptron),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):modified_huber
		penalty: TransitionChoice(choices=Tuple(l2,l1,elasticnet),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):l1
		alpha: Log{exp=14.677992676220699,Cl(1e-05,100)}:0.03162277660168379
		l1_ratio: Scalar{Cl(0,1)}[sigma=Log{exp=1.2}]:0.5
		max_iter: 5000
		learning_rate: TransitionChoice(choices=Tuple(optimal,invscaling,adaptive,constant),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):adaptive
		eta0: Log{exp=31.62277660168379,Cl(1e-06,1000)}:0.03162277660168379
		power_t: Scalar{Cl(0.1,0.9)}[sigma=Log{exp=1.2}]:0.5
		early_stopping: TransitionChoice(choices=Tuple(False,True),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):False
		validation_fraction: Scalar{Cl(0.05,0.5)}[sigma=Log{exp=1.2}]:0.275
		n_iter_no_change: TransitionChoice(choices=Tuple(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):11
		class_weight: TransitionChoice(choices=Tuple(None,balanced),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None


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
		gamma: Log{exp=9.999999999999998,Cl(1e-06,1)}:0.001
		C: Log{exp=21.544346900318843,Cl(0.0001,10000)}:1.0
		probability: True
		class_weight: TransitionChoice(choices=Tuple(None,balanced),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None


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
		n_estimators: Scalar{int,Cl(3,500)}[sigma=Log{exp=1.2}]:100
		min_child_weight: Log{exp=31.62277660168379,Cl(1e-05,10000)}:0.31622776601683794
		subsample: Scalar{Cl(0.3,0.95)}[sigma=Log{exp=1.2}]:0.625
		colsample_bytree: Scalar{Cl(0.3,0.95)}[sigma=Log{exp=1.2}]:0.625
		reg_alpha: ng.p.TransitionChoice([0, ng.p.Log(lower=1e-5, upper=1)])
		reg_lambda: ng.p.TransitionChoice([0, ng.p.Log(lower=1e-5, upper=1)])

	2. "xgb classifier dist2" ::

		verbosity: 0
		objective: binary:logistic
		max_depth: TransitionChoice(choices=Tuple(None,Scalar{int,Cl(2,200)}[sigma=Log{exp=1.2}]),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):None
		learning_rate: Scalar{Cl(0.01,0.5)}[sigma=Log{exp=1.2}]:0.255
		n_estimators: Scalar{int,Cl(3,500)}[sigma=Log{exp=1.2}]:252
		min_child_weight: TransitionChoice(choices=Tuple(1,5,10,50),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):10
		subsample: Scalar{Cl(0.5,1)}[sigma=Log{exp=1.2}]:0.75
		colsample_bytree: Scalar{Cl(0.4,0.95)}[sigma=Log{exp=1.2}]:0.675

	3. "xgb classifier dist3" ::

		verbosity: 0
		objective: binary:logistic
		learning_rare: Scalar{Cl(0.005,0.3)}[sigma=Log{exp=1.2}]:0.1525
		min_child_weight: Scalar{Cl(0.5,10)}[sigma=Log{exp=1.2}]:5.25
		max_depth: TransitionChoice(choices=Tuple(3,4,5,6,7,8,9),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):6
		subsample: Scalar{Cl(0.5,1)}[sigma=Log{exp=1.2}]:0.75
		colsample_bytree: Scalar{Cl(0.5,1)}[sigma=Log{exp=1.2}]:0.75
		reg_alpha: Log{exp=6.812920690579612,Cl(1e-05,1)}:0.0031622776601683794



.. _Scorers:
 
*******
Scorers
*******

Different availible choices for the `scorer` parameter are shown below.
`scorer` is accepted by :class:`Problem_Spec<ABCD_ML.Problem_Spec>`, :class:`Param_Search<ABCD_ML.Param_Search>` and :class:`Feat_Importance<ABCD_ML.Feat_Importance>`
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


regression
==========
"explained_variance"
********************

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

"neg_mean_absolute_error"
*************************

  Base Func Documenation: :func:`sklearn.metrics.mean_absolute_error`

"neg_mean_squared_error"
************************

  Base Func Documenation: :func:`sklearn.metrics.mean_squared_error`

"neg_mean_squared_log_error"
****************************

  Base Func Documenation: :func:`sklearn.metrics.mean_squared_log_error`

"neg_root_mean_squared_error"
*****************************

  Base Func Documenation: :func:`sklearn.metrics.mean_squared_error`

"neg_mean_poisson_deviance"
***************************

  Base Func Documenation: :func:`sklearn.metrics.mean_poisson_deviance`

"neg_mean_gamma_deviance"
*************************

  Base Func Documenation: :func:`sklearn.metrics.mean_gamma_deviance`


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


.. _Loaders:
 
*******
Loaders
*******

Different base obj choices for the :class:`Loader<ABCD_ML.Loader>` are shown below
The exact str indicator, as passed to the `obj` param is represented by the sub-heading (within "")
Additionally, a link to the original models documentation as well as the implemented parameter distributions are shown.

All Problem Types
=================
"identity"
**********

  Base Class Documenation: :class:`ABCD_ML.extensions.Loaders.Identity`

  Param Distributions

	0. "default" ::

		defaults only


"surface rois"
**************

  Base Class Documenation: :class:`ABCD_ML.extensions.Loaders.SurfLabels`

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

  Base Class Documenation: :class:`ABCD_ML.extensions.Loaders.Connectivity`

  Param Distributions

	0. "default" ::

		defaults only



.. _Imputers:
 
********
Imputers
********

Different base obj choices for the :class:`Imputer<ABCD_ML.Imputer>` are shown below
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
		skip_complete: True



.. _Scalers:
 
*******
Scalers
*******

Different base obj choices for the :class:`Scaler<ABCD_ML.Scaler>` are shown below
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

		quantile_range: TransitionChoice(choices=Tuple((1, 99),(2, 98),(3, 97),(4, 96),(5, 95),(6, 94),(7, 93),(8, 92),(9, 91),(10, 90),(11, 89),(12, 88),(13, 87),(14, 86),(15, 85),(16, 84),(17, 83),(18, 82),(19, 81),(20, 80),(21, 79),(22, 78),(23, 77),(24, 76),(25, 75),(26, 74),(27, 73),(28, 72),(29, 71),(30, 70),(31, 69),(32, 68),(33, 67),(34, 66),(35, 65),(36, 64),(37, 63),(38, 62),(39, 61)),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):(20, 80)


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

  Base Class Documenation: :class:`ABCD_ML.extensions.Scalers.Winsorizer`

  Param Distributions

	0. "base winsorize" ::

		quantile_range: (1, 99)

	1. "winsorize gs" ::

		quantile_range: TransitionChoice(choices=Tuple((1, 99),(2, 98),(3, 97),(4, 96),(5, 95),(6, 94),(7, 93),(8, 92),(9, 91),(10, 90),(11, 89),(12, 88),(13, 87),(14, 86),(15, 85),(16, 84),(17, 83),(18, 82),(19, 81),(20, 80),(21, 79),(22, 78),(23, 77),(24, 76),(25, 75),(26, 74),(27, 73),(28, 72),(29, 71),(30, 70),(31, 69),(32, 68),(33, 67),(34, 66),(35, 65),(36, 64),(37, 63),(38, 62),(39, 61)),position=Scalar[sigma=Log{exp=1.2}],transitions=[1. 1.]):(20, 80)


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

Different base obj choices for the :class:`Transformer<ABCD_ML.Transformer>` are shown below
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

		n_components: Scalar{Cl(0.1,0.99)}[sigma=Log{exp=1.2}]:0.75
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


"truncated svd"
***************

  Base Class Documenation: :class:`sklearn.decomposition.TruncatedSVD`

  Param Distributions

	0. "default" ::

		defaults only


"one hot encoder"
*****************

  Base Class Documenation: :class:`category_encoders.one_hot.OneHotEncoder`

  Param Distributions

	0. "default" ::

		defaults only


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

Different base obj choices for the :class:`Feat_Selector<ABCD_ML.Feat_Selector>` are shown below
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

		n_features_to_select: Scalar{Cl(0.1,0.99)}[sigma=Log{exp=1.2}]:0.5


"selector"
**********

  Base Class Documenation: :class:`ABCD_ML.extensions.Feat_Selectors.FeatureSelector`

  Param Distributions

	0. "random" ::

		mask: sets as random features

	1. "searchable" ::

		mask: sets as hyperparameters


"univariate selection c"
************************

  Base Class Documenation: :class:`sklearn.feature_selection.SelectPercentile`

  Param Distributions

	0. "base univar fs classifier" ::

		score_func: <function f_classif at 0x7f2dcb147b90>
		percentile: 50

	1. "univar fs classifier dist" ::

		score_func: <function f_classif at 0x7f2dcb147b90>
		percentile: Scalar{Cl(1,99)}[sigma=Log{exp=1.2}]:50.0


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

		n_features_to_select: Scalar{Cl(0.1,0.99)}[sigma=Log{exp=1.2}]:0.5


"selector"
**********

  Base Class Documenation: :class:`ABCD_ML.extensions.Feat_Selectors.FeatureSelector`

  Param Distributions

	0. "random" ::

		mask: sets as random features

	1. "searchable" ::

		mask: sets as hyperparameters


"univariate selection r"
************************

  Base Class Documenation: :class:`sklearn.feature_selection.SelectPercentile`

  Param Distributions

	0. "base univar fs regression" ::

		score_func: <function f_regression at 0x7f2dcb147f80>
		percentile: 50

	1. "univar fs regression dist" ::

		score_func: <function f_regression at 0x7f2dcb147f80>
		percentile: Scalar{Cl(1,99)}[sigma=Log{exp=1.2}]:50.0


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

		n_features_to_select: Scalar{Cl(0.1,0.99)}[sigma=Log{exp=1.2}]:0.5


"selector"
**********

  Base Class Documenation: :class:`ABCD_ML.extensions.Feat_Selectors.FeatureSelector`

  Param Distributions

	0. "random" ::

		mask: sets as random features

	1. "searchable" ::

		mask: sets as hyperparameters


"univariate selection c"
************************

  Base Class Documenation: :class:`sklearn.feature_selection.SelectPercentile`

  Param Distributions

	0. "base univar fs classifier" ::

		score_func: <function f_classif at 0x7f2dcb147b90>
		percentile: 50

	1. "univar fs classifier dist" ::

		score_func: <function f_classif at 0x7f2dcb147b90>
		percentile: Scalar{Cl(1,99)}[sigma=Log{exp=1.2}]:50.0


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

Different base obj choices for the :class:`Ensemble<ABCD_ML.Ensemble>` are shown below
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

  Base Class Documenation: :class:`sklearn.ensemble.StackingClassifier`

  Param Distributions

	0. "default" ::

		defaults only


"voting classifier"
*******************

  Base Class Documenation: :class:`sklearn.ensemble.VotingClassifier`

  Param Distributions

	0. "default" ::

		defaults only



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

  Base Class Documenation: :class:`sklearn.ensemble.StackingRegressor`

  Param Distributions

	0. "default" ::

		defaults only


"voting regressor"
******************

  Base Class Documenation: :class:`sklearn.ensemble.VotingRegressor`

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

  Base Class Documenation: :class:`sklearn.ensemble.StackingClassifier`

  Param Distributions

	0. "default" ::

		defaults only


"voting classifier"
*******************

  Base Class Documenation: :class:`sklearn.ensemble.VotingClassifier`

  Param Distributions

	0. "default" ::

		defaults only



