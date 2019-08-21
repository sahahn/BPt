***********
Model Types
***********

binary
======
* dt classifier :class:`sklearn.tree.DecisionTreeClassifier`
* elastic net logistic :class:`sklearn.linear_model.LogisticRegression`
* gaussian nb :class:`sklearn.naive_bayes.GaussianNB`
* gp classifier :class:`sklearn.gaussian_process.GaussianProcessClassifier`
* knn classifier :class:`sklearn.neighbors.KNeighborsClassifier`
* lasso logistic :class:`sklearn.linear_model.LogisticRegression`
* light gbm classifier :class:`lightgbm.LGBMClassifier`
* logistic :class:`sklearn.linear_model.LogisticRegression`
* mlp classifier :class:`sklearn.neural_network.MLPClassifier`
* random forest classifier :class:`sklearn.ensemble.RandomForestClassifier`
* ridge logistic :class:`sklearn.linear_model.LogisticRegression`
* svm classifier :class:`sklearn.svm.SVC`

regression
==========
* dt regressor :class:`sklearn.tree.DecisionTreeRegressor`
* elastic net regressor :class:`sklearn.linear_model.ElasticNet`
* gp regressor :class:`sklearn.gaussian_process.GaussianProcessRegressor`
* knn regressor :class:`sklearn.neighbors.KNeighborsRegressor`
* light gbm regressor :class:`lightgbm.LGBMRegressor`
* light gbm regressor early stop :class:`ABCD_ML.Early_Stop.EarlyStopLGBMRegressor`
* linear regressor :class:`sklearn.linear_model.LinearRegression`
* mlp regressor :class:`sklearn.neural_network.MLPRegressor`
* random forest regressor :class:`sklearn.ensemble.RandomForestRegressor`
* svm regressor :class:`sklearn.svm.SVR`

categorical multilabel
======================
* dt classifier :class:`sklearn.tree.DecisionTreeClassifier`
* knn classifier :class:`sklearn.neighbors.KNeighborsClassifier`
* mlp classifier :class:`sklearn.neural_network.MLPClassifier`
* random forest classifier :class:`sklearn.ensemble.RandomForestClassifier`

categorical multiclass
======================
* dt classifier :class:`sklearn.tree.DecisionTreeClassifier`
* elastic net logistic :class:`sklearn.linear_model.LogisticRegression`
* gaussian nb :class:`sklearn.naive_bayes.GaussianNB`
* gp classifier :class:`sklearn.gaussian_process.GaussianProcessClassifier`
* knn classifier :class:`sklearn.neighbors.KNeighborsClassifier`
* lasso logistic :class:`sklearn.linear_model.LogisticRegression`
* light gbm classifier :class:`lightgbm.LGBMClassifier`
* logistic :class:`sklearn.linear_model.LogisticRegression`
* mlp classifier :class:`sklearn.neural_network.MLPClassifier`
* random forest classifier :class:`sklearn.ensemble.RandomForestClassifier`
* ridge logistic :class:`sklearn.linear_model.LogisticRegression`
* svm classifier :class:`sklearn.svm.SVC`

***********
Samplers
***********

binary
======
* adasyn :class:`imblearn.over_sampling.ADASYN`
* all knn :class:`imblearn.under_sampling.AllKNN`
* borderline smote :class:`imblearn.over_sampling.BorderlineSMOTE`
* cluster centroids :class:`imblearn.under_sampling.ClusterCentroids`
* condensed nn :class:`imblearn.under_sampling.CondensedNearestNeighbour`
* enn :class:`imblearn.under_sampling.EditedNearestNeighbours`
* kmeans smote :class:`imblearn.over_sampling.KMeansSMOTE`
* near miss :class:`imblearn.under_sampling.NearMiss`
* neighbourhood cleaning rule :class:`imblearn.under_sampling.NeighbourhoodCleaningRule`
* one sided selection :class:`imblearn.under_sampling.OneSidedSelection`
* random over sampler :class:`imblearn.over_sampling.RandomOverSampler`
* random under sampler :class:`imblearn.under_sampling.RandomUnderSampler`
* renn :class:`imblearn.under_sampling.RepeatedEditedNearestNeighbours`
* smote :class:`imblearn.over_sampling.SMOTE`
* smote enn :class:`imblearn.combine.SMOTEENN`
* smote nc :class:`imblearn.over_sampling.SMOTENC`
* smote tomek :class:`imblearn.combine.SMOTETomek`
* svm smote :class:`imblearn.over_sampling.SVMSMOTE`
* tomek links :class:`imblearn.under_sampling.TomekLinks`

regression
==========

categorical multilabel
======================

categorical multiclass
======================
* adasyn :class:`imblearn.over_sampling.ADASYN`
* all knn :class:`imblearn.under_sampling.AllKNN`
* borderline smote :class:`imblearn.over_sampling.BorderlineSMOTE`
* cluster centroids :class:`imblearn.under_sampling.ClusterCentroids`
* condensed nn :class:`imblearn.under_sampling.CondensedNearestNeighbour`
* enn :class:`imblearn.under_sampling.EditedNearestNeighbours`
* kmeans smote :class:`imblearn.over_sampling.KMeansSMOTE`
* near miss :class:`imblearn.under_sampling.NearMiss`
* neighbourhood cleaning rule :class:`imblearn.under_sampling.NeighbourhoodCleaningRule`
* one sided selection :class:`imblearn.under_sampling.OneSidedSelection`
* random over sampler :class:`imblearn.over_sampling.RandomOverSampler`
* random under sampler :class:`imblearn.under_sampling.RandomUnderSampler`
* renn :class:`imblearn.under_sampling.RepeatedEditedNearestNeighbours`
* smote :class:`imblearn.over_sampling.SMOTE`
* smote enn :class:`imblearn.combine.SMOTEENN`
* smote nc :class:`imblearn.over_sampling.SMOTENC`
* smote tomek :class:`imblearn.combine.SMOTETomek`
* svm smote :class:`imblearn.over_sampling.SVMSMOTE`
* tomek links :class:`imblearn.under_sampling.TomekLinks`

**************
Feat Selectors
**************

binary
======
* rfe :class:`ABCD_ML.Feature_Selectors.RFE`
* univariate selection classification :class:`sklearn.feature_selection.SelectPercentile`

regression
==========
* rfe :class:`ABCD_ML.Feature_Selectors.RFE`
* univariate selection regression :class:`sklearn.feature_selection.SelectPercentile`

categorical multilabel
======================

categorical multiclass
======================
* rfe :class:`ABCD_ML.Feature_Selectors.RFE`
* univariate selection classification :class:`sklearn.feature_selection.SelectPercentile`

***************
Ensemble Types
***************

binary
======
* aposteriori :class:`deslib.dcs.a_posteriori.APosteriori`
* apriori :class:`deslib.dcs.a_priori.APriori`
* des clustering :class:`deslib.des.des_clustering.DESClustering`
* des knn :class:`deslib.des.des_knn.DESKNN`
* deskl :class:`deslib.des.probabilistic.DESKL`
* desmi :class:`deslib.des.des_mi.DESMI`
* desp :class:`deslib.des.des_p.DESP`
* exponential :class:`deslib.des.probabilistic.Exponential`
* knop :class:`deslib.des.knop.KNOP`
* knorae :class:`deslib.des.knora_e.KNORAE`
* knrau :class:`deslib.des.knora_u.KNORAU`
* lca :class:`deslib.dcs.lca.LCA`
* logarithmic :class:`deslib.des.probabilistic.Logarithmic`
* mcb :class:`deslib.dcs.mcb.MCB`
* metades :class:`deslib.des.meta_des.METADES`
* min dif :class:`deslib.des.probabilistic.MinimumDifference`
* mla :class:`deslib.dcs.mla.MLA`
* ola :class:`deslib.dcs.ola.OLA`
* rank :class:`deslib.dcs.rank.Rank`
* rrc :class:`deslib.des.probabilistic.RRC`
* single best :class:`deslib.static.single_best.SingleBest`
* stacked :class:`deslib.static.stacked.StackedClassifier`

regression
==========

categorical multilabel
======================

categorical multiclass
======================
* aposteriori :class:`deslib.dcs.a_posteriori.APosteriori`
* apriori :class:`deslib.dcs.a_priori.APriori`
* des clustering :class:`deslib.des.des_clustering.DESClustering`
* des knn :class:`deslib.des.des_knn.DESKNN`
* deskl :class:`deslib.des.probabilistic.DESKL`
* desmi :class:`deslib.des.des_mi.DESMI`
* desp :class:`deslib.des.des_p.DESP`
* exponential :class:`deslib.des.probabilistic.Exponential`
* knop :class:`deslib.des.knop.KNOP`
* knorae :class:`deslib.des.knora_e.KNORAE`
* knrau :class:`deslib.des.knora_u.KNORAU`
* lca :class:`deslib.dcs.lca.LCA`
* logarithmic :class:`deslib.des.probabilistic.Logarithmic`
* mcb :class:`deslib.dcs.mcb.MCB`
* metades :class:`deslib.des.meta_des.METADES`
* min dif :class:`deslib.des.probabilistic.MinimumDifference`
* mla :class:`deslib.dcs.mla.MLA`
* ola :class:`deslib.dcs.ola.OLA`
* rank :class:`deslib.dcs.rank.Rank`
* rrc :class:`deslib.des.probabilistic.RRC`
* single best :class:`deslib.static.single_best.SingleBest`
* stacked :class:`deslib.static.stacked.StackedClassifier`

