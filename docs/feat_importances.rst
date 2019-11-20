*******************
Feat Importances
*******************

Determining Feature Importance is an important step in making sense of ML output.
The following e-book provides an extensive background on different feature importance methods,
in particular this section: https://christophm.github.io/interpretable-ml-book/agnostic.html


Different feat importances can be broadly considered as either global (one value per feature) or 
local (features can ussually be averaged over to produce a global measure of feature importance). This param 
is the 'scopes'.
A feature importance can further be calculated on different parts of the data, or rather the split can differ.
Splits can be either, train only, test only or both.
Train only refers to calculating feature importances based on only the trained model, or with access to only how that model
performs on the same data it was trained. For example: if computing permutation feature importance, train only would
calculate this score on the training set, and therefore describe the behavior of the trained model,
but not neccisarily how well those feature generalize. In this case if the model is overfit, then the train only
feature importances would most likely not be meaningful. Test only is determined based on performance or predictions 
made on the test or validation set only. This method, using the example from before, would indicate how
the trained model genralizes to new unseen data.

scopes
    The scopes in which the feature importance method calculates over.

    - 'local' : One feature importance per data point
    - 'global' : One feature importance per feature

split
    What portion of the data is used to determine feature importance

    - 'train' : On only the training data
    - 'test' : On only the testing or validation data
    - 'all' : On both the training and the testing data


"base"
**************
Base feature importance refers to the feature importance as calculated automatically
by the underlying model. For example, for linear models, base feature importance
reflects the beta weights of the trained model. For tree based models, e.g., random forest,
the feature importances represent the calculated feature importances (gini split index in this case).

Base feature importances have scope == 'global' and split == 'train'.


  Param Distributions

	0. "default" ::

		defaults only


"shap"
**************
This computes Shap feature importance, which can be read about in more detail at: https://github.com/slundberg/shap
A good more general description is also found at: https://christophm.github.io/interpretable-ml-book/shap.html

This base "shap" setting uses split == 'test', so importances are determined from the
test or validation set.

Shap generates both local and global feature importances.

The underlying shap method with change if a linear underlying model, tree-based or anything else.
Linear and tree-based models run quickly, but take care in computing shap values
for anything else! In this case, the kernel shap computer is used, which approximates the shap
values and is hugely compute intensive!! See: https://shap.readthedocs.io/en/latest/ for more info
on the TreeExplainer and KernelExplainer.

The parameters that can be changed in shap are as follows (descriptions from the shap documentation):

shap__global__avg_abs
    This parameter is considered regardless of the underlying model. If
    set to True, then when computing global feature importance from the
    initially computed local shap feature importance the average of the absolute value 
    will be taken! When set to False, the average value will be taken. One might want to
    set this to True if only concerned with the magnitude of feature importance, rather
    than the sign.
    
    (default = False)

shap__linear__feature_dependence
    Only used with linear base models.
    There are two ways we might want to compute SHAP values, either the full conditional SHAP
    values or the independent SHAP values. For independent SHAP values we break any
    dependence structure between features in the model and so uncover how the model would behave if we
    intervened and changed some of the inputs. For the full conditional SHAP values we respect
    the correlations among the input features, so if the model depends on one input but that
    input is correlated with another input, then both get some credit for the model's behavior. The
    independent option stays "true to the model" meaning it will only give credit to features that are
    actually used by the model, while the correlation option stays "true to the data" in the sense that
    it only considers how the model would behave when respecting the correlations in the input data.

    (default = 'independent')

shap__linear__nsamples
    Only used with linear base models.
    Number of samples to use when estimating the transformation matrix used to account for
    feature correlations. Only used if shap__linear__feature_dependence is set to 'correlation'.

    (default = 1000)


shap__tree__feature_dependence
    Only used with tree base models.
    Since SHAP values rely on conditional expectations we need to decide how to handle correlated
    (or otherwise dependent) input features. The default "tree_path_dependent" approach is to just
    follow the trees and use the number of training examples that went down each leaf to represent
    the background distribution. This approach repects feature dependecies along paths in the trees.
    However, for non-linear marginal transforms (like explaining the model loss)  we don't yet
    have fast algorithms that respect the tree path dependence, so instead we offer an "independent"
    approach that breaks the dependencies between features, but allows us to explain non-linear
    transforms of the model's output. Note that the "independent" option requires a background
    dataset and its runtime scales linearly with the size of the background dataset you use. Anywhere
    from 100 to 1000 random background samples are good sizes to use.

    (default = 'tree_path_dependent')

shap__tree__model_output
    Only used with tree base models.
    What output of the model should be explained. If "margin" then we explain the raw output of the
    trees, which varies by model (for binary classification in XGBoost this is the log odds ratio).
    If "probability" then we explain the output of the model transformed into probability space
    (note that this means the SHAP values now sum to the probability output of the model). If "log_loss"
    then we explain the log base e of the model loss function, so that the SHAP values sum up to the
    log loss of the model for each sample. This is helpful for breaking down model performance by feature.
    Currently the probability and log_loss options are only supported when feature_dependence="independent".

    (default = 'margin')

shap__tree__tree_limit
    Only used with tree base models.
    Limit the number of trees used by the model. By default None means no use the limit of the
    original model, and -1 means no limit.

    (default = None)

shap__kernel__nkmean
    Used when the underlying model is not linear or tree based.
    This setting offers a speed up to the kernel estimator by replacing
    the background dataset with a kmeans representation of the data.
    Set this option to None in order to use the full dataset directly,
    otherwise the int passed will the determine 'k' in the kmeans algorithm.
    
    (default = 10)


shap__kernel__nsamples
    Used when the underlying model is not linear or tree based.
    Number of times to re-evaluate the model when explaining each prediction.
    More samples lead to lower variance estimates of the SHAP values.
    The 'auto' setting uses nsamples = 2 * X.shape[1] + 2048.
    
    (default = 'auto')


shap__kernel__l1_reg
    Used when the underlying model is not linear or tree based.
    The l1 regularization to use for feature selection (the estimation procedure is based on
    a debiased lasso). The auto option currently uses "aic" when less that 20% of the possible sample
    space is enumerated, otherwise it uses no regularization. THE BEHAVIOR OF "auto" WILL CHANGE
    in a future version to be based on num_features instead of AIC.
    The "aic" and "bic" options use the AIC and BIC rules for regularization.
    Using "num_features(int)" selects a fix number of top features. Passing a float directly sets the
    "alpha" parameter of the sklearn.linear_model.Lasso model used for feature selection.

    (default = 'aic')



  Param Distributions

	0. "base shap" ::

		shap__global__avg_abs: False
		shap__linear__feature_dependence: independent
		shap__linear__nsamples: 1000
		shap__tree__feature_dependence: tree_path_dependent
		shap__tree__model_output: margin
		shap__tree__tree_limit: None
		shap__kernel__nkmean: 10
		shap__kernel__nsamples: auto
		shap__kernel__l1_reg: aic



"shap train"
**************
See above "shap", this option simply changes the split to
computing shap values on the training set. The parameters are
the same.


"shap all"
**************
See above "shap", this option simply changes the split to
computing shap values on both the training and testing/validation set.
The parameters are
the same.


"perm"
**************
This refers to computing feature importance through a permutation and predict strategy,
For more info see: https://christophm.github.io/interpretable-ml-book/feature-importance.html

Note the following article may be of interest before deciding to use permuation feature importance:
https://arxiv.org/pdf/1905.03151.pdf

This base "perm" setting using split == 'test', so importances are determined from the
test or validation set.

The 'perm__n_perm' parameter determines the number of time each feature column is permuted.


  Param Distributions

	0. "base perm" ::

		perm__n_perm: 10


"perm train"
**************
See above "perm", this option simply changes the split to
computing permutation values on both the training and testing/validation set.
The parameters are
the same.

"perm all"
**************
See above "perm", this option simply changes the split to
computing permutation values on both the training and testing/validation set.
The parameters are
the same.
