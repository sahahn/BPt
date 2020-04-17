.. _Feat Importances:

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


"sklearn perm"
**************
This is also permutation feature importance, but it has a few differences from the base 'perm'.
This will notably call sklearns implementation of permutation feature importance.
See: https://scikit-learn.org/dev/modules/generated/sklearn.inspection.permutation_importance.html#sklearn.inspection.permutation_importance

What nice about their implementation is that it can respect random_state.
Their implementation is likely better tested then mine as well, but for whatever reason mine
runs much quicker than theirs, and with ~100 permutations they seem to give roughly the same
results. It is therefore up to user which implementation they would like to use.

"sklearn perm train"
**************
See above "sklearn perm", this option simply changes the split to
computing permutation values on both the training and testing/validation set.
The parameters are
the same.

"sklearn perm all"
**************
See above "sklearn perm", this option simply changes the split to
computing permutation values on both the training and testing/validation set.
The parameters are
the same.