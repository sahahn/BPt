.. _results:

{{ header }}

*********************
Interpreting Results
**********************

.. currentmodule:: BPt

Results from every machine learning based evaluation in BPt return a special results object called :class:`EvalResults`.
This object stores by default key information related to the conducted experiment,
which allows the user to then easily access or additionally compute a range of useful
measures. Listed below are some of the available options:

- Base common machine learning metrics are provided, across regression, binary and multi-class predictions,
  for example R2, negative mean squared error, ROC AUC, balanced accuracy, and others. In the case of employing a cross validation strategy like K-fold,
  these metrics can be accessed either per fold, or averaged across multiple folds (or even the weighted average across folds of different sizes).
  See :data:`EvalResults.mean_scores`, :data:`EvalResults.scores`, :data:`EvalResults.score`.

- Raw predictions made per participant in the validation set(s) can be accessed in multiple formats,
  and can be useful in performing further analysis beyond those implemented in the
  base library (e.g., computing new metrics or feature importances.
  See :data:`EvalResults.preds`, :func:`EvalResults.get_preds_df`.

- In the case that the underlying machine learning model natively supports a measure of feature importance
  (e.g., beta weights in a linear model), then these importances can be directly accessed.
  Additionally, feature importances can be estimated regardless of underlying pipeline through a built-in permutation based feature importance method.
  When working with neuroimaging objects directly, (e.g., volumetric or surface representations of the data),
  an interface for back-projecting feature importances back into their original space is provided.
  See :func:`EvalResults.permutation_importance`, :func:`EvalResults.get_fis`


- The results of a single evaluation, regardless of cross-validation method, can be investigated further in order to
  ask questions around the statistical significance of results and/or the potential influence of confounds on results.
  One of the most powerful tools for this type of analysis is a permutation test, wherein the analysis is repeated but with the target labels shuffled.
  An important extension to this base method is the ability to restrain the shuffling of target labels according to an underlying group or nested group structure.
  See :func:`EvalResults.run_permutation_test`.

- Another available method related to probing the significance of results, is the ability to statistically compare between two or more similar
  results objects, that perhaps vary on choice of a meaningful hyper-parameter. See :func:`EvalResults.compare`.


When it comes to presenting a final set of results within a manuscript or project write up, there is no one-size fits all solution.
Insead, how one reports results will depend fundamentally on the question(s) of interest.
In practice, the typical advice is that all metrics from experiments related to questions should be reported.
Likewise, all related experimental configurations tested should also be reported, the key point being that the user
should do their best to accurately and fairly present their results. As tempting or desirable as publishing a
very accurate classifier may be, authors should take care not to overstate their findings.
This principle holds in the context of null findings as well, where it is valuable to highlight the areas where predictive models fail. 

