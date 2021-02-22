.. _Scorers:
 
*******
Scorers
*******

Different available choices for the `scorer` parameter are shown below.
`scorer` is accepted by :class:`ProblemSpec<BPt.ProblemSpec>` and :class:`ParamSearch<BPt.ParamSearch>`.
The str indicator for each `scorer` is represented bythe sub-heading (within "")
The avaliable scorers are further broken down by which can work with different problem_types.
Additionally, a link to the original models documentation is shown.

binary
======
"accuracy"
**********

  Base Func Documentation: :func:`sklearn.metrics.accuracy_score`

"roc_auc"
*********

  Base Func Documentation: :func:`sklearn.metrics.roc_auc_score`

"roc_auc_ovr"
*************

  Base Func Documentation: :func:`sklearn.metrics.roc_auc_score`

"roc_auc_ovo"
*************

  Base Func Documentation: :func:`sklearn.metrics.roc_auc_score`

"roc_auc_ovr_weighted"
**********************

  Base Func Documentation: :func:`sklearn.metrics.roc_auc_score`

"roc_auc_ovo_weighted"
**********************

  Base Func Documentation: :func:`sklearn.metrics.roc_auc_score`

"balanced_accuracy"
*******************

  Base Func Documentation: :func:`sklearn.metrics.balanced_accuracy_score`

"average_precision"
*******************

  Base Func Documentation: :func:`sklearn.metrics.average_precision_score`

"neg_log_loss"
**************

  Base Func Documentation: :func:`sklearn.metrics.log_loss`

"neg_brier_score"
*****************

  Base Func Documentation: :func:`sklearn.metrics.brier_score_loss`

"precision"
***********

  Base Func Documentation: :func:`sklearn.metrics.precision_score`

"precision_macro"
*****************

  Base Func Documentation: :func:`sklearn.metrics.precision_score`

"precision_micro"
*****************

  Base Func Documentation: :func:`sklearn.metrics.precision_score`

"precision_samples"
*******************

  Base Func Documentation: :func:`sklearn.metrics.precision_score`

"precision_weighted"
********************

  Base Func Documentation: :func:`sklearn.metrics.precision_score`

"recall"
********

  Base Func Documentation: :func:`sklearn.metrics.recall_score`

"recall_macro"
**************

  Base Func Documentation: :func:`sklearn.metrics.recall_score`

"recall_micro"
**************

  Base Func Documentation: :func:`sklearn.metrics.recall_score`

"recall_samples"
****************

  Base Func Documentation: :func:`sklearn.metrics.recall_score`

"recall_weighted"
*****************

  Base Func Documentation: :func:`sklearn.metrics.recall_score`

"f1"
****

  Base Func Documentation: :func:`sklearn.metrics.f1_score`

"f1_macro"
**********

  Base Func Documentation: :func:`sklearn.metrics.f1_score`

"f1_micro"
**********

  Base Func Documentation: :func:`sklearn.metrics.f1_score`

"f1_samples"
************

  Base Func Documentation: :func:`sklearn.metrics.f1_score`

"f1_weighted"
*************

  Base Func Documentation: :func:`sklearn.metrics.f1_score`

"jaccard"
*********

  Base Func Documentation: :func:`sklearn.metrics.jaccard_score`

"jaccard_macro"
***************

  Base Func Documentation: :func:`sklearn.metrics.jaccard_score`

"jaccard_micro"
***************

  Base Func Documentation: :func:`sklearn.metrics.jaccard_score`

"jaccard_samples"
*****************

  Base Func Documentation: :func:`sklearn.metrics.jaccard_score`

"jaccard_weighted"
******************

  Base Func Documentation: :func:`sklearn.metrics.jaccard_score`

"neg_hamming"
*************

  Base Func Documentation: :func:`sklearn.metrics.hamming_loss`

"matthews"
**********

  Base Func Documentation: :func:`sklearn.metrics.matthews_corrcoef`

"default"
*********

  Base Func Documentation: :func:`sklearn.metrics.roc_auc_score`


regression
==========
"explained_variance"
********************

  Base Func Documentation: :func:`sklearn.metrics.explained_variance_score`

"explained_variance score"
**************************

  Base Func Documentation: :func:`sklearn.metrics.explained_variance_score`

"r2"
****

  Base Func Documentation: :func:`sklearn.metrics.r2_score`

"max_error"
***********

  Base Func Documentation: :func:`sklearn.metrics.max_error`

"neg_median_absolute_error"
***************************

  Base Func Documentation: :func:`sklearn.metrics.median_absolute_error`

"median_absolute_error"
***********************

  Base Func Documentation: :func:`sklearn.metrics.median_absolute_error`

"neg_mean_absolute_error"
*************************

  Base Func Documentation: :func:`sklearn.metrics.mean_absolute_error`

"mean_absolute_error"
*********************

  Base Func Documentation: :func:`sklearn.metrics.mean_absolute_error`

"neg_mean_squared_error"
************************

  Base Func Documentation: :func:`sklearn.metrics.mean_squared_error`

"mean_squared_error"
********************

  Base Func Documentation: :func:`sklearn.metrics.mean_squared_error`

"neg_mean_squared_log_error"
****************************

  Base Func Documentation: :func:`sklearn.metrics.mean_squared_log_error`

"mean_squared_log_error"
************************

  Base Func Documentation: :func:`sklearn.metrics.mean_squared_log_error`

"neg_root_mean_squared_error"
*****************************

  Base Func Documentation: :func:`sklearn.metrics.mean_squared_error`

"root_mean_squared_error"
*************************

  Base Func Documentation: :func:`sklearn.metrics.mean_squared_error`

"neg_mean_poisson_deviance"
***************************

  Base Func Documentation: :func:`sklearn.metrics.mean_poisson_deviance`

"mean_poisson_deviance"
***********************

  Base Func Documentation: :func:`sklearn.metrics.mean_poisson_deviance`

"neg_mean_gamma_deviance"
*************************

  Base Func Documentation: :func:`sklearn.metrics.mean_gamma_deviance`

"mean_gamma_deviance"
*********************

  Base Func Documentation: :func:`sklearn.metrics.mean_gamma_deviance`

"default"
*********

  Base Func Documentation: :func:`sklearn.metrics.r2_score`


categorical
===========
"accuracy"
**********

  Base Func Documentation: :func:`sklearn.metrics.accuracy_score`

"roc_auc"
*********

  Base Func Documentation: :func:`sklearn.metrics.roc_auc_score`

"roc_auc_ovr"
*************

  Base Func Documentation: :func:`sklearn.metrics.roc_auc_score`

"roc_auc_ovo"
*************

  Base Func Documentation: :func:`sklearn.metrics.roc_auc_score`

"roc_auc_ovr_weighted"
**********************

  Base Func Documentation: :func:`sklearn.metrics.roc_auc_score`

"roc_auc_ovo_weighted"
**********************

  Base Func Documentation: :func:`sklearn.metrics.roc_auc_score`

"balanced_accuracy"
*******************

  Base Func Documentation: :func:`sklearn.metrics.balanced_accuracy_score`

"average_precision"
*******************

  Base Func Documentation: :func:`sklearn.metrics.average_precision_score`

"neg_log_loss"
**************

  Base Func Documentation: :func:`sklearn.metrics.log_loss`

"neg_brier_score"
*****************

  Base Func Documentation: :func:`sklearn.metrics.brier_score_loss`

"precision"
***********

  Base Func Documentation: :func:`sklearn.metrics.precision_score`

"precision_macro"
*****************

  Base Func Documentation: :func:`sklearn.metrics.precision_score`

"precision_micro"
*****************

  Base Func Documentation: :func:`sklearn.metrics.precision_score`

"precision_samples"
*******************

  Base Func Documentation: :func:`sklearn.metrics.precision_score`

"precision_weighted"
********************

  Base Func Documentation: :func:`sklearn.metrics.precision_score`

"recall"
********

  Base Func Documentation: :func:`sklearn.metrics.recall_score`

"recall_macro"
**************

  Base Func Documentation: :func:`sklearn.metrics.recall_score`

"recall_micro"
**************

  Base Func Documentation: :func:`sklearn.metrics.recall_score`

"recall_samples"
****************

  Base Func Documentation: :func:`sklearn.metrics.recall_score`

"recall_weighted"
*****************

  Base Func Documentation: :func:`sklearn.metrics.recall_score`

"f1"
****

  Base Func Documentation: :func:`sklearn.metrics.f1_score`

"f1_macro"
**********

  Base Func Documentation: :func:`sklearn.metrics.f1_score`

"f1_micro"
**********

  Base Func Documentation: :func:`sklearn.metrics.f1_score`

"f1_samples"
************

  Base Func Documentation: :func:`sklearn.metrics.f1_score`

"f1_weighted"
*************

  Base Func Documentation: :func:`sklearn.metrics.f1_score`

"jaccard"
*********

  Base Func Documentation: :func:`sklearn.metrics.jaccard_score`

"jaccard_macro"
***************

  Base Func Documentation: :func:`sklearn.metrics.jaccard_score`

"jaccard_micro"
***************

  Base Func Documentation: :func:`sklearn.metrics.jaccard_score`

"jaccard_samples"
*****************

  Base Func Documentation: :func:`sklearn.metrics.jaccard_score`

"jaccard_weighted"
******************

  Base Func Documentation: :func:`sklearn.metrics.jaccard_score`

"neg_hamming"
*************

  Base Func Documentation: :func:`sklearn.metrics.hamming_loss`

"matthews"
**********

  Base Func Documentation: :func:`sklearn.metrics.matthews_corrcoef`

"default"
*********

  Base Func Documentation: :func:`sklearn.metrics.roc_auc_score`


