.. _Ensemble Types:
 
**************
Ensemble Types
**************

Different base obj choices for the :class:`Ensemble<BPt.Ensemble>` are shown below
The exact str indicator, as passed to the `obj` param is represented by the sub-heading (within "")
The avaliable ensembles are further broken down by which can workwith different problem_types.
Additionally, a link to the original models documentation as well as the implemented parameter distributions are shown.
Also note that ensemble may require a few extra params!

binary
======
"adaboost classifier"
*********************

  Base Class Documentation: :class:`sklearn.ensemble.AdaBoostClassifier`

  Param Distributions

	0. "default" ::

		defaults only


"bagging classifier"
********************

  Base Class Documentation: :class:`sklearn.ensemble.BaggingClassifier`

  Param Distributions

	0. "default" ::

		defaults only


"balanced bagging classifier"
*****************************

  Base Class Documentation: :class:`imblearn.ensemble.BalancedBaggingClassifier`

  Param Distributions

	0. "default" ::

		defaults only


"stacking classifier"
*********************

  Base Class Documentation: :class:`BPt.pipeline.ensemble_wrappers.BPtStackingClassifier`

  Param Distributions

	0. "default" ::

		defaults only


"voting classifier"
*******************

  Base Class Documentation: :class:`BPt.pipeline.ensemble_wrappers.BPtVotingClassifier`

  Param Distributions

	0. "voting classifier" ::

		voting: 'soft'



regression
==========
"adaboost regressor"
********************

  Base Class Documentation: :class:`sklearn.ensemble.AdaBoostRegressor`

  Param Distributions

	0. "default" ::

		defaults only


"bagging regressor"
*******************

  Base Class Documentation: :class:`sklearn.ensemble.BaggingRegressor`

  Param Distributions

	0. "default" ::

		defaults only


"stacking regressor"
********************

  Base Class Documentation: :class:`BPt.pipeline.ensemble_wrappers.BPtStackingRegressor`

  Param Distributions

	0. "default" ::

		defaults only


"voting regressor"
******************

  Base Class Documentation: :class:`BPt.pipeline.ensemble_wrappers.BPtVotingRegressor`

  Param Distributions

	0. "default" ::

		defaults only



categorical
===========
"adaboost classifier"
*********************

  Base Class Documentation: :class:`sklearn.ensemble.AdaBoostClassifier`

  Param Distributions

	0. "default" ::

		defaults only


"bagging classifier"
********************

  Base Class Documentation: :class:`sklearn.ensemble.BaggingClassifier`

  Param Distributions

	0. "default" ::

		defaults only


"balanced bagging classifier"
*****************************

  Base Class Documentation: :class:`imblearn.ensemble.BalancedBaggingClassifier`

  Param Distributions

	0. "default" ::

		defaults only


"stacking classifier"
*********************

  Base Class Documentation: :class:`BPt.pipeline.ensemble_wrappers.BPtStackingClassifier`

  Param Distributions

	0. "default" ::

		defaults only


"voting classifier"
*******************

  Base Class Documentation: :class:`BPt.pipeline.ensemble_wrappers.BPtVotingClassifier`

  Param Distributions

	0. "voting classifier" ::

		voting: 'soft'



