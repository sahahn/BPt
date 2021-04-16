.. _Feat Selectors:
 
**************
Feat Selectors
**************

Different base obj choices for the :class:`FeatSelector<BPt.FeatSelector>` are shown below
The exact str indicator, as passed to the `obj` param is represented by the sub-heading (within "")
The avaliable feat selectors are further broken down by which can work with different problem_types.
Additionally, a link to the original models documentation as well as the implemented parameter distributions are shown.

binary
======
"rfe"
*****

  Base Class Documentation: :class:`sklearn.feature_selection.RFE`

  Param Distributions

	0. "base rfe" ::

		n_features_to_select: None

	1. "rfe num feats dist" ::

		n_features_to_select: Scalar(init=0.5, lower=0.1, upper=0.99).set_mutation(sigma=0.14833333333333334).set_bounds(full_range_sampling=False, lower=0.1, upper=0.99)


"selector"
**********

  Base Class Documentation: :class:`BPt.extensions.FeatSelectors.FeatureSelector`

  Param Distributions

	0. "random" ::

		mask: 'sets as random features'

	1. "searchable" ::

		mask: 'sets as hyperparameters'


"univariate selection c"
************************

  Base Class Documentation: :class:`sklearn.feature_selection.SelectPercentile`

  Param Distributions

	0. "base univar fs classifier" ::

		score_func: <function f_classif at 0x7f0e1d36e3a0>
		percentile: 50

	1. "univar fs classifier dist" ::

		score_func: <function f_classif at 0x7f0e1d36e3a0>
		percentile: Scalar(init=50, lower=1, upper=99).set_mutation(sigma=16.333333333333332).set_bounds(full_range_sampling=False, lower=1, upper=99)

	2. "univar fs c keep more" ::

		score_func: <function f_classif at 0x7f0e1d36e3a0>
		percentile: Scalar(init=75, lower=50, upper=99).set_mutation(sigma=8.166666666666666).set_bounds(full_range_sampling=False, lower=50, upper=99)

	3. "univar fs c keep less" ::

		score_func: <function f_classif at 0x7f0e1d36e3a0>
		percentile: Scalar(init=25, lower=1, upper=50).set_mutation(sigma=8.166666666666666).set_bounds(full_range_sampling=False, lower=1, upper=50)


"variance threshold"
********************

  Base Class Documentation: :class:`sklearn.feature_selection.VarianceThreshold`

  Param Distributions

	0. "default" ::

		defaults only



regression
==========
"rfe"
*****

  Base Class Documentation: :class:`sklearn.feature_selection.RFE`

  Param Distributions

	0. "base rfe" ::

		n_features_to_select: None

	1. "rfe num feats dist" ::

		n_features_to_select: Scalar(init=0.5, lower=0.1, upper=0.99).set_mutation(sigma=0.14833333333333334).set_bounds(full_range_sampling=False, lower=0.1, upper=0.99)


"selector"
**********

  Base Class Documentation: :class:`BPt.extensions.FeatSelectors.FeatureSelector`

  Param Distributions

	0. "random" ::

		mask: 'sets as random features'

	1. "searchable" ::

		mask: 'sets as hyperparameters'


"univariate selection r"
************************

  Base Class Documentation: :class:`sklearn.feature_selection.SelectPercentile`

  Param Distributions

	0. "base univar fs regression" ::

		score_func: <function f_regression at 0x7f0e1d36e670>
		percentile: 50

	1. "univar fs regression dist" ::

		score_func: <function f_regression at 0x7f0e1d36e670>
		percentile: Scalar(init=50, lower=1, upper=99).set_mutation(sigma=16.333333333333332).set_bounds(full_range_sampling=False, lower=1, upper=99)

	2. "univar fs r keep more" ::

		score_func: <function f_regression at 0x7f0e1d36e670>
		percentile: Scalar(init=75, lower=50, upper=99).set_mutation(sigma=8.166666666666666).set_bounds(full_range_sampling=False, lower=50, upper=99)

	3. "univar fs r keep less" ::

		score_func: <function f_regression at 0x7f0e1d36e670>
		percentile: Scalar(init=25, lower=1, upper=50).set_mutation(sigma=8.166666666666666).set_bounds(full_range_sampling=False, lower=1, upper=50)


"variance threshold"
********************

  Base Class Documentation: :class:`sklearn.feature_selection.VarianceThreshold`

  Param Distributions

	0. "default" ::

		defaults only



categorical
===========
"rfe"
*****

  Base Class Documentation: :class:`sklearn.feature_selection.RFE`

  Param Distributions

	0. "base rfe" ::

		n_features_to_select: None

	1. "rfe num feats dist" ::

		n_features_to_select: Scalar(init=0.5, lower=0.1, upper=0.99).set_mutation(sigma=0.14833333333333334).set_bounds(full_range_sampling=False, lower=0.1, upper=0.99)


"selector"
**********

  Base Class Documentation: :class:`BPt.extensions.FeatSelectors.FeatureSelector`

  Param Distributions

	0. "random" ::

		mask: 'sets as random features'

	1. "searchable" ::

		mask: 'sets as hyperparameters'


"univariate selection c"
************************

  Base Class Documentation: :class:`sklearn.feature_selection.SelectPercentile`

  Param Distributions

	0. "base univar fs classifier" ::

		score_func: <function f_classif at 0x7f0e1d36e3a0>
		percentile: 50

	1. "univar fs classifier dist" ::

		score_func: <function f_classif at 0x7f0e1d36e3a0>
		percentile: Scalar(init=50, lower=1, upper=99).set_mutation(sigma=16.333333333333332).set_bounds(full_range_sampling=False, lower=1, upper=99)

	2. "univar fs c keep more" ::

		score_func: <function f_classif at 0x7f0e1d36e3a0>
		percentile: Scalar(init=75, lower=50, upper=99).set_mutation(sigma=8.166666666666666).set_bounds(full_range_sampling=False, lower=50, upper=99)

	3. "univar fs c keep less" ::

		score_func: <function f_classif at 0x7f0e1d36e3a0>
		percentile: Scalar(init=25, lower=1, upper=50).set_mutation(sigma=8.166666666666666).set_bounds(full_range_sampling=False, lower=1, upper=50)


"variance threshold"
********************

  Base Class Documentation: :class:`sklearn.feature_selection.VarianceThreshold`

  Param Distributions

	0. "default" ::

		defaults only



