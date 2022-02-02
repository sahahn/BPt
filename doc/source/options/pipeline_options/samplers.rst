.. _Samplers:
 
********
Samplers
********

Different base obj choices for the :class:`Scaler<BPt.Sampler>` are shown below
The exact str indicator, as passed to the `obj` param is represented by the sub-heading (within "")
Additionally, a link to the original models documentation as well as the implemented parameter distributions are shown.

"standard"
**********

  Base Class Documentation: :class:`sklearn.preprocessing.StandardScaler`

  Param Distributions

	0. "base standard" ::

		with_mean: True
		with_std: True


"minmax"
********

  Base Class Documentation: :class:`sklearn.preprocessing.MinMaxScaler`

  Param Distributions

	0. "base minmax" ::

		feature_range: (0, 1)


"maxabs"
********

  Base Class Documentation: :class:`sklearn.preprocessing.MaxAbsScaler`

  Param Distributions

	0. "default" ::

		defaults only


"robust"
********

  Base Class Documentation: :class:`sklearn.preprocessing.RobustScaler`

  Param Distributions

	0. "base robust" ::

		quantile_range: (5, 95)

	1. "robust gs" ::

		quantile_range: TransitionChoice([(1, 99), (2, 98), (3, 97), (4, 96), (5, 95), (6, 94), (7, 93), (8, 92), (9, 91), (10, 90), (11, 89), (12, 88), (13, 87), (14, 86), (15, 85), (16, 84), (17, 83), (18, 82), (19, 81), (20, 80), (21, 79), (22, 78), (23, 77), (24, 76), (25, 75), (26, 74), (27, 73), (28, 72), (29, 71), (30, 70), (31, 69), (32, 68), (33, 67), (34, 66), (35, 65), (36, 64), (37, 63), (38, 62), (39, 61)])


"yeo"
*****

  Base Class Documentation: :class:`sklearn.preprocessing.PowerTransformer`

  Param Distributions

	0. "base yeo" ::

		method: 'yeo-johnson'
		standardize: True


"boxcox"
********

  Base Class Documentation: :class:`sklearn.preprocessing.PowerTransformer`

  Param Distributions

	0. "base boxcox" ::

		method: 'box-cox'
		standardize: True


"winsorize"
***********

  Base Class Documentation: :class:`BPt.extensions.Scalers.Winsorizer`

  Param Distributions

	0. "base winsorize" ::

		quantile_range: (1, 99)

	1. "winsorize gs" ::

		quantile_range: TransitionChoice([(1, 99), (2, 98), (3, 97), (4, 96), (5, 95), (6, 94), (7, 93), (8, 92), (9, 91), (10, 90), (11, 89), (12, 88), (13, 87), (14, 86), (15, 85), (16, 84), (17, 83), (18, 82), (19, 81), (20, 80), (21, 79), (22, 78), (23, 77), (24, 76), (25, 75), (26, 74), (27, 73), (28, 72), (29, 71), (30, 70), (31, 69), (32, 68), (33, 67), (34, 66), (35, 65), (36, 64), (37, 63), (38, 62), (39, 61)])


"quantile norm"
***************

  Base Class Documentation: :class:`sklearn.preprocessing.QuantileTransformer`

  Param Distributions

	0. "base quant norm" ::

		output_distribution: 'normal'


"quantile uniform"
******************

  Base Class Documentation: :class:`sklearn.preprocessing.QuantileTransformer`

  Param Distributions

	0. "base quant uniform" ::

		output_distribution: 'uniform'


"normalize"
***********

  Base Class Documentation: :class:`sklearn.preprocessing.Normalizer`

  Param Distributions

	0. "default" ::

		defaults only



