.. _Transformers:
 
************
Transformers
************

Different base obj choices for the :class:`Transformer<BPt.Transformer>` are shown below
The exact str indicator, as passed to the `obj` param is represented by the sub-heading (within "")
Additionally, a link to the original models documentation as well as the implemented parameter distributions are shown.

"pca"
*****

  Base Class Documentation: :class:`sklearn.decomposition.PCA`

  Param Distributions

	0. "default" ::

		defaults only

	1. "pca var search" ::

		n_components: Scalar(init=0.75, lower=0.1, upper=0.99).set_mutation(sigma=0.14833333333333334).set_bounds(full_range_sampling=False, lower=0.1, upper=0.99)
		svd_solver: 'full'


"sparse pca"
************

  Base Class Documentation: :class:`sklearn.decomposition.SparsePCA`

  Param Distributions

	0. "default" ::

		defaults only


"mini batch sparse pca"
***********************

  Base Class Documentation: :class:`sklearn.decomposition.MiniBatchSparsePCA`

  Param Distributions

	0. "default" ::

		defaults only


"factor analysis"
*****************

  Base Class Documentation: :class:`sklearn.decomposition.FactorAnalysis`

  Param Distributions

	0. "default" ::

		defaults only


"dictionary learning"
*********************

  Base Class Documentation: :class:`sklearn.decomposition.DictionaryLearning`

  Param Distributions

	0. "default" ::

		defaults only


"mini batch dictionary learning"
********************************

  Base Class Documentation: :class:`sklearn.decomposition.MiniBatchDictionaryLearning`

  Param Distributions

	0. "default" ::

		defaults only


"fast ica"
**********

  Base Class Documentation: :class:`sklearn.decomposition.FastICA`

  Param Distributions

	0. "default" ::

		defaults only


"incremental pca"
*****************

  Base Class Documentation: :class:`sklearn.decomposition.IncrementalPCA`

  Param Distributions

	0. "default" ::

		defaults only


"kernel pca"
************

  Base Class Documentation: :class:`sklearn.decomposition.KernelPCA`

  Param Distributions

	0. "default" ::

		defaults only


"nmf"
*****

  Base Class Documentation: :class:`sklearn.decomposition.NMF`

  Param Distributions

	0. "default" ::

		defaults only


"truncated svd"
***************

  Base Class Documentation: :class:`sklearn.decomposition.TruncatedSVD`

  Param Distributions

	0. "default" ::

		defaults only


"one hot encoder"
*****************

  Base Class Documentation: :class:`sklearn.preprocessing.OneHotEncoder`

  Param Distributions

	0. "ohe" ::

		sparse: False
		handle_unknown: 'ignore'


"dummy coder"
*************

  Base Class Documentation: :class:`sklearn.preprocessing.OneHotEncoder`

  Param Distributions

	0. "dummy code" ::

		sparse: False
		drop: 'first'
		handle_unknown: 'error'



