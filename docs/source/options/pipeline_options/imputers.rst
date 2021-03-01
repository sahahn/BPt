.. _Imputers:
 
********
Imputers
********

Different base obj choices for the :class:`Imputer<BPt.Imputer>` are shown below
The exact str indicator, as passed to the `obj` param is represented by the sub-heading (within "")
Additionally, a link to the original models documentation as well as the implemented parameter distributions are shown.
Note that if the iterative imputer is requested, base_model must also be passed.

"mean"
******

  Base Class Documentation: :class:`sklearn.impute.SimpleImputer`

  Param Distributions

	0. "mean imp" ::

		strategy: 'mean'


"median"
********

  Base Class Documentation: :class:`sklearn.impute.SimpleImputer`

  Param Distributions

	0. "median imp" ::

		strategy: 'median'


"most frequent"
***************

  Base Class Documentation: :class:`sklearn.impute.SimpleImputer`

  Param Distributions

	0. "most freq imp" ::

		strategy: 'most_frequent'


"constant"
**********

  Base Class Documentation: :class:`sklearn.impute.SimpleImputer`

  Param Distributions

	0. "constant imp" ::

		strategy: 'constant'


"iterative"
***********

  Base Class Documentation: :class:`sklearn.impute.IterativeImputer`

  Param Distributions

	0. "iterative imp" ::

		initial_strategy: 'mean'
		skip_complete: True



