.. _one_shot:
 
**********************
One Shot Optimization
**********************

Implemented one-hot optimization methods which are 'hopefully better than random search by ensuring more uniformity'.
The algorithms vary on the following parameters,


sampler
    Type of random sampling. Either,

    - 'Halton' : A low quality sampling method when the dimension is high
    - 'Hammersley' : Hammersley sampling
    - 'LHS' : Latin Hypercube Sampling

    (default = 'Halton')

scrambled
    Adds scrambling to the search

    - True : scrambling is added
    - False : scrambling is not added

    (default = False)

middle_point
    Optional enforcement of the first suggested point as zero.
    Either,

    - True : Enforced middle suggested point
    - False : Not enforced

    (default = False)

cauchy
    Use Cauchy inverse distribution instead of Gaussian when fitting points to real space
    Either,

    - True : Use the cauchy ditribution 
    - False : Use a gaussian distribution

    (default = False)

rescaled
    Rescale the sampling pattern to reach the boundaries.
    Either,

    - True : rescale
    - False : don't rescale

    (default = False)

autorescale
    Perform auto-rescaling

    - True : Auto rescale
    - False : don't auto rescale

    (default = False)


recommendation_rule
    Method for selecting best point.
    Either,
   
    - 'average_of_best' : take average over all better then median
    - 'pessimistic' : selecting pessimistic best
    
    (default = 'pessimistic')

opposition_mode
    symmetrizes exploration wrt the center: (e.g. https://ieeexplore.ieee.org/document/4424748)
    - "opposite" : full symmetry 
    - "quasi" : Random * symmetric
    - None

    (default = None)


'HaltonSearch'
==============================

::

    Defaults Only


'HaltonSearchPlusMiddlePoint'
==============================

::

    middle_point: True


'ScrHaltonSearch'
==============================

::

    scrambled: True 


'ScrHaltonSearchPlusMiddlePoint'
============================================================

::

    middle_point: True
    scrambled: True

'HammersleySearch'
==============================

::

    sampler: 'Hammersley'


'HammersleySearchPlusMiddlePoint'
============================================================

::

    sampler: 'Hammersley'
    middle_point: True 

'ScrHammersleySearchPlusMiddlePoint'
============================================================

::

    scrambled: True
    sampler: 'Hammersley'
    middle_point: True

'ScrHammersleySearch'
==============================

::

    sampler: 'Hammersley'
    scrambled: True


'OScrHammersleySearch'
==============================

::

    sampler: 'Hammersley'
    scrambled: True
    opposition_mode: 'opposite'


'QOScrHammersleySearch'
==============================

::

    sampler: 'Hammersley'
    scrambled: True
    opposition_mode: 'quasi'


'CauchyScrHammersleySearch'
==============================

::

    cauchy: True
    sampler: 'Hammersley'
    scrambled: True

'LHSSearch'
==============================

::

    sampler: 'LHS'

'CauchyLHSSearch'
==============================

::

    sampler: 'LHS'
    cauchy: True

'MetaRecentering'
==============================

::

    cauchy: False
    autorescale: True
    sampler: 'Hammersley'

'MetaTuneRecentering'
==============================


::

    cauchy: False
    autorescale: "autotune"
    sampler: 'Hammersley'
    scrambled: True

HAvgMetaRecentering
==============================

::

    cauchy: False
    autorescale: True,
    sampler: "Hammersley"
    scrambled: True
    recommendation_rule: "average_of_hull_best"

AvgMetaRecenteringNoHull
==============================

::

    cauchy: False
    autorescale: True
    sampler: "Hammersley"
    scrambled: True,
    recommendation_rule: "average_of_exp_best"
