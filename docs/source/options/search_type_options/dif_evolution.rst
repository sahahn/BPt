.. _dif_evolution:
 
*************************
Differential Evolution
*************************

Background: https://en.wikipedia.org/wiki/Differential_evolution

In the below descriptions the different DE choices vary on a few different parameters.

initialization 
    The algorithm/distribution used for the initialization phase.
    Either,

    - 'LHS' : Latin Hypercube Sampling
    - 'QR' : Quasi-Random
    - 'gaussian' : Normal Distribution

    (default = 'gaussian')

scale
    The scale of random component of the updates

    Either,
    
    - 'mini' : 1 / sqrt(dimension)
    - 1 : no change

    (default = 1)

crossover
    The crossover rate value / strategy used during DE.
    Either,

    - 'dimension' : crossover rate of  1 / dimension
    - 'random' : different random (uniform) crossover rate at each iteration
    - 'onepoint' : one point crossover
    - 'twopoints' : two points crossover
    
    (default = .5)

popsize
    The size of the population to use.
    Either,

    - 'standard' : max(num_workers, 30)
    - 'dimension' : max(num_workers, 30, dimension +1)
    - 'large' : max(num_workers, 30, 7 * dimension)
    
    Note: dimension refers to the dimensions of the hyper-parameters being searched over.
    'standard' by default.s

    (default = 'standard')

recommendation
    Choice of the criterion for the best point to recommend.
    Either,

    - 'optimistic' : best
    - 'noisy' : add noise to choice of best

    (default = 'optimistic')


'DE'
=====================

::

    Defaults Only


'OnePointDE'
=====================

::

    crossover: 'onepoint'

'TwoPointsDE'
=====================

::

    crossover: 'twopoint'


'LhsDE'
=====================

::

    initialization: 'LHS'

'QrDE'
=====================

::

    initialization: 'QE'
    

'MiniDE'
=====================

::

    scale: 'mini'


'MiniLhsDE'
=====================

::

    initialization: 'LHS'
    scale: 'mini'


'MiniQrDE'
=====================

::

    initialization: 'QE'
    scale: 'mini'


'NoisyDE'
=====================

::

    recommendation: 'noisy'

'AlmostRotationInvariantDE'
==========================================
::

    crossover: .9


'AlmostRotationInvariantDEAndBigPop'
==========================================

::

    crossover: .9
    popsize: 'dimension'


'RotationInvariantDE'
==========================================

::

    crossover: 1
    popsize: 'dimension'


'BPRotationInvariantDE'
==========================================

::

    crossover: 1
    popsize: 'large'
