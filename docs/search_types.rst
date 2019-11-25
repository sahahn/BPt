.. _SearchTypes:


The backend library for conducting hyper-parameter searches within ABCD_ML is facebook's nevergrad.
They implement a whole bunch of methods, and have limited documentation explaining them.
This page will try to break down the different avaliable options.


Random Search
=============
Background: https://en.wikipedia.org/wiki/Random_search

There are few optional parameters you may specify in order to produce different random search behavior.

middle_point
    Optional enforcement of the first suggested point as zero.
    Either,

    - True : Enforced middle suggested point
    - False : Not enforced

    (default = False)

cauchy
    To use cauchy random distribution or not.
    Either,

    - True : Use the cauchy ditribution 
    - False : Use a gaussian distribution

    (default = False)

'RandomSearch'
**************

::

    Defaults Only

'RandomSearchPlusMiddlePoint'
*****************************

::

    middle_point: True

'CauchyRandomSearch'
********************

::

    cauchy: True


One Shot Optimization
=====================
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

recommendation_rule
    Method for selecting best point.
    Either,
   
    - 'average_of_best' : take average over all better then median
    - 'pessimistic' : selecting pessimistic best
    
    (default = 'pessimistic')


'HaltonSearch'
**************

::

    Defaults Only


'HaltonSearchPlusMiddlePoint'
*****************************

::

    middle_point: True


'ScrHaltonSearch'
*****************

::

    scrambled: True 


'ScrHaltonSearchPlusMiddlePoint'
********************************

::

    middle_point: True
    scrambled: True

'HammersleySearch'
******************

::

    sampler: 'Hammersley'


'HammersleySearchPlusMiddlePoint'
*********************************

::

    sampler: 'Hammersley'
    middle_point: True 

'ScrHammersleySearchPlusMiddlePoint'
************************************

::

    scrambled: True
    sampler: 'Hammersley'
    middle_point: True

'ScrHammersleySearch'
*********************

::

    sampler: 'Hammersley'
    scrambled: True


'CauchyScrHammersleySearch'
***************************

::

    cauchy: True
    sampler: 'Hammersley'
    scrambled: True

'LHSSearch'
***********

::

    sampler: 'LHS'

'CauchyLHSSearch'
*****************

::

    sampler: 'LHS', cauchy: True



One Plus One
=============
This is a family of evolutionary algorithms that use a technique called 1+1 or One Plus One.
'simple but sometimes powerful class of optimization algorithm.
We use asynchronous updates, so that the 1+1 can actually be parallel and even
performs quite well in such a context - this is naturally close to 1+lambda.'

The algorithms vary on the following parameters,

noise_handling
    How re-evaluations are performed.
    
    - 'random' : a random point is reevaluated regularly
    - 'optimistic' : the best optimistic point is reevaluated regularly
    - a coefficient can to tune the regularity of these reevaluations

    (default = (None, .05))

mutation
    The strategy for producing changes / mutations.

    - 'gaussian' : standard mutation by adding a Gaussian random variable (with progressive widening) to the best pessimistic point
    - 'cauchy' : same as Gaussian but with a Cauchy distribution.
    - 'discrete' : discrete distribution
    - 'fastga' : FastGA mutations from the current best
    - 'doublefastga' : double-FastGA mutations from the current best (Doerr et al, Fast Genetic Algorithms, 2017)
    - 'portfolio' : Random number of mutated bits (called niform mixing in Dang & Lehre 'Self-adaptation of Mutation Rates in Non-elitist Population', 2016)

    (default = 'gaussian')

crossover
    Optional additional of genetic cross over.

    - True : Add genetic crossover step every other step.
    - False : No crossover.
    
    (default = False)



'OnePlusOne'
*************************************************

::

    Defaults Only


'NoisyOnePlusOne'
*********************************************************

::
    
    noise_handling: 'random'


'OptimisticNoisyOnePlusOne'
*********************************************************

::
    
    noise_handling: 'optimistic'


'DiscreteOnePlusOne'
*********************************************************

::
    
    mutation: 'discrete'


'OptimisticDiscreteOnePlusOne'
*********************************************************

::
    
    noise_handling: 'optimistic'
    mutation: 'discrete'


'NoisyDiscreteOnePlusOne'
*********************************************************

::
    
    noise_handling: ('random', 1.0)
    mutation: 'discrete'


'DoubleFastGADiscreteOnePlusOne'
*********************************************************

::
    
    mutation: 'doublefastga'


'FastGADiscreteOnePlusOne'
*********************************************************

::
    
    mutation: 'fastga'


'DoubleFastGAOptimisticNoisyDiscreteOnePlusOne'
*********************************************************

::
    
    noise_handling: 'optimistic'
    mutation: 'doublefastga'


'FastGAOptimisticNoisyDiscreteOnePlusOne'
*********************************************************

::
    
    noise_handling: 'optimistic'
    mutation: 'fastga'


'FastGANoisyDiscreteOnePlusOne'
*********************************************************

::
    
    noise_handling: 'random'
    mutation: 'fastga'


'PortfolioDiscreteOnePlusOne'
*********************************************************

::
    
    mutation: 'portfolio'


'PortfolioOptimisticNoisyDiscreteOnePlusOne'
*********************************************************

::
    
    noise_handling: 'optimistic'
    mutation: 'portfolio'


'PortfolioNoisyDiscreteOnePlusOne'
*********************************************************

::
    
    noise_handling: 'random'
    mutation: 'portfolio'


'CauchyOnePlusOne'
*********************************************************

::
    
    mutation: 'cauchy'


'RecombiningOptimisticNoisyDiscreteOnePlusOne'
*********************************************************

::
    
    crossover: True
    mutation: 'discrete'
    noise_handling: 'optimistic'


'RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne'
*********************************************************

::
    
    crossover: True
    mutation: 'portfolio'
    noise_handling: 'optimistic'


CMA
===
This refers to the covariance matrix adaptation evolutionary optimzation strategy
Background: https://en.wikipedia.org/wiki/CMA-ES

The following parameter is changed

diagonal
    To use the diagonal version of CMA (advised in large dimensions)

    - True : Use diagonal
    - False : Don't use diagonal

'CMA'
*****

::

    Defaults Only

'DiagonalCMA'
*************

::

    diagonal: True
 


Differential Evolution
======================

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
    
    Note: dimension refers to the dimensions of the hyperparameters being searched over.
    'standard' by default.s

    (default = 'standard')

recommendation
    Choice of the criterion for the best point to recommend.
    Either,

    - 'optimistic' : best
    - 'noisy' : add noise to choice of best

    (default = 'optimistic')

'DE'
****

::

    Defaults Only


'OnePointDE'
************

::

    crossover: 'onepoint'

'TwoPointsDE'
*************

::

    crossover: 'twopoint'


'LhsDE'
*******

::

    initialization: 'LHS'

'QrDE'
******

::

    initialization: 'QE'
    

'MiniDE'
********

::

    scale: 'mini'


'MiniLhsDE'
***********

::

    initialization: 'LHS'
    scale: 'mini'


'MiniQrDE'
***********

::

    initialization: 'QE'
    scale: 'mini'


'NoisyDE'
**********

::

    recommendation: 'noisy'

'AlmostRotationInvariantDE'
***************************

::

    crossover: .9


'AlmostRotationInvariantDEAndBigPop'
************************************

::

    crossover: .9
    popsize: 'dimension'


'RotationInvariantDE'
*********************

::

    crossover: 1
    popsize: 'dimension'


'BPRotationInvariantDE'
***********************

::

    crossover: 1
    popsize: 'large'


Scipy Optimizers
================
Various optimizers as introduced in scipy.
See: https://docs.scipy.org/doc/scipy/reference/optimize.html

Params vary on,

method
    The scipy implemented method to use

    - 'Nelder-Mead' : https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html
    - 'Powell' : https://docs.scipy.org/doc/scipy/reference/optimize.minimize-powell.html
    - 'COBYLA' : https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cobyla.html
    - 'SLSQP' : https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html

random_restart
    Whether to restart at a random point if the optimizer converged but the budget is not entirely
    spent yet (otherwise, restarts from best point).

    - True : True
    - False : False

'NelderMead'
************

::

    method: 'Nelder-Mead'

'Powell'
********

::

    method: 'Powell'


'RPowell'
*********

method='Powell'
random_restart=True

'Cobyla'
*********

::

    method: 'COBYLA'

'RCobyla'
**********

::

    method: 'COBYLA'
    random_restart: True

'SQP'
******

::

    method: 'SLSQP'

'RSQP'
*******

::

    method: 'SLSQP'
    random_restart: True

