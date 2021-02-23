.. _Search Types:

The backend library for conducting hyper-parameter searches within the BPt is nevergrad, a library developed by facebook.
They implement a whole bunch of methods, and have limited documentation explaining them.
This page will try to break down the different available options.


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

opposition_mode
    symmetrizes exploration wrt the center: (e.g. https://ieeexplore.ieee.org/document/4424748)
    - "opposite" : full symmetry 
    - "quasi" : Random * symmetric
    - None

    (default = None)


'RandomSearch'
**************

::

    Defaults Only

'RandomSearchPlusMiddlePoint'
*****************************

::

    middle_point: True

'QORandomSearch'
********************

::

    opposition_mode: 'quasi'


ORandomSearch
******************

::

    opposition_mode: 'opposite'



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


'OScrHammersleySearch'
************************

::

    sampler: 'Hammersley'
    scrambled: True
    opposition_mode: 'opposite'


'QOScrHammersleySearch'
*************************

::

    sampler: 'Hammersley'
    scrambled: True
    opposition_mode: 'quasi'


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

    sampler: 'LHS'
    cauchy: True

'MetaRecentering'
*****************

::

    cauchy: False
    autorescale: True
    sampler: 'Hammersley'

'MetaTuneRecentering'
**********************


::

    cauchy: False
    autorescale: "autotune"
    sampler: 'Hammersley'
    scrambled: True

HAvgMetaRecentering
**********************

::

    cauchy: False
    autorescale: True,
    sampler: "Hammersley"
    scrambled: True
    recommendation_rule: "average_of_hull_best"

AvgMetaRecenteringNoHull
*************************

::

    cauchy: False
    autorescale: True
    sampler: "Hammersley"
    scrambled: True,
    recommendation_rule: "average_of_exp_best"



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

'DiscreteLenglerOnePlusOne'
*********************************************************

::

    mutation: 'lengler'

'AdaptiveDiscreteOnePlusOne'
*********************************************************

::

    mutation: "adaptive"

'AnisotropicAdaptiveDiscreteOnePlusOne'
*********************************************************

::

    mutation: "coordinatewise_adaptive"

'DiscreteBSOOnePlusOne'
*********************************************************

::

    mutation: "discreteBSO"

'DiscreteDoerrOnePlusOne'
*********************************************************

::

    mutation: "doerr"

'CauchyOnePlusOne'
*********************************************************

::

    mutation: "cauchy"


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

The following parameters are changed

diagonal
    To use the diagonal version of CMA (advised in large dimensions)

    - True : Use diagonal
    - False : Don't use diagonal

fcmaes
    To use fast implementation, doesn't support diagonal=True.
    produces equivalent results, preferable for high dimensions or
    if objective function evaluation is fast.

'CMA'
*****

::

    diagonal: False
    fcmaes: False

'DiagonalCMA'
*************

::

    diagonal: True
    fcmaes: False

'FCMA'
********

::

    diagonal: False
    fcmaes: True


Further variants of CMA include CMA with test based population size adaption.
It sets Population-size equal to lambda = 4 x dimension.
It further introduces the parameters:

popsize_adaption
    To use CMA with popsize adaptation

    - True : Use popsize adaptation
    - False : Don't...

covariance_memory
    Use covariance_memory

    - True : Use covariance
    - False : Don't...



'EDA'
*************

::

    popsize_adaption: False
    covariance_memory: False


'PCEDA'
*************

::

    popsize_adaption: True
    covariance_memory: False

'MPCEDA'
*************

::

    popsize_adaption: True
    covariance_memory: True

'MEDA'
*************

::

    popsize_adaption: False
    covariance_memory: True


Evolution Strategies
=====================

Experimental evolution-strategy-like algorithms. Seems to use mutations and cross-over.
The following parameters can be changed

recombination_ratio
    If 1 then will recombine all of the population, if 0 then won't use any combinations
    just mutations

    (default = 0)

popsize
    The number of individuals in the population

    (default = 40)

offsprings
    The number of offspring from every generation

    (default = None)

only_offsprings
    If true then only keep offspring, none of the original population.

    (default = False)

ranker
    Either 'simple' or 'nsga2'


'ES'
************

::

    recombination_ratio: 0
    popsize: 40
    offsprings: 60
    only_offsprings: True
    ranker: 'simple'


'RecES'
************

::

    recombination_ratio:1
    popsize: 40
    offsprings: 60
    only_offsprings: True
    ranker: 'simple'


'RecMixES'
************

::

    recombination_ratio: 1
    popsize: 40
    offsprings: 20
    only_offsprings: False
    ranker: 'simple'


'RecMutDE'
************

::

    recombination_ratio: 1
    popsize: 40
    offsprings: None
    only_offsprings: False
    ranker: 'simple'


'MixES'
************

::

    recombination_ratio: 0
    popsize: 40
    offsprings: 20
    only_offsprings: False
    ranker: 'simple'


'MutDE'
************

::

    recombination_ratio: 0
    popsize: 40
    offsprings: None
    only_offsprings: False
    ranker: 'simple'

'NSGAIIES'
************

::

    recombination_ratio: 0
    popsize: 40
    offsprings: 60
    only_offsprings: True
    ranker: "nsga2"
 

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



Algorithm Selection
=====================

Algorithm selection works by first splitting the search budget up between trying different 
search algorithms, and the 'budget_before_choosing' is up, it uses the rest of the search
budget on the strategy that did the best.

In the case that budget_before_choosing is 1, then the algorithm is a passive portfolio of
the different options, and will split the full budget between all of them.

The parameter options refers to the algorithms it tries before choosing.


'ASCMA2PDEthird'
******************

::

    options: ['CMA', 'TwoPointsDE']
    budget_before_choosing: 1/3


'ASCMADEQRthird'
*****************

::

    options: ['CMA', 'LhsDE', 'ScrHaltonSearch']
    budget_before_choosing: 1/3



'ASCMADEthird'
*****************

::

    options: ['CMA', 'LhsDE']
    budget_before_choosing: 1/3



'TripleCMA'
*****************

::

    options: ['CMA', 'CMA', 'CMA']
    budget_before_choosing: 1/3


'MultiCMA'
*****************

::

    options: ['CMA', 'CMA', 'CMA']
    budget_before_choosing: 1/10


'MultiScaleCMA'
*****************

::

    options: ['CMA', 'ParametrizedCMA(scale=1e-3)', 'ParametrizedCMA(scale=1e-6)']
    budget_before_choosing: 1/3



'Portfolio'
****************

::

    options: ['CMA', 'TwoPointsDE', 'ScrHammersleySearch']
    budget_before_choosing: 1


'ParaPortfolio'
****************

::

    options: ['CMA', 'TwoPointsDE', 'PSO', 'SQP', 'ScrHammersleySearch']
    budget_before_choosing: 1


'SQPCMA'
************

::

    options: ['CMA', n_jobs - n_jobs // 2 'SQP']
    budget_before_choosing: 1




Competence Maps
=====================

Competence Maps essentially just automatically select an algorithm based on the parameters
passed, the number of workers, the budget, ect...



'NGO'
*****************
Nevergrad optimizer by competence map., Based on One-Shot options

'NGOpt'
*****************
Nevergrad optimizer by competence map.

'CM'
*****
Competence map, simplest


'CMandAS'
**********
Competence map, with algorithm selection in one of the cases 


'CMandAS2'
***********
Competence map, with algorithm selection in one of the cases (3 CMAs).


'CMandAS3'
***********
Competence map, with algorithm selection in one of the cases (3 CMAs).


'Shiva'
*********
"Shiva" choices - "Nevergrad optimizer by competence map"





Misc.
=====================
These optimizers did not seem to naturally fall into a category. Brief descriptions are listed below.


'NaiveIsoEMNA'
***************
Estimation of Multivariate Normal Algorithm
This algorithm is quite efficient in a parallel context, i.e. when
the population size is large.


'TBPSA'
***************
Test-based population-size adaptation, for noisy problems where the best points will be an
average of the final population.


'NaiveTBPSA'
***************
Test-based population-size adaptation
Where the best point is the best point, no average across final population.



'NoisyBandit'
**************
Noisy bandit simple optimization



'PBIL'
*********
Population based incremental learning 
"Implementation of the discrete algorithm PBIL"
https://www.ri.cmu.edu/pub_files/pub1/baluja_shumeet_1994_2/baluja_shumeet_1994_2.pdf



'PSO'
********
Standard Particle Swarm Optimisation, but no randomization of the population order.



'SQP'
*******
Scipy Minimize Base
See: https://docs.scipy.org/doc/scipy-1.1.0/reference/generated/scipy.optimize.minimize.html
Note: does not support multiple jobs at once.



'SPSA'
********
The First order SPSA algorithm, See: https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation
Note: does not support multiple jobs at once.


'SplitOptimizer'
*****************
Combines optimizers, each of them working on their own variables.
By default uses CMA and RandomSearch's


'cGA'
*******
Implementation of the discrete Compact Genetic Algorithm (cGA)
https://pdfs.semanticscholar.org/4b0b/5733894ffc0b2968ddaab15d61751b87847a.pdf



'chainCMAPowell'
*****************
A chaining consists in running algorithm 1 during T1, then algorithm 2 during T2, then algorithm 3 during T3, etc.
Each algorithm is fed with what happened before it. This 'chainCMAPowell' chains first 'CMA' then the 'Powell' optimizers.
Note: does not support multiple jobs at once.



Experimental Variants
=====================

Nevergrad also comes with a number of Experimental variants, to see all of the different options run:

::

    import nevergrad as ng
    import nevergrad.optimization.experimentalvariants
    print(sorted(ng.optimizers.registry.keys()))