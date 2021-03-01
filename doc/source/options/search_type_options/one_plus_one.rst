.. _one_plus_one:

{{ header }}
 
**********************
One Plus One
**********************

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
==============================

::

    Defaults Only


'NoisyOnePlusOne'
==============================

::
    
    noise_handling: 'random'


'OptimisticNoisyOnePlusOne'
==============================

::
    
    noise_handling: 'optimistic'


'DiscreteOnePlusOne'
==============================

::
    
    mutation: 'discrete'

'DiscreteLenglerOnePlusOne'
==============================

::

    mutation: 'lengler'

'AdaptiveDiscreteOnePlusOne'
==============================

::

    mutation: "adaptive"

'AnisotropicAdaptiveDiscreteOnePlusOne'
============================================================

::

    mutation: "coordinatewise_adaptive"

'DiscreteBSOOnePlusOne'
==============================

::

    mutation: "discreteBSO"

'DiscreteDoerrOnePlusOne'
==============================

::

    mutation: "doerr"

'CauchyOnePlusOne'
==============================

::

    mutation: "cauchy"


'OptimisticDiscreteOnePlusOne'
============================================================

::
    
    noise_handling: 'optimistic'
    mutation: 'discrete'


'NoisyDiscreteOnePlusOne'
==============================

::
    
    noise_handling: ('random', 1.0)
    mutation: 'discrete'


'DoubleFastGADiscreteOnePlusOne'
============================================================

::
    
    mutation: 'doublefastga'


'FastGADiscreteOnePlusOne'
==============================

::
    
    mutation: 'fastga'


'RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne'
============================================================

::
    
    crossover: True
    mutation: 'portfolio'
    noise_handling: 'optimistic'