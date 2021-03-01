.. _misc:
 
***************
Misc.
***************

These optimizers did not seem to naturally fall into a category. Brief descriptions are listed below.


'NaiveIsoEMNA'
===================
Estimation of Multivariate Normal Algorithm
This algorithm is quite efficient in a parallel context, i.e. when
the population size is large.


'TBPSA'
===================
Test-based population-size adaptation, for noisy problems where the best points will be an
average of the final population.


'NaiveTBPSA'
===================
Test-based population-size adaptation
Where the best point is the best point, no average across final population.


'NoisyBandit'
===================
Noisy bandit simple optimization


'PBIL'
===================
Population based incremental learning 
"Implementation of the discrete algorithm PBIL"
https://www.ri.cmu.edu/pub_files/pub1/baluja_shumeet_1994_2/baluja_shumeet_1994_2.pdf



'PSO'
===================
Standard Particle Swarm Optimization, but no randomization of the population order.



'SQP'
===================
Scipy Minimize Base
See: https://docs.scipy.org/doc/scipy-1.1.0/reference/generated/scipy.optimize.minimize.html
Note: does not support multiple jobs at once.


'SPSA'
===================
The First order SPSA algorithm, See: https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation
Note: does not support multiple jobs at once.


'SplitOptimizer'
===================
Combines optimizers, each of them working on their own variables.
By default uses CMA and RandomSearch's


'cGA'
===================
Implementation of the discrete Compact Genetic Algorithm (cGA)
https://pdfs.semanticscholar.org/4b0b/5733894ffc0b2968ddaab15d61751b87847a.pdf



'chainCMAPowell'
===================
A chaining consists in running algorithm 1 during T1, then algorithm 2 during T2, then algorithm 3 during T3, etc.
Each algorithm is fed with what happened before it. This 'chainCMAPowell' chains first 'CMA' then the 'Powell' optimizers.
Note: does not support multiple jobs at once.


Experimental Variants
======================

Nevergrad also comes with a number of Experimental variants, to see all of the different options run:

::

    import nevergrad as ng
    import nevergrad.optimization.experimentalvariants
    print(sorted(ng.optimizers.registry.keys()))

Note: you do not have to run this import to select any of these options, this is just
to see the different options.