.. _alg_selection:
 
*********************
Algorithm Selection
*********************

Algorithm selection works by first splitting the search budget up between trying different 
search algorithms, and the 'budget_before_choosing' is up, it uses the rest of the search
budget on the strategy that did the best.

In the case that budget_before_choosing is 1, then the algorithm is a passive portfolio of
the different options, and will split the full budget between all of them.

The parameter options refers to the algorithms it tries before choosing.


'ASCMA2PDEthird'
=====================

::

    options: ['CMA', 'TwoPointsDE']
    budget_before_choosing: 1/3


'ASCMADEQRthird'
=====================

::

    options: ['CMA', 'LhsDE', 'ScrHaltonSearch']
    budget_before_choosing: 1/3



'ASCMADEthird'
=====================

::

    options: ['CMA', 'LhsDE']
    budget_before_choosing: 1/3



'TripleCMA'
=====================

::

    options: ['CMA', 'CMA', 'CMA']
    budget_before_choosing: 1/3


'MultiCMA'
=====================

::

    options: ['CMA', 'CMA', 'CMA']
    budget_before_choosing: 1/10


'MultiScaleCMA'
=====================

::

    options: ['CMA', 'ParametrizedCMA(scale=1e-3)', 'ParametrizedCMA(scale=1e-6)']
    budget_before_choosing: 1/3



'Portfolio'
=====================

::

    options: ['CMA', 'TwoPointsDE', 'ScrHammersleySearch']
    budget_before_choosing: 1


'ParaPortfolio'
=====================

::

    options: ['CMA', 'TwoPointsDE', 'PSO', 'SQP', 'ScrHammersleySearch']
    budget_before_choosing: 1


'SQPCMA'
=====================

::

    options: ['CMA', n_jobs - n_jobs // 2 'SQP']
    budget_before_choosing: 1

