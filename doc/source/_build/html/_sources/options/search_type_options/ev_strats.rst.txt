.. _ev_strats:
 
********************
Evolution Strategies
********************

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
=====================

::

    recombination_ratio: 0
    popsize: 40
    offsprings: 60
    only_offsprings: True
    ranker: 'simple'


'RecES'
=====================

::

    recombination_ratio:1
    popsize: 40
    offsprings: 60
    only_offsprings: True
    ranker: 'simple'


'RecMixES'
=====================

::

    recombination_ratio: 1
    popsize: 40
    offsprings: 20
    only_offsprings: False
    ranker: 'simple'


'RecMutDE'
=====================

::

    recombination_ratio: 1
    popsize: 40
    offsprings: None
    only_offsprings: False
    ranker: 'simple'


'MixES'
=====================

::

    recombination_ratio: 0
    popsize: 40
    offsprings: 20
    only_offsprings: False
    ranker: 'simple'


'MutDE'
=====================

::

    recombination_ratio: 0
    popsize: 40
    offsprings: None
    only_offsprings: False
    ranker: 'simple'

'NSGAIIES'
=====================

::

    recombination_ratio: 0
    popsize: 40
    offsprings: 60
    only_offsprings: True
    ranker: "nsga2"
