.. _random_search:
 
***************
Random Search
***************

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
===================

::

    Defaults Only

'RandomSearchPlusMiddlePoint'
======================================

::

    middle_point: True

'QORandomSearch'
===================

::

    opposition_mode: 'quasi'


ORandomSearch
===================

::

    opposition_mode: 'opposite'