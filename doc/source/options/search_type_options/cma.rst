.. _cma:
 
***************
CMA
***************

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
=====================

::

    diagonal: False
    fcmaes: False

'DiagonalCMA'
=====================

::

    diagonal: True
    fcmaes: False

'FCMA'
=====================

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
=====================

::

    popsize_adaption: False
    covariance_memory: False


'PCEDA'
=====================

::

    popsize_adaption: True
    covariance_memory: False

'MPCEDA'
=====================

::

    popsize_adaption: True
    covariance_memory: True

'MEDA'
*************

::

    popsize_adaption: False
    covariance_memory: True