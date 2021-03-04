.. _test_split:

{{ header }}

***********
Test Split
***********

.. currentmodule:: BPt

It can often be useful in machine learning applications, especially neuroimaging based ones
to set aside a reserved group of subjects as test or holdout subjects. The :class:`Dataset`
has a few useful utilities for doing this (For example: :func:`Dataset.set_test_split`).
That said, it may not always be necessary, and the correct cross-validation strategy will depend
greatly on the underlying goal of the specific project.

When to use a Test Split?
~~~~~~~~~~~~~~~~~~~~~~~~~~
Deciding when to use a test split usually is not always an easy decision, but
listing some examples here may be useful for cases when it is appropriate. 

1. You are interested in performing some level of pipeline / model exploration before settling
   on a final technique. In this case, a reasonable setup would be to define a global-train-set split
   and to explore different model / pipeline performance with cross-validation on the train set first,
   for example with :func:`evaluate`:

    ::

        evaluate(pipeline=pipeline, dataset=dataset, subjects='train')

   Then, once a suitable pipeline configuration is identified, you can re-train
   the pipeline with the full training set and evaluate it on the test set, again
   with :func:`evaluate`:

    ::

        evaluate(pipeline=pipeline, dataset=dataset, cv='test')

2. You want an additional confirmation of generalizability. In this case perhaps no
   pipeline exploration is needed, but enough subjects / data points are available
   that a testing set can be reserved to act as an extra test of generalizability.
   Additional constraints could be put on the this testing set, for example, if using
   subject's data from multiple sites, then the testing set could be defined with
   :class:`Value_Subset` to be data from only an unseen site (or sites). In this case,
   one could perform both cross-validation on the training set to establish the average
   generalizability of a modelling approach, in addition to training with the full training
   set and evaluating on the test set.

   Some perks of this setup are that it may allow you to test two different types of generalization,
   e.g., in the case of data from multiple sites, you could ignore that in the internal training set
   validation and use generalizability to the test set as a way of addressing generalizability to new sites.
   Another perk is that it may ease interpretation, as the model trained on the full training set is only
   a single model, in contrast to explaining multiple models as trained and tested with cross-validation techniques
   like K-Fold.

3. Ease of computation. Given an already defined pipeline / choice of model, and a large enough
   set of subjects / data points, evaluating performance via training on the full training set and
   testing on the full testing set just once may be desirable for computational reasons. This method
   of cross-validation is often seen in deep learning applications, where the data sizes are large
   and it is extremely intensive to train models.
