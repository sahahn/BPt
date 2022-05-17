.. _validation:

{{ header }}

***********
Validation
***********

.. currentmodule:: BPt


Background
~~~~~~~~~~~~~~~~~~~~~~~~~~

In order for the results from a ML based predictive experiment to be valid, some sort of cross or external validation is essential.
So how do we decide between say a training-test split between two matched samples
and K-fold cross validation on the whole sample? In short, it depends. There is no silver-bullet that works
for every scenario, but the good news is that for the most part it really shouldn't matter!
The most important element to properly using an external validation strategy isn't between 3 folds versus 10 folds,
but instead is in how the chosen strategy is used. That is to say, the validation data should only be used in
answering the main predictive question of interest. If instead the current experiment isn't related to the primary research question the result will not be reported,
then the validation data should not be used in any way. Let's consider an explicit example of what not to do: 

Let's say we decide to use a 3-fold cross validation strategy, predicting age from cortical thickness,
and we start by evaluating a simple linear regression model, but it doesn't do very well.
Next, we try a random forest model, which does a little better, but still not great, so we try changing a
few of its parameters, run the 3-fold cross validation again, change a few more parameters and after a little
tweaking eventually get a score we are satisfied with. We then report just this result: “a random forest model predicted age, R2=XXX”. 

The issue with the example above is namely one of over-using the validation data.
By repeatedly testing different models with the same set of validation data,
be it through K-fold or a left-aside testing set, we have increased our chances of
obtaining an artificially high performance metric through chance alone
(i.e., this is a phenomenon pretty similar in nature to p-hacking in classical statistics).
Now in this example the fix is fairly easy. If we want to perform model selection and model hyper-parameter tuning,
we can, but as long as both the model selection and hyper-parameter tuning are conducted with nested validation
(e.g., on a set-aside training dataset). Fundamentally, it depends on what our ultimate question of interest is. For example,
if we are explicitly interested in the difference in performance between different ML models, then it is reasonable to
evaluate all of the different models of interest on the validation data, as long as all of their respective performances are reported.


When to use a Test Split?
~~~~~~~~~~~~~~~~~~~~~~~~~~

It can often be useful in machine learning applications, especially neuroimaging based ones, or those where
the underlying compution is very expense, to employ a holdout set of subject, or train-test split as a validation strategy of choice. The :class:`Dataset`
has a few useful utilities for doing this (For example: :func:`Dataset.set_test_split`).
That said, it may not always be necessary, and the correct cross-validation strategy will depend
greatly on the underlying goal of the specific project.

Deciding when to use a test split (versus say a 5-fold CV on the full dataset) is not always an easy decision, but
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

   Note: This same process could be performed with nested K-Fold cross-validation, but
   it tends to be harder to conceptualize / implement correctly.

2. You want an additional confirmation of generalizability. In this case perhaps no
   pipeline exploration is needed, but enough subjects / data points are available
   that a testing set can be reserved to act as an extra test of generalizability.
   Additional constraints could be put on the this testing set, for example, if using
   subject's data from multiple sites, then the testing set could be defined with
   :class:`ValueSubset` to be data from only an unseen site (or sites). In this case,
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

Other Considerations
~~~~~~~~~~~~~~~~~~~~~~

There are of course other potential pitfalls in selecting and employing validation strategies that may
vary depending on the underlying complexity of the problem of interest. For example, if using multi-site data, there
is a difference between a model generalizing to other participants from the same site (random split k-fold validation)
versus generalizing to new participants from unseen sites (group k-fold validation where site is preserved within fold).
While choice of optimal strategy will vary, BPt provides an easy interface for employing varied and potentially complex validation strategies,
though the :class:`CV` object. 