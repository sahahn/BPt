.. _ml_pipeline:

{{ header }}

*****************
The ML Pipeline
*****************

.. currentmodule:: BPt

A machine learning pipeline is importanly not just the choice of ML model, it is the full set of transformations to the data prior to
input to an ML algorithm. This is, in a lot of ways, the area with the most researcher degrees of freedom,
as we can think of both the presence or absence of a transformation, as well as the choice of model and that model's parameters as
all 'hyper-parameters' of the broader ML pipeline.
These could be choices like what parcellation to use, to z-score each feature or not, which type of
fMRI connectivity metric to use, the type of ML estimator, the parameters associated with that estimator, etc.
The number of permutations grows quite rapidly, so in practice how should the researcher decide? We recommend treating each
possible 'hyper-parameter' according to the following set of options:

If this parameter is important to the research question, test and report the results by each possible value or a
reasonable set of values of interest that this parameter might take. For example, let's say we want to know how our
prediction varies by choice of paracellation, so we repeat our full ML experiment with 3 different parcellations, and report the results of each. 

Otherwise, if not directly important or related to the question of interest the researcher can either 1.
Fix the value ahead of time based on a priori knowledge or guess, or 2. Assign the value through some nested validation strategy
(e.g., train-validation/test split or nested K-fold).
In general, option 1 is preferable, as it is simpler to both implement and conceptualize fixing a value ahead of time.
That said, setting values through nested validation can be useful in certain cases, for example it is often used for
setting hyper-parameters specific to an ML estimator. In other words, option 2 is used as a way to try and
improve down-stream performance, with an emphasis on “try”, as it is difficult in practice to correctly
identify the choices which will benefit from this approach.

While designing an ML Pipeline can be daunting and introduce lots of researcher degrees of freedom,
it is also the area most amenable to creativity. As long as proper validation is kept in mind, testing
and trying new / different pipelines can be an important piece of ML modeling.
This becomes especially important when the researcher starts to consider ML modeling in the context of potential confounds,
where potential corrections for confounds are themselves steps within the pipeline. That said, especially as a newer researcher,
it may be a good idea to start by replicating previous strategies from the literature that have been
found to work well. Default pipelines can be easily specified within BPt. 
