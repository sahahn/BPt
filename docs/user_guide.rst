



**********
New Users
**********

BPt is provided as a python based library and api, with workflows designed to be run in jupyter notebook-esque environments.
That said, a complementary web interface is under active development and can be downloaded and used from https://github.com/sahahn/BPt_app.
Users can choose to either use the more flexible python based library and / or make use of the web interface application. 
Trade-offs to consider namely revolve around prior user experience (i.e., those without coding or python experience may find
the web interface easier, whereas more experienced users might prefer the greater
flexibility and integration with the rest of the python data science environment that the python api offers) and personal preference.

A few general introductory resources for learning python, jupyter notebooks and machine learning are provided below:

- Introduction to python: https://jakevdp.github.io/WhirlwindTourOfPython/

- Intro to Jupyter-notebook: https://www.dataquest.io/blog/jupyter-notebook-tutorial/

- Brief intro to Machine Learning in python / jupyter environment: https://www.kaggle.com/learn/intro-to-machine-learning


**********
Why BPt?
**********

BPt seeks to integrate a number of lower level packages towards providing a higher level package and user experience.
On the surface it may appear that this utility overlaps highly with libraries like scikit-learn,
or a combination of scikit-learn with nilearn.
While this is in some cases true with respect to the base python library,
BPt seeks to provide extended functionality beyond what is found in the base scikit-learn package.
The most notable is perhaps direct integration with a web based interface, allowing most of the library
functionality to be accessed by those without prior programming experience.
That said, the python based library, independent of the GUI, seeks to offer new utility beyond
that found in scikit-learn and other packages in a number of key ways:

- Scikit-learn is fundamentally a lower level package, one which provides a number of the building blocks used in our package.
  That said, scikit-learn does not impose any sort of inherent order on the analysis workflow, and is far more flexible than our library.
  Flexibility is highly useful for a number of use cases, but still leaves room for higher level packages.
  Specifically, higher level packages like ours can help to both impose a conceptual ordering of recommended steps,
  as well as abstracting away boilerplate code in addition to providing other convenience functions.
  While it is obviously possible for users to re-create all of the functionality found in our library
  with the base libraries we employ, it would require in some cases immense effort, thus the existence of our package. 

- We offer advanced and flexible hyper-parameter optimization via tight integration with the python nevergrad package.
  This allows BPt to offer a huge range of optimization strategies beyond the Random and Grid Searches found in scikit-learn. 
  Beyond offering differing optimization methods, Nevergrad importantly supports more advanced functionality such as
  nesting hyperparameters with dictionaries or choice like objects.
  This functionality allows users to easily implement within 
  BPt almost auto-ML like powerful and expressive parameter searches.
 
- BPt allows hyper-parameter distributions to be associated directly with machine learning pipeline objects.
  This functionality we believe makes it easier to optimize hyper-parameters across 
  a number of pipeline pieces concurrently. We further allow preset hyper-parameter
  distributions to be selected or modified.
  This functionality allows less experienced users to still be able to perform hyper-parameter optimization 
  without specifying the sometimes complex parameters themselves.
  In the future we hope to support the sharing of user defined hyper-parameter 
  distributions (note: this feature is already supported in the multi-user version of the web interface).

- We introduce meta hyper-parameter objects, e.g., the :class:`Select<BPt.Select>` object, which allows specifying the choice between
  different pipeline objects as a hyper-parameter. For example, the choice between base two or more base models,
  say a Random Forest model and an Elastic Net Regression (each with their own associated distributions of hyper-parameters), 
  can be specified with the Select wrapper as itself a hyper-parameter.
  In this way, broader modelling strategies can be defined explicitly within the hyper-parameter search.
  This allows researchers the ability to easily perform properly nested model selection,
  and thus avoid common pitfalls associated with performing too many repeated internal validations.
  Along similar lines, we introduce functionality where features can themselves be set as binary hyper-parameters.
  This allows for the possibility of higher level optimizations.

- BPt introduced the concept of Scopes across all pipeline pieces.
  Scopes are a way for different pipeline pieces to optionally act on only a subset of
  features. While a similar functionality is provided via ColumnTransformers in scikit-learn,
  our package improves on this principle in a number of ways. In particular, a number of
  transformers and feature selection objects can alter in sometimes unpredictable ways the number
  of features (i.e., the number of features input before a PCA transformation, vs. the number of output features).
  Our package tracks these changes. What tracking these changes allows is for scopes to be set in downstream pipeline pieces,
  e.g., the model itself, and have that functional set of features passed to the piece still reflect the intention 
  of the original scope. For example, if one wanted to perform one hot encoding on a feature X,
  then further specify that a sub-model within an ensemble learner should only be provided feature X,
  our implemented scope system would make this possible. 
  Importantly, the number of output features from the one hot encoding does not need to be known ahead of time,
  which makes properly nesting transformations like this much more accessible.
  Scopes can be particularly useful in neuroimaging based ML where a user might have data from a number of different modalities,
  and further a combination of categorical and continuous variables, all of which they may want to dynamically treat differently. 

- We seek to more tightly couple in general the interaction between data loading, defining cross validation strategies,
  evaluating ML pipelines and making sense of results. For libraries like scikit-learn,
  this coupling is explicitly discouraged (which allows them to provide an extremely modular set of building blocks).
  On the other hand, by coupling these processes on our end, we can introduce a number of conveniences to the end-user.
  These span a number of common use cases associated with neuroimaging based ML, including: Allowing the previously introduced concept of Scope.
  Abstracting away a number of the considerations that must be made when dealing with loading, plotting and modelling across different data types (e.g., categorical vs. continuous).
  Abstracting away a number of the considerations that must be made when dealing with loading, plotting and modelling with missing data / NaN’s
  Data loading and visualization related functions, e.g., dropping outliers and automatically visually viewing the distributions of a number of input variables.
  The generation of automatic feature importances across different cross validation schemes, and their plotting.
  Exporting of loaded variable summary information to .docx format. 
  The ability to easily save and load full projects. And more!

- At the cost of some flexibility, but with the aims of reducing potentially redundant and cognitively demanding choices,
  the Model Pipeline’s constructed within BPt restrict the order in which the different pieces can be specified (e.g., Imputation must come before feature selection).
  That said, as all pipeline pieces are designed to accept custom objects, experienced users can easily get around this by passing in their own custom pipelines in relevant places.
  Whenever possible, we believe it to be a benefit to reduce researcher degrees of freedom.

- BPt provides a loader module which allows the arbitrary inclusion of non-typical 2D scikit-learn input directly integrated into the ML pipeline.
  This functionality is designed to work with the hyper-parameter optimization, scope and other features already mentioned,
  and allows the usage of common neuroimaging features such as 3D MRI volumes, timeseries, connectomes, surface based inputs,
  and more. Explicitly, this functionality is designed to be used and evaluated in a properly nested machine learning context. 
  An immensely useful package in this regard is nilearn, and by extension nibabel.
  Nilearn provides a number of functions which work very nicely with this loader module.
  This can be seen as combining the functionality of our package and nilearn, as an alternative to combining scikit-learn and nilearn.
  While the latter is obviously possible and preferable for some users, we hope that providing a higher level interface is still useful to others. 

- Along with the loader module, we currently include a number of extension objects beyond those found in the base nilearn library.
  These span the extraction of Network metrics from an adjacency matrix, support for extracting regions of interest from static or timeseries surfaces,
  the generation of randomly defined surface parcellations,and the automatic caching of user defined, potentially computationally expensive,
  loading transformations. We plan to continue to add more useful functions like these in the future.

- Additional measures of feature importance can be specified to be automatically computed. Further, by tracking how features change,
  it can be useful in certain circumstances to back project computed feature importances to their 
  original space (e.g., in the case of pipeline where surfaces from a few modalities are loaded from disk along
  with a number of phenotypic categorical variables, a parcellation applied on just the surface volumes,
  feature selection performed separately for say a number of different modalities, and then a base model evaluated, 
  feature importances from the base model can be automatically projected back to the different modalities surfaces).

- BPt integrates useful pieces from a number of other scikit-learn adjacent packages.
  These span from popular gradient boosting libraries lightgbm and xgboost, to ensemble options offered by deslib,
  feature importances as computed by the SHAP library, the Categorical Encoders library for categorical encoding options
  and more. By providing a unified interface for accessing these popular and powerful tools,
  we hope to make it easier for users to easily integrate the latest advances in machine learning.


**************
Core Concepts
**************

This section is devoted as a placeholder with more detailed information about different core components of the library.
In particular, you will often find within other sections of the documentation links to sub-sections within the sections as a way
of referring to a more detailed explanation around a concept when warranted. 

.. _Pipeline Objects:

Pipeline Objects
================

Across all base :class:`Model_Pipeline<BPt.Model_Pipeline>` pieces, e.g., :class:`Model<BPt.Model>`
or :class:`Scaler<BPt.Scaler>`, there exists an `obj` param when initizalizing these objects. This parameter
can broadly refer to either a str, which indicates a valid pre-defined custom obj for that piece, or depending
on the pieces, this parameter can be passed a custom object directly.


.. _Params:

Params
======

On the back-end, if a :class:`Param_Search<BPt.Param_Search>` object is passed when creating a
:class:`Model_Pipeline <BPt.Model_Pipeline>`, then a hyperparameter search will be conducted.
All Hyperparameter search types are implemented on the backend with facebook's
`Nevergrad <https://github.com/facebookresearch/nevergrad>`_ library.

Specific hyperparameters distributions in which to search over are set within their corresponding
base Model_Pipeline object, e.g., the params argument is :class:`Model<BPt.Model>`. For any object
with a params argument you can set an associated hyperparameter distribution, which specifies values to
search over (again assuming that param_search != None, if param_search is None, only passed params with constant
values will be applied to object of interest, and any with associated Nevergrad parameter distributions will just
be ignored).

You have two different options in terms of input that params can accept, these are:

    - Select a preset distribution
        To select a preset, BPt defined, distribution, the selected object must first
        have atleast one preset distribution. These options can be found for each object
        specifically in the documentation under where that object is defined. Specifially,
        they will be listed with both an integer index, and a corresponding str name
        (see :ref:`Models`).
        
        For example, in creating a binary :class:`Model<BPt.Model>` we could pass:
        
        ::
            
            # Option 1 - as int
            model = Model(obj = "dt classifier",
                          params = 1)

            # Option 2 - as str
            model = Model(obj = "dt classifier",
                          params = "dt classifier dist")

        In both cases, this selects the same preset distribution for the decision
        tree classifier.


    - Pass a custom nevergrad distribution
        If you would like to specify your own custom hyperparameter distribution to search over,
        you can, you just need to specify it as a python dictionary of 
        `nevergrad parameters <https://facebookresearch.github.io/nevergrad/parametrization.html>`_ 
        (follow the link to learn more about how to specify nevergrad params).
        You can also go into the source code for BPt, specifically BPt/helpers/Default_Params.py,
        to see how the preset distributions are defined, as a further example.

        Specifically the dictionary of params should follow the scikit_learn param dictionary format,
        where the each key corresponds to a parameter, but the value as a nevergrad parameter (instead of scikit_learn style).
        Further, if you need to specify nested parameters, e.g., for a custom object, you seperate parameters with '__',
        so e.g., if your custom model has a base_estimator param, you can pass:
        
        ::

            params = {'base_estimator__some_param' : nevergrad dist}

        Lastly, it is worth noting that you can pass either just static values or a combination of nevergrad distributions
        and static values, e.g.,

        ::

            {'base_estimator__some_param' : 6} 

        (Note: extra params can also be used to pass static values, and extra_params takes precedence
        if a param is passed to both params and extra_params).

The special input wrapper :class:`Select<BPt.Select>` can also be used to implicitly introduce hyperparameters
into the :class:`Model_Pipeline <BPt.Model_Pipeline>`. 


.. _Scopes:

Scopes
=======

During the modeling and testing phases, it is often desirable to specify a subset of the total loaded columns/features.
Within BPt the way subsets of columns can be specifed to different functions is through scope parameters.

The `scope` argument can be found across different :class:`Model_Pipeline <BPt.Model_Pipeline>` pieces and within Problem_Spec.

The base preset str options that can be passed to scope are:

    - 'all'
        To specify all features, everything, regardless of data type.
    
    - 'float'
        To apply to all non-categorical columns, in both
        loaded data and covars.

    - 'data'
        To apply to all loaded data columns only.

    - 'data files'
        To apply to just columns which were originally loaded as data files.

    - 'float covars' or 'fc'
        To apply to all non-categorical, float covars columns only.

    - 'cat' or 'categorical'
        To apply to just loaded categorical data.

    - 'covars'
        To apply to all loaded covar columns only.

Beyond these base options, their exists a system for passing in either an array-like or tuple
of keys to_use, wildcard stub strs for selecting which columns to use, or a combination.
We will discuss these options in more detail below:

In the case that you would like to select a custom array-like of column names, you could
simply pass: (where selected columns are the features that would be selected by that scope)

::
    
    # As tuple
    scope = ('name1', 'name2', 'name3')

    # This is the hypothetical output, not what you pass
    selected_columns = ['name1', 'name2', 'name3']


    # Or as array
    scope = np.array(['some long list of specific keys'])

    selected_columns = ['some long list of specific keys']

In this case, we are assuming the column/feature names passed correspond exactly to loaded
column/ feature names. In this case, if all items within the array-like scope are specific keys,
the columns used by that scope will be just those keys.

The way the wildcard systems works is similar to the custom array option above, but instead
of passing an array of specific column names, you can pass one or more wildcard strs where in order
for a column/feature to be included that column/feature must contain as a sub-string ALL of the passed
substrings. For example: if the loaded data had columns 'name1', 'name2', 'name3' and 'somethingelse3'.
By passing different scopes, you can see the corresponding selected columns:

::

    # Single wild card
    scope = '3'

    selected_columns = ['name3', 'somethingelse3']

    # Array-like of wild cards
    scope =  ['3', 'name']

    selected_columns = ['name3']

You can further provide a composition of different choices also as an array-like list. The way this
composition works is that every entry in the passed list can be either: one of the base preset
str options, a specific column name, or a substring wildcard.

The selected columns can then be thought of as a combination of these three types, where the output will be
the same as if took the union from any of the preset keys, specific key names and the columns selected by the wildcard.
For example, assuming we have the same loaded columns as above, and that 'name2' is the only loaded feature with datatype 'float':

::

    scope = ['float', 'name1', 'something']

    # 'float' selects 'name2', 'name1' selects 'name1', and wildcard something selects 'somethingelse3'
    # The union of these is
    selected_columns = ['name2', 'name1', 'somethingelse3']

    # Likewise, if you pass multiple wildcard sub-strs, only the overlap will be taken as before
    scope = ['float', '3', 'name']

    selected_columns = ['name2', 'name3']

Scopes more generally are associated 1:1 with their corresponding base Model_Pipeline objects (except for the Problem_Spec scope).
One useful function designed specifically for objects with Scope is the :class:`Duplicate<BPt.Duplicate>` Inute Wrapper, which
allows us to conviently replicate pipeline objects across a number of scopes. This functionality is especially useful with
:class:`Transformer<BPt.Transformer>` objects, (though still usable with other pipeline pieces, though other pieces
tend to work on each feature independenly, ruining some of the benefit). For example consider a case where you would like to
run a PCA tranformer on different groups of variables seperately, or say you wanted to use a categorical encoder on 15 different
categorical variables. Rather then having to manually type out every combination or write a for loop, you can use :class:`Duplicate<BPt.Duplicate>`.

See :class:`Duplicate<BPt.Duplicate>` for more information on how to use this funcationality.


.. _Extra Params:

Extra Params
=============

All base :class:`Model_Pipeline <BPt.Model_Pipeline>` have the input argument `extra params`. This parameter is designed
to allow passing additional values to the base objects, seperate from :ref:`Params`. Take the case where you
are using a preset model, with a preset parameter distribution, but you only want to change 1 parameter in the model while still keeping
the rest of the parameters associated with the param distribution. In this case, you could pass that value in extra params.

`extra params` are passed as a dictionary, where the keys are the names of parameters (only those accessible to the base classes init), for example
if we were selecting the 'dt' ('decision tree') :class:`Model<BPt.Model>`, and we wanted to use the first built in
preset distribution for :ref:`Params`, but then fix the number of `max_features`, we could do it is as:

::

    model = Model(obj = 'dt',
                  params = 1,
                  extra_params = {'max_features': 10}) 
                  

.. _Custom Input Objects:

Custom Input Objects
=====================

Custom input objects can be passed to the `obj` parameter for a number of base :class:`Model_Pipeline <BPt.Model_Pipeline>` pieces.

There are though, depending on which base piece is being passed, different considerations you may have to make. More information will be
provided here soon.