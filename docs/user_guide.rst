**************
Core Concepts
**************

.. _Pipeline Objects:

Pipeline Objects
================

Across all base :class:`Model_Pipeline<ABCD_ML.Model_Pipeline>` pieces, e.g., :class:`Model<ABCD_ML.Model>`
or :class:`Scaler<ABCD_ML.Scaler>`, there exists an `obj` param when initizalizing these objects. This parameter
can broadly refer to either a str, which indicates a valid pre-defined custom obj for that piece, or depending
on the pieces, this parameter can be passed a custom object directly.


.. _Params:

Params
======

On the back-end, if a :class:`Param_Search<ABCD_ML.Param_Search>` object is passed when creating a
:class:`Model_Pipeline <ABCD_ML.Model_Pipeline>`, then a hyperparameter search will be conducted.
All Hyperparameter search types are implemented on the backend with facebook's
`Nevergrad <https://github.com/facebookresearch/nevergrad>`_ library.

Specific hyperparameters distributions in which to search over are set within their corresponding
base Model_Pipeline object, e.g., the params argument is :class:`Model<ABCD_ML.Model>`. For any object
with a params argument you can set an associated hyperparameter distribution, which specifies values to
search over (again assuming that param_search != None, if param_search is None, only passed params with constant
values will be applied to object of interest, and any with associated Nevergrad parameter distributions will just
be ignored).

You have two different options in terms of input that params can accept, these are:

    - Select a preset distribution
        To select a preset, ABCD_ML defined, distribution, the selected object must first
        have atleast one preset distribution. These options can be found for each object
        specifically in the documentation under where that object is defined. Specifially,
        they will be listed with both an integer index, and a corresponding str name
        (see :ref:`Models`).
        
        For example, in creating a binary :class:`Model<ABCD_ML.Model>` we could pass:
        
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
        You can also go into the source code for ABCD_ML, specifically ABCD_ML/helpers/Default_Params.py,
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

The special input wrapper :class:`Select<ABCD_ML.Select>` can also be used to implicitly introduce hyperparameters
into the :class:`Model_Pipeline <ABCD_ML.Model_Pipeline>`. 


.. _Scopes:

Scopes
=======

During the modeling and testing phases, it is often desirable to specify a subset of the total loaded columns/features.
Within ABCD_ML the way subsets of columns can be specifed to different functions is through scope parameters.

The `scope` argument can be found across different :class:`Model_Pipeline <ABCD_ML.Model_Pipeline>` pieces and within Problem_Spec.

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
One useful function designed specifically for objects with Scope is the :class:`Duplicate<ABCD_ML.Duplicate>` Inute Wrapper, which
allows us to conviently replicate pipeline objects across a number of scopes. This functionality is especially useful with
:class:`Transformer<ABCD_ML.Transformer>` objects, (though still usable with other pipeline pieces, though other pieces
tend to work on each feature independenly, ruining some of the benefit). For example consider a case where you would like to
run a PCA tranformer on different groups of variables seperately, or say you wanted to use a categorical encoder on 15 different
categorical variables. Rather then having to manually type out every combination or write a for loop, you can use :class:`Duplicate<ABCD_ML.Duplicate>`.

See :class:`Duplicate<ABCD_ML.Duplicate>` for more information on how to use this funcationality.


.. _Extra Params:

Extra Params
=============

All base :class:`Model_Pipeline <ABCD_ML.Model_Pipeline>` have the input argument `extra params`. This parameter is designed
to allow passing additional values to the base objects, seperate from :ref:`Params`. Take the case where you
are using a preset model, with a preset parameter distribution, but you only want to change 1 parameter in the model while still keeping
the rest of the parameters associated with the param distribution. In this case, you could pass that value in extra params.

`extra params` are passed as a dictionary, where the keys are the names of parameters (only those accessible to the base classes init), for example
if we were selecting the 'dt' ('decision tree') :class:`Model<ABCD_ML.Model>`, and we wanted to use the first built in
preset distribution for :ref:`Params`, but then fix the number of `max_features`, we could do it is as:

::

    model = Model(obj = 'dt',
                  params = 1,
                  extra_params = {'max_features': 10}) 
                  

.. _Custom Input Objects:

Custom Input Objects
=====================

Custom input objects can be passed to the `obj` parameter for a number of base :class:`Model_Pipeline <ABCD_ML.Model_Pipeline>` pieces.

There are though, depending on which base piece is being passed, different considerations you may have to make. More information will be
provided here soon.