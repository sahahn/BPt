**************
Core Concepts
**************

ABCD ML Objects
==================

After importing the library, the first step is to create an instance of the ABCD_ML object.
Multiple instances can be defined in parallel, which may serve useful for investigating some problems, but could also incur high memory costs.
Each ABCD_ML object is an interface for loading data, performing ML, creating visualizations,
ect… all within a self contained and user defined problem scope. When defining the main object,
it is encouraged to supply an ‘exp_name’ and ‘log_dr’, which define respectively the name of the
folder in which to save experiment logs and which directory that folder should be saved.
By defining a log directory, ABCD_ML will automatically save detailed text output logs,
copies of generated figures and train and test subjects. Logging is encouraged as both a tool for aiding reproducibility
and as a potentially sharable record ensuring best practices. For example, jupyter notebooks allow chunks of python code to be run in arbitrary orders or re-run, replacing specific cells prior output. This is a useful feature which helps promote exploration and clean output, but is not ideal when a detailed record of a full experiment is desired. With ABCD_ML’s logging, the user is free to use the notebook as intended, for natural exploration, while still retaining a detailed record of all experiments run.

A few additional options are provided when defined a class object, which include: further logging parameters, optional printed verbosity and a low memory mode among other choices. An ABCD dataset specific option can also be set, which flexibility handles and merges different input versions of the ‘NDAR’ style id.  Lastly, worth mentioning is a global ‘random_state’ parameter, which can further aid reproducibility by making use of a fixed random number seed across all workflow steps. 



.. _Data Types:

Data Types
============
ABCD_ML supports the loading and distinction of the following internal data types: binary, categorical, ordinal or float.
Binary variables are those that take on one of only two unordered values e.g., True or False, Male or Female.
Categorical variables are an extension of the binary data type, and represent those that take on only one of a limited number
of unordered values (no sense of ordinal relationship between different categories), e.g., race or blood type.
Multilabel variables are similar to categorical variables,
but with the distinction that it is permissible for a variable to take on more than one value
from the set of possible values e.g., languages spoken. Right now Multilabel data can only be loaded as a co-variate and not specified as a target datatype/ problemtype.
Float represents any continuous variable,
though in practice can be used to represent any variable with a relative ordering between measurements e.g., age or neuroimaging ROI data.
Specifying the correct internal data type is an important step,
as it relates to various options in how variables are encoded as well as handled by different pieces of the ML pipeline.
An effort was made to limit the number of data types to those which are
functionally distinct within ML applications, and while the choice of how to represent a given variable will never be trivial, we hope this helps. 


.. _Loading Structure:

Loading Structure
===================

Beyond internal data types, loading of user input falls into discrete conceptual categories, with different considerations taken depending.
These categories are primarily: Data, Targets, Covars and Strat.
Data actually refers to only ‘neuroimaging style’ data, typically information derived from regions of interest.
All variables loaded as Data are assumed to be of the ‘float’ data type.
Targets represent the variable to be predicted or outcome variables, and can take on any datatype.
Covars represent variables which are typically treated as covariates, and are treated differently from Data in how they are loaded and stored.
Covars, like Targets, can be loaded as any available internal data type.
Lastly, Strat or stratification values are used to distinguish variables which are not used directly as input during modelling,
but instead are used to define conceptual groupings of subjects.
These groupings can then be used to define behavior such as custom validation or resampling strategies,
and therefore must represent a binary or categorical data type (though tools are provided to convert float to categorical).  

The underlying structure of ABCD_ML is built upon each data point as a unique subject.
Each subject can be considered a row within a dataframe,
and the various variable values associated with that subject (loaded within Data, Covars, Targets, ect…) a column.
In this way, loading can be conducted across each category separately,
but ultimately only the overlapping subjects will be considered for ML modelling. 


.. _Dataset Types:

Dataset Types
================

When loading from a file, support for a range of ‘dataset types’ is provided.
Dataset types are one of ‘basic’, ‘explorer’ or ‘custom’.
The ‘basic’ dataset type refers to loading from an official ABCD tabular release text file,
it therefore assumes the file is tab separated and automatically handles the dropping of non-relevant columns.
The ‘explorer’ dataset type is similar to ‘basic’ in that it is ABCD specific, 
but is designed to handle loading from the ‘2.0 ABCD Data Explorer’ style files.
The user can also specify the ‘custom’ dataset type, which will work with any user defined tab separated file.
For the most flexibility, all loading functions are designed
to alternatively accept user defined pandas DataFrame or base python objects (depending on the function).


.. _Pipeline Objects:

Pipeline Objects
================

Pipelike objects are... fill me in please.


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

    selected_columns = ['name3' and 'somethingelse3']

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

    selected_columns = ['name2', name3']

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

All base :class:`Model_Pipeline <ABCD_ML.Model_Pipeline>` have the input argument extra params.... fill me in.