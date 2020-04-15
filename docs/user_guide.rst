**************
Core Concepts
**************

ABCD ML Objects
==================

After importing the library, the first step is to create an instance of the ABCD_ML object. Multiple instances can be defined in parallel, which may serve useful for investigating some problems, but could also incur high memory costs. Each ABCD_ML object is an interface for loading data, performing ML, creating visualizations, ect… all within a self contained and user defined problem scope. When defining the main object, it is encouraged to supply an ‘exp_name’ and ‘log_dr’, which define respectively the name of the folder in which to save experiment logs and which directory that folder should be saved. By defining a log directory, ABCD_ML will automatically save detailed text output logs, copies of generated figures and train and test subjects. Logging is encouraged as both a tool for aiding reproducibility and as a potentially sharable record ensuring best practices. For example, jupyter notebooks allow chunks of python code to be run in arbitrary orders or re-run, replacing specific cells prior output. This is a useful feature which helps promote exploration and clean output, but is not ideal when a detailed record of a full experiment is desired. With ABCD_ML’s logging, the user is free to use the notebook as intended, for natural exploration, while still retaining a detailed record of all experiments run.

A few additional options are provided when defined a class object, which include: further logging parameters, optional printed verbosity and a low memory mode among other choices. An ABCD dataset specific option can also be set, which flexibility handles and merges different input versions of the ‘NDAR’ style id.  Lastly, worth mentioning is a global ‘random_state’ parameter, which can further aid reproducibility by making use of a fixed random number seed across all workflow steps. 



.. _Data Types:

Data Types
============
ABCD_ML supports the loading and distinction of the following internal data types: binary, categorical, multilabel, ordinal or float.
Binary variables are those that take on one of only two unordered values e.g., True or False, Male or Female.
Categorical variables are an extension of the binary data type, and represent those that take on only one of a limited number
of unordered values (no sense of ordinal relationship between different categories), e.g., race or blood type.
Multilabel variables are similar to categorical variables,
but with the distinction that it is permissible for a variable to take on more than one value
from the set of possible values e.g., languages spoken. Float represents any continuous variable,
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


.. _Scopes:

Scopes
=======

During modelling and testing phases, it is often desirable to specify a subset of the total loaded columns/features.
Within ABCD_ML the way subsets of columns can be specifed to different functions is through scopes.

The `scope` argument can be found across different pieces of the Model_Pipeline and Problem_Spec

The base preset str options that can be passed to scope are:

    - 'all'
        To apply to everything, regardless of float/cat or data/covar.
    
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

Beyond these base options, their exists a system for passing in either a list
or array-like (but not tuple, read further below for why), of specific loaded column
keys to use, wildcard stub strs for selecting which columns to use, or a combination.
We will discuss these options in more detail:

In the case that you would like to select a custom array-like of column names, you
simply pass in e.g., ('name1', 'name2', 'name3', ect...) - note not a list*, where the names correspond
to loaded column names. In this case, only those columns/features specifically passed will be used.

The way the wildcard systems works is similar to the custom array option above, but instead
of passing an array of specific column names, you can pass one or more wildcard strs where in order
for a column/feature to be included that column/feature must contain as a sub-string ALL of the passed
substrings. For example: if the loaded data had columns 'name1', 'name2', 'name3', 'somethingelse3',
you could pass '3', to select both 'name3' and 'somethingelse3'. Or you could pass ['3', 'name'] to select
just 'name3'.

You can provide a composition of different choices as an array-like list. The way this
composition works is that every entry in the passed list can be either one of the base preset
str options, a specific column name, or a substring wildcard. The returned scope can be thought of 
as the combination of these three types, for example, if you passed ['float', 'name1', 'something'],
all float columns, the name1 column and 'somethingelse3' columns would be your scope. Likewise, if you
pass multiple sub-strs, only the overlap will be taken as before. So for example input ['covars', '3', 'name'],
would select the combination of loaded covars columns, and the 'name3' column.

Scopes (for every scope besides the actual 'scope' param in Evaluate), are associated with specific ML pipeline objects.
Let's take the scaler and scaler_scope params as an example, in this case, the above inputs are all valid if one scaler is passed.
In the case that multiple scalers are passed, e.g. ['robust', 'standard'], then a scope must be provided for each one, in a simmilar
list, where the inds correspond, for example ['all', 'float'] along with the above scaler input, would set the scope to the robust scaler
to all, and the scope for the subsequent standard scaler to just float columns. Importantly any of the previously introduced compositions
could be passed to each object, for example [['float', 'name1', 'something'], ['covars', '3', 'name']], would be a valid input when two
objects are passed. In this case, ['float', 'name1', 'something'] is passed to the robust scaler, and ['covars', '3', 'name'] to the standard scaler.

One other useful function is built into the ML pipeline scopes, which allows us to replicate objects. 
Lets take transformer and transformer_scope as our example params.
In this case, if we had transformer input as 'pca', we can pass any of the valid scopes from before for transformer_scope, but we
can also pass special input as a tuple. In python tuples are like lists, but created with ()'s instead of []'s, and likewise a list can
be converted to a tuple easily:

::

    some_list = ['one', 'two', 'three']
    as_a_tuple = tuple(some_list)


What passing a tuple of scopes does is specifies that you would like the base ML pipeline object, in this case the transformer='pca',
to be replicated for every element of the tuple scope. For example for transformer_scope=('covars', ['name1', 'name2']) within the pipeline
two seperate pca's (with their own copies and values of hyper-params if passed), with be created, the first operating on just the covar columns
and the second operating on just the 'name1' and 'name2' columns. This functionality is especially useful with transformers
(though technically provided for scalers and loaders, but these pieces tend to work on each feature independenly, ruining the benefit),
as transformers will produce different output if given different columns, and e.g., in the example above it is perfectly reasonble to not
want to run one single pca on all the features, but to instead run one on just the co-variates and one on a different grouping of features.
This functionality also might be particullary useful with the different categorical encoder options within transformers. 
