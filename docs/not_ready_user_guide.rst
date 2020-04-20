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
