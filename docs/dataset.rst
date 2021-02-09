.. currentmodule:: BPt

**********************
Overview
**********************

In order to get data ready for machine learning, BPt has a specially designed :ref:`Dataset` class.
This class is new as of BPt version >= 2 (replacing the building in loading functions of the
old BPt_ML). This class is built on top of the DataFrame from the pandas library. As we will see,
the reccomended way of preparing data actually first involves using the DataFrame class from pandas
directly. The is that pandas and the DataFrame class should be used to load all of the data you might
end up wanting to use. Luckily pandas contains a huge wealth of useful functions
for accomplishing this already. Next, once all of the data is loaded, we cast the DataFrame
to the BPt :ref:`Dataset` class, and then use the built in :ref:`Dataset` methods to get the data
ready for use with the rest of BPt. This includes steps like specifying which variables are in what
role (e.g., target variables vs. data variables), outlier detection, transformations like binning and binarizing,
tools for plotting / viewing distributions and specifying a global train / test split. We will introduce all of
this functionality below!

*************
Loading Data
*************

Data of interest is invetibly going to come from a wide range of different sources, luckily the python library pandas
has an incredible amount of support for loading data from different sources into DataFrames.
Likewise, pandas offers a hueg amount of support material, e.g., https://pandas.pydata.org/docs/getting_started/intro_tutorials/02_read_write.html
for getting starting with loading in raw data (or a google search with a specific question will almsot always help).
Pandas should be used to accomplish the initial loading and merging of all tabular data of interest into a DataFrame.

For example, let's say our data of interest is stored in a file called data.csv, we could load it with:

::

    data = pd.read_csv('data.csv')

Next let's say we wanted to specify that the subject column is called 'subject', we can do this with another
call to the native pandas API.

::

    data = data.set_index('subject')

Then when we are finished with loading and merging the data of interest into a DataFrame, we can cast it to a BPt Dataset!

::

    from BPt import Dataset
    data = Dataset(data)

We can now still use a number of the native pandas api methods in addition now to the added functionality of the BPt :ref:`Dataset`!

**********************
Key Dataset Concepts
**********************

.. _role:

Role
======

There are three roles in the :ref:`Dataset` class / BPt. These are
'data', 'target' and 'non input'.


.. _scope:

Scope
======

See concept :ref:`Scope`. In particular, with respect to the
:ref:`Dataset` class is the option for passing the name of a scope.
Functions: :func:`add_scope <Dataset.add_scope>` and
:func:`remove_scope <Dataset.remove_scope>`


Though, this class also uses scope quite frequently as an argument, the same way as
described in the reference for :ref:`Pipeline Objects`
 
.. _data_types:

Data Types
======================

We consider loaded variables to be essentially of three types,
'float' which are continuous variables, categorical or a data file.
By default if not specified, variables are considered to be of type 'float'.

Not taking into account data files, which we will discuss below, all one generally
has to worry about with respect to data types are telling the Dataset class which columns
are categorical. By default, if any columns are set to pandas type 'category', e.g., via:

::

    data['col'] = data['col'].astype('category')

Then this example column, 'col', is already set within BPt as categorical too. You
may also specify if a column is categorical or not by adding 'category' to that columns
scope via :func:`add_scope <Dataset.add_scope>`.
For example:

::

    data.add_scope(col='col', scope='category')

In addition to explicitly setting columns as categorical, it is important to note
that a number of Dataset methods will automatically cast relevant columns to type 'category'.
These methods include :func:`auto_detect_categorical <Dataset.auto_detect_categorical>` which
will try to automatically detect categorical columns, but also functions like:
:func:`binarize <Dataset.binarize>`,
:func:`filter_categorical_by_percent <Dataset.filter_categorical_by_percent>`,
:func:`ordinalize <Dataset.ordinalize>`,
:func:`copy_as_non_input <Dataset.copy_as_non_input>` and more.


Data files allow BPt to work with any type of arbitrary data beyond simply tabular data.



Warnings
==========

Column names within the :ref:`Dataset` class must be strings in order for the concept
of scopes to work more easily. Therefore if any columns are loaded as a non-string, they
will be renamed to the string version of that non-string.

Their are some caveats to using some DataFrame function once the DataFrame has
been cast as a :ref:`Dataset`. While a great deal will continue to work, their are
certain types of operations which can end up either re-casting the result back to
a DataFrame (therefore losing all of the Dataset's metadata), or renaming columns,
which may cause internal errors and metadata loss.


.. _Dataset:

********
Dataset
********
The different methods for the Dataset class are listed below, though note most methods from
the pandas DataFrame will also work.


********************************
Dataset - Base Methods
********************************

get_cols
==========
.. automethod:: Dataset.get_cols


get_subjects
=============
.. automethod:: Dataset.get_subjects


get_values
=============
.. automethod:: Dataset.get_values


add_scope
=============
.. automethod:: Dataset.add_scope


remove_scope
=============
.. automethod:: Dataset.remove_scope


set_role
=============
.. automethod:: Dataset.set_role


set_roles
=============
.. automethod:: Dataset.set_roles


get_roles
=============
.. automethod:: Dataset.get_roles


add_data_files
===============================
.. automethod:: Dataset.add_data_files


get_file_mapping
===============================
.. automethod:: Dataset.get_file_mapping

copy
=============
.. automethod:: Dataset.copy


auto_detect_categorical
===============================
.. automethod:: Dataset.auto_detect_categorical


get_Xy
===============================
.. automethod:: Dataset.get_Xy


get_train_Xy
===============================
.. automethod:: Dataset.get_train_Xy


get_test_Xy
===============================
.. automethod:: Dataset.get_test_Xy


********************************
Dataset - Encoding Methods
********************************

to_binary
==========
.. automethod:: Dataset.to_binary


binarize
==================================
.. automethod:: Dataset.binarize


k_bin
=============
.. automethod:: Dataset.k_bin


ordinalize
==================================
.. automethod:: Dataset.ordinalize


nan_to_class
=============
.. automethod:: Dataset.nan_to_class


copy_as_non_input
==================================
.. automethod:: Dataset.copy_as_non_input


add_unique_overlap
===============================
.. automethod:: Dataset.add_unique_overlap


***********************************
Dataset - Filtering & Drop Methods
***********************************


filter_outliers_by_std
=======================
.. automethod:: Dataset.filter_outliers_by_std


filter_outliers_by_percent
=====================================
.. automethod:: Dataset.filter_outliers_by_percent


filter_categorical_by_percent
==================================
.. automethod:: Dataset.filter_categorical_by_percent


drop_cols
=====================
.. automethod:: Dataset.drop_cols


drop_nan_subjects
=====================
.. automethod:: Dataset.drop_nan_subjects


drop_subjects_by_nan
=====================
.. automethod:: Dataset.drop_subjects_by_nan


drop_cols_by_unique_val
=====================
.. automethod:: Dataset.drop_cols_by_unique_val


drop_cols_by_nan
=================
.. automethod:: Dataset.drop_cols_by_nan


drop_id_cols
=============
.. automethod:: Dataset.drop_id_cols


drop_duplicate_cols
=====================
.. automethod:: Dataset.drop_duplicate_cols


apply_inclusions
=====================
.. automethod:: Dataset.apply_inclusions


apply_exclusions
=====================
.. automethod:: Dataset.apply_exclusions



*************************************
Dataset - Plotting / Viewing Methods
*************************************

plot
=====================
.. automethod:: Dataset.plot


plot_vars
=====================
.. automethod:: Dataset.plot_vars


show
=====================
.. automethod:: Dataset.show


show_nan_info
=====================
.. automethod:: Dataset.show_nan_info


info
=====================
.. automethod:: Dataset.info


print_nan_info
=====================
.. automethod:: Dataset.print_nan_info


********************************
Dataset - Test Split Methods
********************************

set_test_split
=====================
.. automethod:: Dataset.set_test_split


set_train_split
=====================
.. automethod:: Dataset.set_train_split


save_test_subjects
=====================
.. automethod:: Dataset.save_test_subjects


save_train_subjects
=====================
.. automethod:: Dataset.save_train_subjects

