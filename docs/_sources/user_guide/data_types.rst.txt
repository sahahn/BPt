.. _data_types:

{{ header }}

***********
Data Types
***********

.. currentmodule:: BPt

We consider loaded variables to be essentially of three types,
'float' which are continuous variables, categorical or a data file.
By default if not specified, variables are considered to be of type 'float'.

Not taking into account :ref:`data_files`, which we will discuss below, all one generally
has to worry about with respect to data types are telling the Dataset class which columns
are categorical. By default, if any columns are set to pandas type 'category', e.g., via:

::

    data['col'] = data['col'].astype('category')

Then this example column, 'col', is already set within BPt as categorical too. You
may also specify if a column is categorical or not by adding 'category' to that columns
scope via :func:`add_scope <Dataset.add_scope>`.
For example:

::

    data.add_scope('col', 'category')

In addition to explicitly setting columns as categorical, it is important to note
that a number of Dataset methods will automatically cast relevant columns to type 'category'.
These methods include :func:`auto_detect_categorical <Dataset.auto_detect_categorical>` which
will try to automatically detect categorical columns, but also functions like:
:func:`binarize <Dataset.binarize>`,
:func:`filter_categorical_by_percent <Dataset.filter_categorical_by_percent>`,
:func:`ordinalize <Dataset.ordinalize>`,
:func:`copy_as_non_input <Dataset.copy_as_non_input>` and more.