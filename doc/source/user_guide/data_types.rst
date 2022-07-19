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

Then this example column, 'col', is already set within BPt as categorical too. 
For example:

.. warning::
    
    In the case where the categorical column is composed of say strings or objects,
    it is not enough to just cast it as type category for use as input to a machine learning pipeline!
    In order for it to be valid input, the strings or objects or whatever must be converted to float / int
    representations, e.g., the best way to do this is ussually :func:`ordinalize <Dataset.ordinalize>`. 

You may also specify if a column is categorical or not by adding 'category' to that columns
scope via :func:`add_scope <Dataset.add_scope>`. Again though, this should only be done for columns which
are already ordinally or one-hot encoded.

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


Basic Example
--------------

    import BPt as bp
    data = bp.Dataset([['cow', 1, 3],
                       ['horse', 2, 2],
                       ['cat', 3, 2],],
                      columns=['f1', 'f2', 'f3'])
    data

Define a basic dataset with 3 columns, we will assume they are all categorical, and use :func:`ordinalize <Dataset.ordinalize>` on all of them.

.. ipython:: python

    data = data.ordinalize(scope='data')
    data


We can confirm they were cast to categorical:

.. ipython:: python

    data.get_cols(scope='category')

Using functions like :func:`ordinalize <Dataset.ordinalize>`, should be the prefered way of letting Dataset's know which variables are categorical.