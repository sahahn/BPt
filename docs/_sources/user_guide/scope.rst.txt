.. _scope:

{{ header }}

***********
Scope
***********

.. currentmodule:: BPt

Scope's represent a key concept within BPt, that are present when preparing data with
the :class:`Dataset` class (See functions for adding and removing scopes
to the Dataset: :func:`add_scope <Dataset.add_scope>` and
:func:`remove_scope <Dataset.remove_scope>`), and during ML.
The `scope` argument can also be
found across different :class:`ModelPipeline <BPt.ModelPipeline>` pieces
and within :class:`ProblemSpec <BPt.ProblemSpec>`. The fundamental idea is
that during loading, plotting, ML, etc... it is often desirable to specify
a subset of the total loaded columns/features. This is accomplished within BPt via the
concept of 'scope' and the 'scope' parameter.

The concept of scopes extends beyond the :class:`Dataset` class to the rest of
BPt. The fundamental idea is that it provides a utility for more easily selecting different
subsets of columns from the full dataset. This is accomplished by providing different functions
and methods with a `scope` argument, which accepts any BPt style :ref:`Scope` input, and then
operates just on that subset of columns. For example consider the example below
with the function :func:`get_cols <Dataset.get_cols>`.

::
    
    # Empty Dataset with 3 columns
    data = Dataset(columns=['1', '2', '3'])
    
    # scope of 'all' will return all columns
    cols = data.get_cols(scope='all')

    # cols == ['1', '2', '3']


In this example, we pass a fixed input str scope: 'all'. This is a special reserved scope
which will always return all columns. In addition to 'all' there are a number of other
reserved special scopes which cannot be set, and have their own fixed behavior. These are:

- 'all'
    All loaded columns

- 'float'
    All loaded columns of type 'float', i.e.,
    a continuous variable and not a categorical variable or a data file,
    see: :ref:`data_types`

- 'category'
    All loaded columns of type / scope 'category', see :ref:`data_types`.

- 'data file'
    All loaded columns of type / scope 'data file', see :ref:`data_types`.

- 'data'
    All loaded columns with role 'data', see :ref:`role`.

- 'target'
    All loaded columns with role 'target', see :ref:`role`.

- 'non input'
    All loaded columns with role 'non input', see :ref:`role`.

- 'data float'
    All loaded columns of type 'float' with role 'data'.

- 'data category'
    All loaded columns of type 'float' with role 'data'.

- 'target float'
    All loaded columns of type 'float' with role 'target'.

- 'target category'
    All loaded columns of type 'float' with role 'target'.


Those enumerated, the scope system also passing other strings, which are not one of the above,
reserved scopes. In the case that a string is passed, the following options are possible
and are checked in this order:

1. Passing the name of a column directly. In this case that column will be returned by name.
E.g., with the variable data from before:

::

    cols = data.get_cols(scope='1')

This will specify just the column '1'.

2. Passing the name of a scope. What this refers to is the ability to add
custom scopes to columns with :func:`add_scope <Dataset.add_scope>`.
This acts as a tagging system, where
you can create custom subsets. For example if we wanted the subset of '1' and '3',
we can pass scope=['1', '3'], but if we were using this same set many times, we can also
set the scopes of each of these columns to a custom scope, e.g.,

::

    data.set_scopes({'1': 'custom', '3': 'custom'})

    cols = data.get_cols(scope='custom')

In this case, cols would return us the scope 'custom'. Likewise, you may remove
scopes with :func:`remove_scope <Dataset.remove_scope>`.

3. Passing a stub. This functionality allows us to pass a common substring present
across a number of columns, and lets us select all columns with that substring. For example,
let's say we have columns 'my_col1', 'my_col2' and 'target' loaded. By passing scope='my_col'
we can select both 'my_col1' and 'my_col2, but not 'target'.

In addition to the 4 different ways scopes can be used enumerated above, we can also
compose any combination by passing a list of scopes. For example:

::

    cols = data.get_cols(scope=['1', '2'])

Returns columns '1' and '2'. We can also combine across methods. E.g.,

::

    cols = data.get_cols(scope=['1', 'category', 'custom', 'non input'])

In this example, we are requesting the union (NOT the overlap) of column '1', any 
category columns, any columns with the scope 'custom' and any 'non input' columns.

Scopes can also be associated 1:1 with their corresponding base
ModelPipeline objects (except for the ProblemSpec scope).
One useful function designed specifically for objects with Scope
is the :class:`Duplicate<BPt.Duplicate>` Input Wrapper, which
allows us to conveniently replicate pipeline objects
across a number of scopes. This functionality is especially useful with
:class:`Transformer<BPt.Transformer>` objects, (though still usable with other pipeline pieces,
though other pieces tend to work on each feature independency,
ruining some of the benefit). For example consider a case where you would like to
run a PCA transformer on different groups of variables separately,
or say you wanted to use a categorical encoder on 15 different
categorical variables. Rather then having to manually type out every combination
or write a for loop, you can use :class:`Duplicate<BPt.Duplicate>`.

See :class:`Duplicate<BPt.Duplicate>` for more information on how to use this functionality.