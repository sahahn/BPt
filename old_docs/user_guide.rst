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

- Brief intro to Machine Learning in python / jupyter environment: https://www.kaggle.com/learn/intro-to-machine-learning


**********
Why BPt?
**********




**************
Core Concepts
**************

This section is devoted as a placeholder with more detailed information about different core components of the library.
In particular, you will often find within other sections of the documentation links to sub-sections within the sections as a way
of referring to a more detailed explanation around a concept when warranted. 

.. _Scope:

Scope
=======

Scope's represent a key concept within BPt, that are present when preparing data with
the :ref:`Dataset` class, and during ML. The `scope` argument can also be
found across different :class:`ModelPipeline <BPt.ModelPipeline>` pieces
and within :class:`ProblemSpec <BPt.ProblemSpec>`. The fundamental idea is
that during loading, plotting, ML, etc... it is often desirable to specify
a subset of the total loaded columns/features. This is accomplished within BPt via the
concept of 'scope' and the 'scope' parameter.

The concept of scopes extends beyond the :ref:`Dataset` class to the rest of
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

In addition to the 4 different ways scopes can be used enemurated above, we can also
compose any combination by passing a list of scopes. For example:

::

    cols = data.get_cols(scope=['1', '2'])

Returns columns '1' and '2'. We can also combine across methods. E.g.,

::

    cols = data.get_cols(scope=['1', 'category', 'custom', 'non input'])

In this example, we are requesting the union (NOT the overlap) of column '1', any 
category columns, any columns with the scope 'custom' and any 'non input' columns.

Scopes can also be associated 1:1 with their corresponding base ModelPipeline objects (except for the ProblemSpec scope).
One useful function designed specifically for objects with Scope is the :class:`Duplicate<BPt.Duplicate>` Input Wrapper, which
allows us to conviently replicate pipeline objects across a number of scopes. This functionality is especially useful with
:class:`Transformer<BPt.Transformer>` objects, (though still usable with other pipeline pieces, though other pieces
tend to work on each feature independenly, ruining some of the benefit). For example consider a case where you would like to
run a PCA tranformer on different groups of variables seperately, or say you wanted to use a categorical encoder on 15 different
categorical variables. Rather then having to manually type out every combination or write a for loop, you can use :class:`Duplicate<BPt.Duplicate>`.

See :class:`Duplicate<BPt.Duplicate>` for more information on how to use this funcationality.


.. _Subjects:

Subjects
=========

Various functions within BPt, and :ref:`Dataset` can accept subjects or some variation on this
name as an argument. The parameter can accept a few different values. These are explained below:

1. You may pass any array-like (e.g., list, set, pandas Index, etc...) of subjects directly.
Warning: Passing a python tuple is reserved for a special MultiIndex case!

For example:

::

  subjects = ['subj1', 'subj2', 'subj3']

Would select those three subjects.

2. You may pass the location of a text file were subject's names are stored as one subject's
name per line. Names should be saved with python style types, e.g., quotes around str's, but
if they are not, it should in most cases still be able to figure out the correct type.
For example if subjects.txt contained:

::

  'subj1'
  'subj2'
  'subj3'

We could pass:

::

  subjects = 'subjects.txt'

To select those three subjects.

3. A reserved key word may be passed. These include:

- 'all'
  Operate on all subjects

- 'nan'
  Select any subjects with any missing values in any of their loaded columns,
  regardless of scope or role.

- 'train'
  Select the set of train subjects as defined by a split in the Dataset, e.g.,
  :func:`set_train_split <BPt.Dataset.set_train_split>`.

- 'test'
  Select the set of test subjects as defined by a split in the Dataset, e.g.,
  :func:`set_test_split <BPt.Dataset.set_test_split>`.

4. You can pass the special input wrapper :class:`Value_Subset`. This can be
used to select subsets of subject by a column's value or values. See :class:`Value_Subset` for more
information on how this input class is used.

There also exists the case where you may wish for the underlying index of subjects to be a MultiIndex.
In this case, there is some extra functionality to discuss. Say for example we have a Dataset multiindex'ed
by subject and eventname, e.g.,

::

  data.set_index(['subject', 'eventname'], inplace=True)

We now have more options for how we might want to index this dataset, and therefore more options
for valid arguments to pass to a subjects argument. Consider first all of the examples from above,
where we are just specifying a subject-like index. In this case, all of those arguments will still work,
and will just return all subjects with all of their eventnames. E.g., assuming there were two eventname values
for each subjects 'e1' and 'e2':

::

  subjects = ['subj1', 'subj2']

Would select subject eventname pairs:
('subj1', 'e1'), ('subj1', 'e2'), ('subj2', 'e1'), ('subj2', 'e2')
and likewise with loading from a text file which just specified 'subj1' and 'subj2'.

Note that if we pass arguments in this manner, BPt will assume they refer to whatever
index is first, in this case 'subject', and not 'eventname'. If we wish to also select
explicitly by eventname, we have two options.

1. Pass fully index'ed tuples in an array-like manner, the same as 1. from before, e.g.:

::

  subjects = ('subj1', 'e1'), ('subj2', 'e2')

To just keep 'subj1' at event 'e1' and 'subj2' at 'e2'. Likewise, we
could select this same subset if subjects.txt was formatted as:

::

  ('subj1', 'e1')
  ('subj2', 'e2')

2. Our second option is to use the special tuple reserved input. In this case,
we must pass a python tuple with the same length at the number of levels in the
underlying MultiIndex, e.g., in the example before, of length two. Each index in the
tuple will then be used to specify the BPt subjects compatible argument for just that
level of the index. For example:

::

  subjects = ('all', ['e1'])

Would select all subjects, and then note the array-like list in the second index of the tuple,
would filter that to include only subject eventname pairs with an eventname of 'e1'. Consider another example:

::

  subjects = ('subjects.txt', 'events.txt')

In this case, the subjects to select would be loaded from 'subjects.txt' and the corresponding eventnames from
'events.txt'.


.. _Pipeline Objects:

Pipeline Objects
================

Across all base :class:`ModelPipeline<BPt.ModelPipeline>` pieces, e.g., :class:`Model<BPt.Model>`
or :class:`Scaler<BPt.Scaler>`, there exists an `obj` param when initizalizing these objects. This parameter
can broadly refer to either a str, which indicates a valid pre-defined custom obj for that piece, or depending
on the pieces, this parameter can be passed a custom object directly.


.. _Params:

Params
======

On the back-end, if a :class:`ParamSearch<BPt.ParamSearch>` object is passed when creating a
:class:`ModelPipeline <BPt.ModelPipeline>`, then a hyperparameter search will be conducted.
All Hyperparameter search types are implemented on the backend with facebook's
`Nevergrad <https://github.com/facebookresearch/nevergrad>`_ library.

Specific hyper-parameters distributions in which to search over are set within their corresponding
base ModelPipeline object, e.g., the params argument is :class:`Model<BPt.Model>`. For any object
with a params argument you can set an associated hyperparameter distribution, which specifies values to
search over (again assuming that param_search != None, if param_search is None, only passed params with constant
values will be applied to object of interest, and any with associated Nevergrad parameter distributions will just
be ignored).

You have two different options in terms of input that params can accept, these are:

    - Select a preset distribution
        To select a preset, BPt defined, distribution, the selected object must first
        have at least one preset distribution. These options can be found for each object
        specifically in the documentation under where that object is defined. Specifically,
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
        Further, if you need to specify nested parameters, e.g., for a custom object, you separate parameters with '__',
        so e.g., if your custom model has a base_estimator param, you can pass:
        
        ::

            params = {'base_estimator__some_param' : nevergrad dist}

        Lastly, it is worth noting that you can pass either just static values or a combination of nevergrad distributions
        and static values, e.g.,

        ::

            {'base_estimator__some_param' : 6} 

        (Note: extra params can also be used to pass static values, and extra_params takes precedence
        if a param is passed to both params and extra_params).

The special input wrapper :class:`Select<BPt.Select>`
can also be used to implicitly introduce hyper-parameters
into the :class:`ModelPipeline <BPt.ModelPipeline>`. 


.. _Extra Params:

Extra Params
=============

All base :class:`ModelPipeline <BPt.ModelPipeline>` have the kwargs style input argument `extra params`. This parameter is designed
to allow passing additional values to the base objects, seperate from :ref:`Params`. Take the case where you
are using a preset model, with a preset parameter distribution, but you only want to change 1 parameter in the model while still keeping
the rest of the parameters associated with the param distribution. In this case, you could pass that value in extra params.

`extra params` are passed as in kwargs style, which means as extra named params, where the names are the names of parameters (only those accessible to the base classes init), for example
if we were selecting the 'dt' ('decision tree') :class:`Model<BPt.Model>`, and we wanted to use the first built in
preset distribution for :ref:`Params`, but then fix the number of `max_features`, we could do it is as:

::

    model = Model(obj = 'dt',
                  params = 1,
                  max_features = 10)

Note: Any parameters passed as extra params will override any values if overlapping with the fixed passed params = 1. In other
words, parameters passed as extra have the highest priority. 
                  

.. _Custom Input Objects:

Custom Input Objects
=====================

Custom input objects can be passed to the `obj` parameter for a number of base :class:`ModelPipeline <BPt.ModelPipeline>` pieces.
