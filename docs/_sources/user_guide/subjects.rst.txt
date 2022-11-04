.. _subjects:

{{ header }}

***********
Subjects
***********

.. currentmodule:: BPt

Various functions within BPt, and :class:`Dataset` can accept subjects or some variation on this
name as an argument. The idea of subjects is similar to :ref:`scope`, where essentially subjects allows
for an expanded upon index'ing system but at the row-level, whereas :ref:`scope` operates as the column level.

The parameter can accept a few different values. These are explained below:

1. array-like
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You may pass any array-like (e.g., list, set, pandas Index, etc...) of subjects directly.
Warning: Passing a python tuple is reserved for a special MultiIndex case!

For example:

::

  subjects = ['subj1', 'subj2', 'subj3']

Would select those three subjects, where the list could also be a numpy array or pandas index for example.

2. location 
~~~~~~~~~~~~~~~~~~~

You may pass the location of a text file were subject's names are stored as one subject's
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

3. reserved keyword
~~~~~~~~~~~~~~~~~~~~~
A reserved key word may be passed. These include:

- 'all'
  Operate on all subjects

- 'nan'
  Select any subjects with any missing values in any of their loaded columns,
  regardless of scope or role.

- 'not nan'
  Like 'nan' but not. Or in english, any subjects without any missing values in
  any of their loaded columns, regardless of scope or role.

- 'train'
  Select the set of train subjects as defined by a split in the Dataset, e.g.,
  :func:`set_train_split <BPt.Dataset.set_train_split>`.

- 'test'
  Select the set of test subjects as defined by a split in the Dataset, e.g.,
  :func:`set_test_split <BPt.Dataset.set_test_split>`.

- 'default'
  This is the default subjects value for :class:`ProblemSpec <BPt.ProblemSpec>`,
  it refers to special behavior where when evaluating if the passed dataset has
  a train/test split defined, and a cv value is passed that isn't 'test', then
  subjects = 'train' will be used. Otherwise, subjects='all' will be used.


1. value subset case
~~~~~~~~~~~~~~~~~~~~~
You can pass the special input wrapper :class:`ValueSubset`. This can be
used to select subsets of subject by a column's
value or values. See :class:`ValueSubset` for more
information on how this input class is used.

5. multi-index case
There also exists the case where you may wish for the underlying index of subjects to be a MultiIndex.
In this case, there is some extra functionality to discuss.
Say for example we have a Dataset multi-indexed
by subject and eventname, e.g.,

::

  data.set_index(['subject', 'eventname'], inplace=True)

We now have more options for how we might want to index this dataset, and therefore more options
for valid arguments to pass to a subjects argument. Consider first all of the examples from above,
where we are just specifying a subject-like index. In this case,
all of those arguments will still work, and will just return all subjects
with all of their eventnames. E.g., assuming there were two eventname values
for each subjects 'e1' and 'e2':

::

  subjects = ['subj1', 'subj2']

Would select subject eventname pairs:
('subj1', 'e1'), ('subj1', 'e2'), ('subj2', 'e1'), ('subj2', 'e2')
and likewise with loading from a text file which just specified 'subj1' and 'subj2'.

Note that if we pass arguments in this manner, BPt will assume they refer to whatever
index is first, in this case 'subject', and not 'eventname'. If we wish to also select
explicitly by eventname, we have two options.

6. multi-index array-like
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You may pass fully indexed tuples in an array-like manner, the same as 1. from before, e.g.:

::

  subjects = ('subj1', 'e1'), ('subj2', 'e2')

To just keep 'subj1' at event 'e1' and 'subj2' at 'e2'. Likewise, we
could select this same subset if subjects.txt was formatted as:

::

  ('subj1', 'e1')
  ('subj2', 'e2')

Our second option is to use the special tuple reserved input. In this case,
we must pass a python tuple with the same length at the number of levels in the
underlying MultiIndex, e.g., in the example before, of length two. Each index in the
tuple will then be used to specify the BPt subjects compatible argument for just that
level of the index. For example:

::

  subjects = ('all', ['e1'])

Would select all subjects, and then note the array-like list in the second index of the tuple,
would filter that to include only subject eventname
pairs with an eventname of 'e1'. Consider another example:

::

  subjects = ('subjects.txt', 'events.txt')

In this case, the subjects to select would be loaded from
'subjects.txt' and the corresponding eventnames from
'events.txt'.

7. name of column
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This option only works in the context of the function :class:`evaluate`, where you may
pass the name of a loaded column, and have the argument converted intenerally to a 
special :class:`Compare` style object, :class:`CompareSubset`, where subsets of 
of subjects would be defined for each unique value in the name of the column passed.

For example:

::

  subjects='sex'

Where 'sex' is a loaded column in a dataset, and would then define a :class:`Compare` object
with seperate options for each unique value of 'sex'.


Examples
~~~~~~~~~~~

First let's define an example dataset to show some examples with.

.. ipython:: python

  import BPt as bp
  import numpy as np

  data = bp.Dataset()
  data['index'] = ['subj1', 'subj2', 'subj3']
  data['col1'] = [1, 2, np.nan]
  data.set_index('index', inplace=True)
  data

Next, we will use :func:`Dataset.get_subjects` to explore what passing
different values to `subjects` will return.

.. ipython:: python
  
  data.get_subjects(subjects=['subj1'])
  data.get_subjects(subjects=['subj1', 'subj2'])

One gotcha is that if we pass a single str value, it will be assumed to
be a file path, so if we pass subjects='subj1', we will get an error.
Let's try using :class:`ValueSubset` next.

.. ipython:: python

  data.get_subjects(bp.ValueSubset('col1', 1))
  data.get_subjects(bp.ValueSubset('col1', [1, 2]))

We can also use special reversed keywords.

.. ipython:: python

  data.get_subjects('nan')
  data.get_subjects('not nan')
  




