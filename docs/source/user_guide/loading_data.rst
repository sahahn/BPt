.. _loading_data:

{{ header }}

**************
Loading Data
**************

.. currentmodule:: BPt

Intro
------

In order to get data ready for machine learning, BPt has a specially designed :class:`Dataset` class.
This class is new as of BPt version >= 2 (replacing the building in loading functions of the
old BPt_ML). This class is built on top of the DataFrame from the pandas library. As we will see,
the recommended way of preparing data actually first involves using the DataFrame class from pandas
directly. The is that pandas and the DataFrame class should be used to load all of the data you might
end up wanting to use. Luckily pandas contains a huge wealth of useful functions
for accomplishing this already. Next, once all of the data is loaded, we cast the DataFrame
to the BPt :class:`Dataset` class, and then use the built in :class:`Dataset` methods to get the data
ready for use with the rest of BPt. This includes steps like specifying which variables are in what
role (e.g., target variables vs. data variables), outlier detection, transformations like binning and converting to binary,
tools for plotting / viewing distributions and specifying a global train / test split. We will introduce all of
this functionality below!


Data of interest is inevitably going to come from a wide range of different sources, luckily the python library pandas
has an incredible amount of support for loading data from different sources into DataFrames.
Likewise, pandas offers a huge amount of support material, e.g., https://pandas.pydata.org/docs/getting_started/intro_tutorials/02_read_write.html
for getting starting with loading in raw data (or a google search with a specific question will almost always help).
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

We can now still use a number of the native pandas api methods in addition now to the added functionality of the BPt :class:`Dataset`!

There are a few key concepts when using :class:`Dataset` which are important to know.
These are :ref:`role`, :ref:`scope`, :ref:`subjects`, :ref:`data_types` and :ref:`data_files`.

Warnings with using :class:`Dataset`:

Column names within the :class:`Dataset` class must be strings in order for the concept
of scopes to work more easily. Therefore if any columns are loaded as a non-string, they
will be renamed to the string version of that non-string.

Their are some caveats to using some DataFrame function once the DataFrame has
been cast as a :class:`Dataset`. While a great deal will continue to work, their are
certain types of operations which can end up either re-casting the result back to
a DataFrame (therefore losing all of the associated metadata), or renaming columns,
which may cause internal errors and metadata loss.

Basic Example
--------------

.. ipython:: python

   import BPt as bp

   data = bp.Dataset()
   data['col 1'] = [1, 2, 3]
   data

We can then perform operations on it, for example change its role.

.. ipython:: python
  
  data.set_role('col 1', 'target')
  data.roles

What happened here? It looks like the role of the target is still 'data' and not 'target'. That 
is because the Dataset class, like the underlying pandas DataFrame, has an inplace argument.
This gives use two options, where both of the below operations will correctly set the role.

.. ipython:: python
  
  data = data.set_role('col 1', 'target')
  data.set_role('col 1', 'target', inplace=True)
  data.roles