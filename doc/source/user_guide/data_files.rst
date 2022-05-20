.. _data_files:

{{ header }}

***********
Data Files
***********

.. currentmodule:: BPt

Data files allow BPt to work with any type of arbitrary data beyond simply tabular data.
For example, Data File's can be used to load and work with volumetric or surface based neuroimaging data
or even connectome data.

See :func:`add_data_files <Dataset.add_data_files>` for how to load data files to a Dataset. Likewise,
additional function :func:`to_data_file <Dataset.to_data_file>` may be used.

During construction of a model pipeline any loaded data files must have a corresponding
:class:`Loader` where those files will be converted into valid down stream predictive features.

**Examples:**

.. toctree::
    :maxdepth: 1

    load_timeseries_example