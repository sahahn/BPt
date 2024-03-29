{{ header }}

.. _api.dataset:

=========
Dataset
=========

.. currentmodule:: BPt

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Dataset

Base
~~~~~

.. autosummary::
   :toctree: api/

   Dataset.get_cols
   Dataset.get_subjects
   Dataset.get_values
   Dataset.add_scope
   Dataset.remove_scope
   Dataset.set_role
   Dataset.set_roles
   Dataset.get_roles
   Dataset.rename
   Dataset.copy
   Dataset.auto_detect_categorical
   Dataset.get_Xy
   Dataset.get_permuted_Xy
   Dataset.split_by

Encoding
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   Dataset.to_binary
   Dataset.binarize
   Dataset.k_bin
   Dataset.ordinalize
   Dataset.nan_to_class
   Dataset.copy_as_non_input
   Dataset.add_unique_overlap

Data Files
~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   Dataset.add_data_files
   Dataset.to_data_file
   Dataset.consolidate_data_files
   Dataset.update_data_file_paths
   Dataset.get_file_mapping


Filtering & Drop
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   Dataset.filter_outliers_by_std
   Dataset.filter_outliers_by_percent
   Dataset.filter_categorical_by_percent
   Dataset.drop_cols
   Dataset.drop_nan_subjects
   Dataset.drop_subjects_by_nan
   Dataset.drop_cols_by_unique_val
   Dataset.drop_cols_by_nan
   Dataset.drop_id_cols
   Dataset.drop_duplicate_cols
   Dataset.apply_inclusions
   Dataset.apply_exclusions


Plotting / Viewing
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   Dataset.plot
   Dataset.plots
   Dataset.plot_bivar
   Dataset.nan_info
   Dataset.summary
   Dataset.display_scopes


Train / Test Split
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   Dataset.set_test_split
   Dataset.set_train_split
   Dataset.test_split
   Dataset.train_split
   Dataset.save_test_split
   Dataset.save_train_split
