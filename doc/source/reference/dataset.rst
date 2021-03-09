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
   Dataset.copy
   Dataset.auto_detect_categorical
   Dataset.get_Xy
   Dataset.get_train_Xy
   Dataset.get_test_Xy

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
   Dataset.show
   Dataset.show_nan_info
   Dataset.info

********************************
Dataset - Test Split Methods
********************************

.. autosummary::
   :toctree: api/

   Dataset.set_test_split
   Dataset.set_train_split
   Dataset.save_test_split
   Dataset.save_train_split
