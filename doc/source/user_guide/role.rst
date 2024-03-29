.. _role:

{{ header }}

***********
Role
***********

.. currentmodule:: BPt

There are three possible roles in the :class:`Dataset` class / BPt. These are
'data', 'target' and 'non input'. By default, all loaded columns will be treated with
role 'input data' until set differently. Roles are set through
methods :func:`set_role <Dataset.set_role>`
and :func:`set_roles <Dataset.set_roles>`. Or through dedicated
helper methods :func:`set_target <Dataset.set_target>` and
:func:`set_non_input <Dataset.set_non_input>`.

The different roles are described below.

- input data
    The default role, input data, is used to indicate all columns which might at some point serve as input features for an eventual predictive task.
    Data can have any of the :ref:`Data_Types` including :ref:`Data_Files`. NaN's are allowed in data columns. The basic idea is that when
    passing a :class:`Dataset` to :func:`evaluate`, by default every column with role='input data' will be used as input to predict, unless this set
    of variables is limited in some way by passing a :ref:`scope`. Note, you may also access this role with shorted keyword 'data'.

- target
    The role of target is used to indicate columns which are to be predicted, and therefore will not serve as input features for any predictive tasks.
    Targets can take on any of the :ref:`Data_Types` except :ref:`Data_Files`. Target columns can include NaN values, although be warned that trying
    passing a target with NaN values to some functions may not work correctly. Likewise, during prediction, any targets with NaN values will not ever
    be used in any training sets, or contribute to any generated metrics on validation sets. In the case of multiple columns with role='target', then
    a specific target variable may be set when using :func:`evaluate`.

- non input
    As the name suggests, any features set with role non input, will not be provided directly as input features to a predictive task.
    Instead, these features are usually categorical and can be used to inform cross-validation behavior or
    to examine predictive results under different groupings. For example see :func:`copy_as_non_input <Dataset.copy_as_non_input>` to
    make an ordinalized copy of an existing column. While there is no strict requirement that columns with role non input be categorical,
    there is a fixed requirement that any columns with role non input cannot contain any NaN's. 