**********
Init Phase
**********

Import
======
To start, you must import the module.
Assuming that it has been downloaded of course.
Import and then make an object, in this example the obj is called "Your_ABCD_ML_Object",
but in practice you would want to choose something shorter, like "ML".

::

    import ABCD_ML
    Your_ABCD_ML_Object = ABCD_ML.ABCD_ML(init_params)

Init
============

.. currentmodule:: ABCD_ML.ABCD_ML
.. autoclass:: ABCD_ML


******************
Data Loading Phase
******************

Set_Default_Load_Params
=======================
.. automethod:: ABCD_ML.Set_Default_Load_Params

Load_Name_Map
==============
.. automethod:: ABCD_ML.Load_Name_Map

Load_Exclusions
===============
.. automethod:: ABCD_ML.Load_Exclusions

Load_Inclusions
===============
.. automethod:: ABCD_ML.Load_Inclusions

Load_Data
=========
.. automethod:: ABCD_ML.Load_Data

Drop_Data_Cols
==============
.. automethod:: ABCD_ML.Drop_Data_Cols

Filter_Data_Cols
================
.. automethod:: ABCD_ML.Filter_Data_Cols

Proc_Data_Unique_Cols
=====================
.. automethod:: ABCD_ML.Proc_Data_Unique_Cols

Drop_Data_Duplicates
=====================
.. automethod:: ABCD_ML.Drop_Data_Duplicates

Show_Data_Dist
==============
.. automethod:: ABCD_ML.Show_Data_Dist

Load_Targets
============
.. automethod:: ABCD_ML.Load_Targets

Binarize_Target
================
.. automethod:: ABCD_ML.Binarize_Target

Show_Targets_Dist
==================
.. automethod:: ABCD_ML.Show_Targets_Dist

Load_Covars
============
.. automethod:: ABCD_ML.Load_Covars

Show_Covars_Dist
==================
.. automethod:: ABCD_ML.Show_Covars_Dist

Load_Strat
===========
.. automethod:: ABCD_ML.Load_Strat

Get_Overlapping_Subjects
========================
.. automethod:: ABCD_ML.Get_Overlapping_Subjects

Clear_Name_Map
==============
.. automethod:: ABCD_ML.Clear_Name_Map

Clear_Exclusions
================
.. automethod:: ABCD_ML.Clear_Exclusions

Clear_Data
==========
.. automethod:: ABCD_ML.Clear_Data

Clear_Targets
==============
.. automethod:: ABCD_ML.Clear_Targets

Clear_Covars
=============
.. automethod:: ABCD_ML.Clear_Covars

Clear_Strat
============
.. automethod:: ABCD_ML.Clear_Strat


****************
Validation Phase
****************

Define_Validation_Strategy
===========================
.. automethod:: ABCD_ML.Define_Validation_Strategy

Train_Test_Split
==========================
.. automethod:: ABCD_ML.Train_Test_Split


****************
Modeling Phase
****************

Set_Default_ML_Params
======================
.. automethod:: ABCD_ML.Set_Default_ML_Params

Set_Default_ML_Verbosity
=========================
.. automethod:: ABCD_ML.Set_Default_ML_Verbosity

Evaluate
========
.. automethod:: ABCD_ML.Evaluate

Show_Models
================
.. automethod:: ABCD_ML.Show_Models

Show_Metrics
==============
.. automethod:: ABCD_ML.Show_Metrics

Show_Imputers
===============
.. automethod:: ABCD_ML.Show_Imputers

Show_Scalers
=================
.. automethod:: ABCD_ML.Show_Scalers

Show_Samplers
=============
.. automethod:: ABCD_ML.Show_Samplers

Show_Feat_Selectors
===================
.. automethod:: ABCD_ML.Show_Feat_Selectors

Show_Ensembles
===================
.. automethod:: ABCD_ML.Show_Ensembles

Get_Base_Feat_Importances
=========================
.. automethod:: ABCD_ML.Get_Base_Feat_Importances

Get_Shap_Feat_Importances
=========================
.. automethod:: ABCD_ML.Get_Shap_Feat_Importances


*************
Testing Phase
*************

Test
========
.. automethod:: ABCD_ML.Test

