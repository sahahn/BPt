**********
Init Phase
**********

Import
======
To start, you must import the module.
Assuming that it has been downloaded of course.
Import and then make an object, in this example the obj is called "ML"

::

    from BPt import BPt_ML
    ML = BPt_ML(**init_params)


Alternatively, if you wish to load from an already saved object, you would do as follows

::

    from BPt import Load
    ML = Load(saved_location)


Load
====
.. automethod:: BPt.main.BPt_ML.Load

To init params as referenced above are those listed here under Init.

Init
============

.. currentmodule:: BPt
.. autoclass:: BPt_ML

******************
Loading Phase
******************

The next 'phase', is the where all of the loading is done, and the structure of the
desired expiriments set up.

Set_Default_Load_Params
=======================
.. automethod:: BPt_ML.Set_Default_Load_Params

Load_Name_Map
==============
.. automethod:: BPt_ML.Load_Name_Map

Load_Exclusions
===============
.. automethod:: BPt_ML.Load_Exclusions

Load_Inclusions
===============
.. automethod:: BPt_ML.Load_Inclusions

Load_Data
=========
.. automethod:: BPt_ML.Load_Data

Load_Data_Files
================
.. automethod:: BPt_ML.Load_Data_Files

Drop_Data_Cols
==============
.. automethod:: BPt_ML.Drop_Data_Cols

Filter_Data_Cols
================
.. automethod:: BPt_ML.Filter_Data_Cols

Proc_Data_Unique_Cols
=====================
.. automethod:: BPt_ML.Proc_Data_Unique_Cols

Drop_Data_Duplicates
=====================
.. automethod:: BPt_ML.Drop_Data_Duplicates

Show_Data_Dist
==============
.. automethod:: BPt_ML.Show_Data_Dist

Load_Targets
============
.. automethod:: BPt_ML.Load_Targets

Binarize_Target
================
.. automethod:: BPt_ML.Binarize_Target

Show_Targets_Dist
==================
.. automethod:: BPt_ML.Show_Targets_Dist

Load_Covars
============
.. automethod:: BPt_ML.Load_Covars

Show_Covars_Dist
==================
.. automethod:: BPt_ML.Show_Covars_Dist

Load_Strat
===========
.. automethod:: BPt_ML.Load_Strat

Show_Strat_Dist
===============
.. automethod:: BPt_ML.Show_Strat_Dist

Get_Overlapping_Subjects
========================
.. automethod:: BPt_ML.Get_Overlapping_Subjects

Clear_Name_Map
==============
.. automethod:: BPt_ML.Clear_Name_Map

Clear_Exclusions
================
.. automethod:: BPt_ML.Clear_Exclusions

Clear_Data
==========
.. automethod:: BPt_ML.Clear_Data

Clear_Targets
==============
.. automethod:: BPt_ML.Clear_Targets

Clear_Covars
=============
.. automethod:: BPt_ML.Clear_Covars

Clear_Strat
============
.. automethod:: BPt_ML.Clear_Strat

Get_Nan_Subjects
================
.. automethod:: BPt_ML.Get_Nan_Subjects


****************
Validation Phase
****************

Define_Validation_Strategy
===========================
.. automethod:: BPt_ML.Define_Validation_Strategy

Train_Test_Split
==========================
.. automethod:: BPt_ML.Train_Test_Split


****************
Modeling Phase
****************

Set_Default_ML_Verbosity
=========================
.. automethod:: BPt_ML.Set_Default_ML_Verbosity

Evaluate
========
.. automethod:: BPt_ML.Evaluate

Plot_Global_Feat_Importances
=============================
.. automethod:: BPt_ML.Plot_Global_Feat_Importances

Plot_Local_Feat_Importances
=============================
.. automethod:: BPt_ML.Plot_Local_Feat_Importances

*************
Testing Phase
*************

Test
========
.. automethod:: BPt_ML.Test

Plot_Global_Feat_Importances
=============================
.. automethod:: BPt_ML.Plot_Global_Feat_Importances

Plot_Local_Feat_Importances
=============================
.. automethod:: BPt_ML.Plot_Local_Feat_Importances


*************
Extras
*************

Save
======
.. automethod:: BPt_ML.Save

Save_Table
=============
.. automethod:: BPt_ML.Save_Table
