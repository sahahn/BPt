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

    from ABCD_ML import ABCD_ML
    Your_ABCD_ML_Object = ABCD_ML(init_params)


Alternatively, if you wish to load from an already saved object, you would do as follows

::

    from ABCD_ML import Load
    Your_ABCD_ML_Object = Load(location)


Load
====
.. automethod:: ABCD_ML.main.ABCD_ML.Load

To init params as referenced above are those listed here under Init.

Init
============

.. currentmodule:: ABCD_ML
.. autoclass:: ABCD_ML

******************
Loading Phase
******************

The next 'phase', is the where all of the loading is done, and the structure of the
desired expiriments set up.

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

Load_Data_Files
================
.. automethod:: ABCD_ML.Load_Data_Files

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

Show_Strat_Dist
===============
.. automethod:: ABCD_ML.Show_Strat_Dist

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

Get_Nan_Subjects
================
.. automethod:: ABCD_ML.Get_Nan_Subjects


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

Set_Default_ML_Verbosity
=========================
.. automethod:: ABCD_ML.Set_Default_ML_Verbosity

Evaluate
========
.. automethod:: ABCD_ML.Evaluate

Plot_Global_Feat_Importances
=============================
.. automethod:: ABCD_ML.Plot_Global_Feat_Importances

Plot_Local_Feat_Importances
=============================
.. automethod:: ABCD_ML.Plot_Local_Feat_Importances

*************
Testing Phase
*************

Test
========
.. automethod:: ABCD_ML.Test

Plot_Global_Feat_Importances
=============================
.. automethod:: ABCD_ML.Plot_Global_Feat_Importances

Plot_Local_Feat_Importances
=============================
.. automethod:: ABCD_ML.Plot_Local_Feat_Importances


*************
Extras
*************

Save
======
.. automethod:: ABCD_ML.Save

Save_Table
=============
.. automethod:: ABCD_ML.Save_Table
