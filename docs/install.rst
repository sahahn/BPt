From Scratch
================

Don't even have python installed, or anything? It is reccomended to download anaconda first
https://www.anaconda.com/distribution/#download-section
as that will take care of a number of dependecies right away (and ease cross os difficulties),
and give you a jupyter notebook environment to optionally work in.


Pip Installation
================

To download the latest stable release, you can do this through pip, pythons built in installer. 
Run on the command line,

::

    pip install ABCD_ML


Github / Pip Installation
=========================

Optionally, you can choose to download the latest development version through github.
You need git installed for this, but on the plus side you are ensured to have the latest version.
Run on the command line or anaconda prompt on windows, (In the location where you want ABCD_ML installed!)

::

    git clone https://github.com/sahahn/ABCD_ML.git

Then navigate into the ABCD_ML folder, and run

::

    cd ABCD_ML
    pip install .

In the future, to grab the latest updated versions, navigate into the folder where you installed ABCD_ML, and run

::

    git pull
    pip install .


Extra Libraries
=========================
Lightgbm and Xgboost are both popular libraries for performing fast extreme gradient boosting.
Some support for both of these libraries is built into ABCD_ML, but they are not listed as explicit
requiriments as they tend to not install correctly via pip all of the time (i.e., on macs). If you wish to use
either of these libraries, please download them yourself!

Lightgbm python install instructions: https://github.com/microsoft/LightGBM/tree/master/python-package
Xgboost install instructions: https://xgboost.readthedocs.io/en/latest/build.html