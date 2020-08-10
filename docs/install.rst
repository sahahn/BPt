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

    pip install BPt


Github / Pip Installation
=========================

Optionally, you can choose to download the latest development version through github.
You need git installed for this, but on the plus side you are ensured to have the latest version.
Run on the command line or anaconda prompt on windows, (In the location where you want BPt installed!)

::

    git clone https://github.com/sahahn/BPt.git

Then navigate into the BPt folder, and run

::

    cd BPt
    pip install .

In the future, to grab the latest updated versions, navigate into the folder where you installed BPt, and run

::

    git pull
    pip install .


Extra Libraries
=========================
There are a number of libraries which extend the functionality of the BPt. These can
be found under docs in the file requiriments.txt. Notably, depending on your operating system,
some of these additional libraries may not be installable through pip alone, and will require 
taking further library specific steps.