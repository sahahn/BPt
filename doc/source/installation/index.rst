{{ header }}

.. _installation:

=============
Installation
=============

.. raw:: html

    <div class="container">
        <div class="row">
            <div class="col-lg-6 col-md-6 col-sm-12 col-xs-12 d-flex install-block">
                <div class="card install-card shadow w-100">
                <div class="card-header">
                    pip
                </div>
                <img src="../_static/pip.jpg" class="card-img-top" alt="python pip logo" height="260">
                <div class="card-body">
                    <p class="card-text">

The latest stable version of BPt can be found and installed through pip the python packaging system.

.. raw:: html

                    </p>
                </div>
                <div class="card-footer text-muted">

.. code-block:: bash

   pip install brain-pred-toolbox

.. raw:: html

                </div>
                </div>
            </div>
            <div class="col-lg-6 col-md-6 col-sm-12 col-xs-12 d-flex install-block">
                <div class="card install-card shadow w-100">
                <div class="card-header">
                    Github
                </div>
                <img src="../_static/github.png" class="card-img-top" alt="github logo" height="200">
                <div class="card-body">
                    <p class="card-text">

The latest development version of BPt can also optionally be installed from github directly.

.. raw:: html

                    </p>
                </div>
                <div class="card-footer text-muted">

.. code-block:: bash

   git clone https://github.com/sahahn/BPt
   cd BPt
   pip install .

.. raw:: html

                </div>
                </div>
            </div>
        </div>
    </div>

=================
Python version
=================

This library is only tested on python versions 3.7+ so while 3.6 might work,
for the most reliable performance please use higher versions of python!


=================
Extra Libraries
=================

BPt has a number of other optional requirements, then when installed allow using more default options. These are not
added as required libraries for a few reasons, either to keep the number of dependencies down, or because sometimes
installation of these libraries is non-trivial.

The different extension libraries can be downloaded with ::
    
    pip install brain-pred-toolbox[extra]

Though note, some may not download properly via pip depending on your operating system.

Different extension libraries are listed below:

bp-neurotools
~~~~~~~~~~~~~~

This is a library by the same maintainers as BPt. It is designed to be less ML specific, but
still contains some useful utilites for neuroimaging ML. See https://github.com/sahahn/neurotools.

lightgbm
~~~~~~~~~~~

This is a library designed to perform extreme gradient boosting. It
is offered under :ref:`Models` under reserved keys 'light gbm' and 'lgbm'.
See https://lightgbm.readthedocs.io/en/latest/Python-Intro.html if having trouble installing through pip.

nilearn
~~~~~~~~

This is a library dedicated to doing ML for neuroimaging, if installed it
allows use of :class:`BPt.extensions.SingleConnectivityMeasure`.
                   
python-docx
~~~~~~~~~~~~~

This library is required to use the save_file option of :func:`BPt.Dataset.summary` and
:func:`BPt.util.save_docx_table`. It is used for saving tables in docx format.

xgboost
~~~~~~~~~

This is another library for performing extreme gradient boosting. It is offered
under :ref:`Models`.

mvlearn
~~~~~~~~~

There is experimental support through the BPt extensions for using objects like CCA from the
multi-vew learn library, mvlearn.

imblearn
~~~~~~~~~~~

There is experimental support for the use of some ensembles methods from library imblearn.
These are ensembles that ensure bagging is done in a class-balanced manner.
