# Brain Predictability Toolbox (BPt)

[![codecov](https://codecov.io/gh/sahahn/BPt/branch/master/graph/badge.svg?token=SCA77VAUAG)](https://codecov.io/gh/sahahn/BPt) [![pip](https://badge.fury.io/py/brain-pred-toolbox.svg)](https://pypi.org/project/brain-pred-toolbox/) [![status](https://github.com/sahahn/BPt/actions/workflows/test_ubuntu_versions.yml/badge.svg)](https://github.com/sahahn/BPt/actions) [![status](https://github.com/sahahn/BPt/actions/workflows/test_mac_versions.yml/badge.svg)](https://github.com/sahahn/BPt/actions) [![status](https://github.com/sahahn/BPt/actions/workflows/test_windows_versions.yml/badge.svg)](https://github.com/sahahn/BPt/actions) ![version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9-blue) [![Downloads](https://static.pepy.tech/personalized-badge/brain-pred-toolbox?period=total&units=international_system&left_color=black&right_color=grey&left_text=Downloads)](https://pepy.tech/project/brain-pred-toolbox)


![logo](https://github.com/sahahn/BPt/blob/master/doc/source/_static/red_logo.png?raw=true)

- The Brain Predictability toolbox (BPt) is a Python based Machine Learning library designed for working   with Neuroimaging data. This library is particularly suited towards working with large neuroimaging datasets, where a large number of subjects with potentially multi-modal data are available.

- Warning: As of 11/30/22, BPt is entering a sort of legacy mode, where it will be no longer actively
  developed. The library should still continue to work, as before, but the required python packages
  are now far more restrictive than before, and may require a dedicated conda or virtual environment.
  Unfortunately this is the nature of open-source academic software when not maintained by a dedicated
  community, but that said, if anyone is interested in taking over,
  they can feel free to message me about it (though in all honesty it may be a better
  use of time to contibute to other projects with a more stable developer community instead).

- Please check out the project documentation at:
<https://sahahn.github.io/BPt/>

- This library is based on python and likely will require atleast some prior experience with python and machine learning.


### Install
----

**Note:** *This library is only tested on python versions 3.7+ so while 3.6 might work, for the most reliable performance please use higher versions of python!*


The easiest way to install the latest stable release of BPt is via pip, just run
``` 
pip install brain-pred-toolbox 
```

The other method, to get the latest stable development version of the library is to clone this repository,
and then install it locally with once navigated into the main BPt folder

```
pip install .
```

### Quick Start Example

Load a pre-set BPt dataset, then
run a default 5-fold cross validation.

```
from BPt.datasets import load_cali
from BPt import evaluate

data = load_cali()
results = evaluate('elastic_pipe', data)
```

The returned object, stored in variable results, is an instance of class [EvalResults](https://sahahn.github.io/BPt/reference/api/BPt.EvalResults.html#BPt.EvalResults), which contains all types of information and metrics from the stored evaluation.

Check out the documentation at <https://sahahn.github.io/BPt/> for more examples on how to get started using BPt!