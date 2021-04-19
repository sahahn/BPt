![logo](https://github.com/sahahn/BPt/blob/master/doc/source/_static/red_logo.png?raw=true)

# Brain Predictability Toolbox (BPt)

- The Brain Predictability toolbox (BPt) is a Python based Machine Learning library designed for working with Neuroimaging data. This library is particularly suited towards working with large neuroimaging datasets, where a large number of subjects with potentially multi-modal data are available.

- Please check out the project documentation at:
<https://sahahn.github.io/BPt/>

- This library is based on python and likely will require atleast some prior experience with python and machine learning.

- This library should work with all python version 3.6+, but for the most reliable performance for all methods and cases, please use python version 3.8+. Note with python 3.6 the only currently known issue is related to caching certain pipeline components, which was solved in an upgrade to the python pickle library.


### Install

The easiest way to install the latest stable release of BPt is via pip, just run
``` 
pip install brain-pred-toolbox 
```

The other method, to get the latest stable development version of the library is to clone this repository,
and then install it locally with once navigated into the main BPt folder

```
pip install .
```



### Old Version

BPt 2.0+  is notably not fully compatible with earlier versions of the code. If still working with older code, it is recommended you update, but you
can still view a version of the the old documentation for now at: https://bpt.readthedocs.io/en/latest/
