.. _why_bpt:

{{ header }}

***********
Why BPt?
***********

The Brain Predictability toolbox (BPt) is a python based library with a unified framework of machine learning (ML) tools designed to work with both
tabulated data (e.g. brain derived, psychiatric, behavioral and physiological variables) and neuroimaging specific data (e.g. brain volumes and surfaces).
This package is suitable for investigating a wide range of different neuroimaging-based ML questions,
in particular, those queried from large human datasets.

BPt seeks to integrate a number of lower level packages towards providing a higher level package and user experience. 
The key point of interest here is that BPt is designed specifically for performing neuroimaging based machine learning.
Towards this end, it builds upon and offers augmented capabilities for core libraries pandas and scikit-learn (as well as others).
A central goal of the library is to automate key pieces of the workflow, with a large array of default behaviors and pipelines, as well as
always also allowing the user full control. This package works to reduce the barrier of entry as ultimately using machine learning
as a research tool should not be restricted only to domain experts.

The toolbox is designed primarily for ‘population’ based predictive neuroimaging, that is to say, machine learning performed across data from multiple
participants rather than many data points from a single or small set of participants.
BPt does support multi-indexing in this latter case, but for the most part the functionality of the toolbox more readily
supports the former. Input data for the toolbox can take a wide range of forms, but generally speaking are
the outputs from a typical neuroimaging pre-processing pipeline. The easiest data to work with are data already in tabular
form, e.g., calculated mean values per region of interest. That said, the toolbox is capable of working with volumetric or
surface projected MRI or fMRI data as well. Other modalities, like EEG, could use the toolbox,
but in these cases it may be a poorer fit (as EEG often requires quite different pre-processing steps). 

In general, the use of any framework like BPt imposes a practical trade-off between flexibility and ease of use.
In the case of working with BPt, as long as the dataset and desired type of analysis are supported,
then a great deal of minor steps, opportunities for bugs, and decisions can be handled automatically.
Alternatively, if a specific analysis isn’t supported, for example using deep learning classifiers as
the machine model of interest, then BPt will be a poor choice.
