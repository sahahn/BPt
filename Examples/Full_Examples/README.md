This folder contains Full Examples, which aim to be notebooks covering more complete project workflows, i.e., from loading data to generating meaningful predictions.

----------------------

## Predicting Substance Dependence from Multi-Site Data

This notebook explores an example using data from the ENIGMA Addiction Consortium. Within this notebook we will be trying to predict between participants with any drug dependence (alcohol, cocaine, etc...), vs. healthy controls. The data for this is sources from a number of individual studies from all around the world and with different scanners etc... making this a challenging problem with its own unique considerations. Structural FreeSurfer ROIs are used. The raw data cannot be made available due to data use agreements.

The key idea explored in this notebook is a particular tricky problem introduced by case-only sites, which are subject's data from site's with only case's. This introduces a confound where you cannot easily tell if the classifier is learning to predict site or the dependence status of interest.

Featured in this notebook as well are some helpful
code snippets for converting from BPt versions earlier than BPt 2.0 to valid BPt 2.0+ code.

----------------------
s
## Predict BMI From Change In Brain

This script shows a real world example using BPt to study the relationship between BMI and the brain. Interestingly, we employ longitudinal brain data from two time points as represented by a change in brain measurements between time points. The data used in this notebook cannot be made public as it is from the ABCD Study, which requires a data use agreement in order to use the data.

This notebook covers a number of different topics:

- Preparing Data
- Evaluating a single pipeline
- Considering different options for how to use a test set
- Use a LinearResidualizer to residualize input brain data
- Introduce and use the Evaluate input option


-----------------------

## Predict Waist Circumference with Diffusion Weighted Imaging

This notebook using diffusion weighted imaging data, and subjects waist circumference in cm from the ABCD Study.
We will use as input feature derived Restriction spectrum imaging (RSI) from diffusion weighted images. This notebook
covers data loading as well as evaluation across a large number of different ML Pipelines. This notebook may be useful
for people looking for more examples on what different Pipelines to try.


----------------------

## Predict Stop Signal Response Time  (SSRT)

This notebook uses stop signal task fMRI data derived contrasts from the ABCD study to 
predict stop signal response time (SSRT). This is an example of a regression type machine learning,
and additionally includes an extra example of how to plot ROIs feature importance on brain surfaces from nilearn. 

----------------------

## Predict Sex

This notebook goes through a simple binary classification example, explaining general library functionality along the way.
Within this notebook we make use of data downloaded from Release 2.0.1 of the the ABCD Study (https://abcdstudy.org/).
This dataset is openly available to researchers (after signing a data use agreement) and is particularly well suited
towards performing neuroimaging based ML given the large sample size of the study.

Within this notebook we will be performing binary classification predicting sex assigned at birth from tabular ROI structural MRI data.

