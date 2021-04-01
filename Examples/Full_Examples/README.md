This folder contains Full_Examples, which aim to be notebooks covering more complete project workflows, i.e., from loading data to generating meaningful predictions.

----------------------

## Predicting Substance Dependence from Multi-Site Data

This notebook explores an example using data from the ENIGMA Addiction Consortium. Within this notebook we will be trying to predict between participants with any drug dependence (alcohol, cocaine, etc...), vs. healthy controls. The data for this is sources from a number of individual studies from all around the world and with different scanners etc... making this a challenging problem with its own unique considerations. Structural FreeSurfer ROIs are used. The raw data cannot be made available due to data use agreements.

The key idea explored in this notebook is a particular tricky problem introduced by case-only sites, which are subject's data from site's with only case's. This introduces a confound where you cannot easily tell if the classifier is learning to predict site or the dependence status of interest.

Featured in this notebook as well are some helpful
code snippets for converting from BPt versions earlier than BPt 2.0 to valid BPt 2.0+ code.

----------------------

## Predict BMI From Change In Brain

This script shows a real world example using BPt to study the relationship between BMI and the brain. Interestingly, we employ longitudinal brain data from two time points as represented by a change in brain measurements between time points. The data used in this notebook cannot be made public as it is from the ABCD Study, which requires a data use agreement in order to use the data.

This notebook covers a number of different topics:

- Preparing Data
- Evaluating a single pipeline
- Considering different options for how to use a test set
- Use a LinearResidualizer to residualize input brain data
- Introduce and use the Evaluate input option

----------------------



- Binary_Sex.ipynb makes use of data from the ABCD Study to explore a simple workflow example. In this notebook, sex at birth is predicted from structural MRI derived regions of interest. This example relative to some of the others contains a great deal more explanation / tutoria text, making it suitable for those new to the library as a starting place.

- Regression_SSRT.ipynb makes use of data from the ABCD Study to explore a regression type analysis. This notebook uses functional mri contrasts from a Stop Signal task to predict a derived measure of stop signal reaction time. This notebook is only lightly commented, so it is best used a reference rather than tutorial. This notebook includes as a bonus some code from the library Neuro_Plotting (https://github.com/sahahn/Neuro_Plotting) on how to easily make a collage of feature importances as projected onto the surface of a brain. Also how to make a gif of subcortical regions.

- Regression_Waist_Circumference.ipynb makes uses of data from the ABCD Study to explore a regression type analysis. This notebook uses DTI derived data to predict waist circumference. THis notebook is only lightly commented, but includes a fairly extensive model selection and testing step. This notebook should likely be employed as a reference for different modelling options to try.