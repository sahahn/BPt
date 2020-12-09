In Dev
************

- New Dataset class
    - The new Dataset class is designed to eventually replace the old system for data loading.

Release 1.3.4
***************

- Added support for pandas >= 1
    - Previously didn't support latest pandas.

- Added initial support for in-place FIs 
    - Moving from plotting via ML to plotting from the Feature Importance object itself
    - Only fully supports global right now.

Release 1.3.3
***************

- Fixed bug with problem type
    - There was an error which was mistakenly setting categorical problem type instead of regression.

- Fixed internal bug with mapping
    - Effected Transformer.

- Added base_dtype option
    - Evaluate and Test now have base dtype options, which allow changing dtype of data
    - Changed default data dtype from float64 to float32, should provide general speed ups


Release 1.3.1 and 1.3.2
************************

- New AutoGluon option
    - Can now specify the auto machine learning package AutoGluon as a :class:`BPt.Model`

- New SurfMaps extension loader
    - Added new extension Loader :class:`BPt.SurfMaps`

- only_fold parameter
    - New optional parameter in Evaluate for running only one fold.

- Better support for scikit-learns VotingClassifier and VotingRegressor
    - Similar to Stacking update from 1.3, but for voting classifier + regressor.

- More support for nested pipelines
    - Can now have nested pipelines propagate their parameter distributions to a parameter search in the top level pipeline.

- Bug Fix with CV
    - Fixed rare bug with CV expecting pandas Series, added support for passing numpy array.

Release 1.3
************

- Support for nested parameter searches
    - :class:`BPt.Model` and :class:`BPt.Ensemble` now support a param_search parameter.
    - The parameter param_search accepts a :class:`BPt.Param_Search` object, and turns the model or ensemble into a nested search object.

- Initial support for passing nested :class:`BPt.Model_Pipeline`
    - Now can pass nested :class:`BPt.Model_Pipeline` if wrapped in a :class:`BPt.Model`
    - Warning: there are still cases which will not work.

- Better support for stacking ensembles
    - Stacking ensembles are ported from scikit-learn's StackingClassifier and StackingRegressor.
    - The :class:`Ensemble` object can now support the arguments base_model and cv_splits.
    - The parameter, base_model allows passing in BPt compatible models to act as the final_estimator in stacking.
    - cv_splits allows passing a new input class :class:`BPt.CV_Splits` which in the context of stacking, allows for custom CV behavior to train the base estimators.

- Add experimental auto data type to loading targets
    - You can now pass 'a' or 'auto' when loading targets to the data_type parameter to specify that the data type should be automatically inferred.

- Change input parameter CV to cv
    - In order to be more compatible with other libraries and intuative, now CV always refers to classes and cv an input parameter.

- New Loky multi-processing support
    - Changed to the new default mp_context.
    - Loky is a python library https://pypi.org/project/loky/ with better multiprocessing support than python's default.

- New Dask multi-processing support
    - Experimental support for dask multiprocessing

- Fixed how n_jobs propegates in complex model pipelines
    - New parameter in :class:`BPt.Ensemble` n_jobs_type, which allows more controls over how n_jobs are spread out in the context of Ensembles.

- Fixed bug with RandomParcels
    - The RandomParcels object can be imported through from BPt.extensions import RandomParcels
    - A previous bug would allow some vertex labelled as medial wall, to be mislabeled, this has been fixed.
    
- Add view to :class:`BPt.Model`
    - Initial support for an experimental `view` method for the :class:`BPt.Model` class.

- Improve the outputted results from Evaluate and Test
    - Default feature importance to calculate is now None.
    - Added more optional parameters here.
    - Added new returned single metric.
    - Optional parameter for returning the trained model(s).

- Add default case for :class:`BPt.Problem_Spec`
    - Now with default detecting of problem type, can optionally not specify a problem spec in Evaluate or Test.

- Add default problem type
    - Now if no target_type is specified, a default type will be set based on the type of the loaded target.

- New default scorers
    - The default scorers have changed, now provides multiple scorers for each type by default

- Speed up working with Data Files
    - Some improved performance in loading Data Files

- Seperate caching for transformers and loaders
    - Loaders and Transformers can now be cached via a cache_loc parameter.

- Added experimental support for target transformation
    - In some cases it is useful to allow nested transformations to the target variable.
    - :class:`BPt.Model` and :class:`BPt.Ensemble` now support an experimental argument for specifying a target transformation.

- Introduce new :class:`BPt.Values_Subset`
    - In addition, added better description of `subjects` as a parameter type, with more universal behavior.

- Large amounts of internal refactoring
    - From docstrings, to structure of code, big amounts of re-factoring.

- Name change from ABCD_ML to BPt
    - Along with this change, the import of the ML object changed.

- New support for k bins encoding when loading targets
    - When loading targets, you may now specify a k-bins encoding scheme directly.

- Renamed metric to scorer
    - The argument metric has been renamed to scorer
    - The scorers accepted have also been re-defined to more closely align with scikit-learn's scorers.

- Added support for categorical encoders and the categorical encoder library
    - The new encouraged way to perform categorical encoding is by specifying transformers, via added options from the categorical encoders library.

- New, now all parameter objects can accept scope as an argument
    - In previous versions, input objects differed in which could accept a `scope` argument, now all can.

- New ML verbosity options
    - Some new ML verbosity options

- Support latest scikit-learn version
    - Backend changes allowing full compat. with latest scikit-learn versions.

- Add more print information
    - In an effort to make more of the library behavior transparent, more verbose print info has been added by default.

- Removed ML class eval and test scores
    - Depreciated the class wide eval and test scores previously stored in ML object