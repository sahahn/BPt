from setuptools import setup, find_packages

setup(name='ABCD_ML',
      version='1.0',
      description='Python based Machine Learning library, for tabular ' +
                  'Neuroimaging data, specifically geared towards the' +
                  ' ABCD dataset.',
      url='http://github.com/sahahn/ABCD_ML',
      author='Sage Hahn',
      author_email='sahahn@euvm.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'scikit-learn>=0.21',
          'xgboost',
          'numpy>=1.16',
          'scipy>=1.2',
          'pandas>=0.24',
          'matplotlib>=3',
          'seaborn>=0.9',
          'deslib',
          'imbalanced-learn',
          'shap',
          'scikit-image>=0.15',
          'tqdm',
          'dill',
          'nevergrad',
          'Ipython',
          'joblib',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
