from setuptools import setup

setup(name='ABCD_ML',
      version='0.1',
      description='Python based Machine Learning library, for tabular ' +
                  'Neuroimaging data, specifically geared towards the' +
                  ' ABCD dataset.',
      url='http://github.com/sahahn/ABCD_ML',
      author='Sage Hahn',
      author_email='sahahn@euvm.edu',
      license='MIT',
      packages=['ABCD_ML'],
      install_requires=[
          'scikit-learn>=0.21',
          'lightgbm>=2',
          'xgboost',
          'numpy>=1.16',
          'scipy>=1.2',
          'pandas>=0.24',
          'matplotlib>=3',
          'seaborn',
          'deslib',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
