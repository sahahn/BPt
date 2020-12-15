from setuptools import setup, find_packages

setup(name='brain-pred-toolbox',
      version='1.3.5',
      description='The Brain Predictability toolbox (BPt) is a ' +
      'Python based machine learning library designed to work with ' +
      'a range of neuroimaging data.',
      url='http://github.com/sahahn/BPt',
      author='Sage Hahn',
      author_email='sahahn@euvm.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'scikit-learn>=0.23.1',
          'numpy>=1.18',
          'scipy>=1.2',
          'pandas>=1.1.5',
          'matplotlib>=3.2.2',
          'seaborn>=0.9',
          'deslib',
          'scikit-image>=0.16',
          'tqdm>=4.51',
          'nevergrad==0.4.2.post5',
          'Ipython',
          'joblib>=0.14',
          'loky',
      ],
      test_suite='nose.collector',
      tests_require=['nose', 'coverage'],
      zip_safe=False)
