from setuptools import setup, find_packages

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='brain-pred-toolbox',
      long_description=long_description,
      long_description_content_type='text/markdown',
      version='2.1.0',
      description='The Brain Predictability toolbox (BPt) is a ' +
      'Python based machine learning library designed to work with ' +
      'a range of neuroimaging data.',
      url='http://github.com/sahahn/BPt',
      author='Sage Hahn',
      author_email='sahahn@euvm.edu',
      license='MIT',
      packages=find_packages(),
      python_requires=">=3.7",
      install_requires=[
          'scikit-learn>=0.23.1',
          'numpy==1.19.5',
          'scipy>=1.2',
          'pandas>=1.1.5',
          'matplotlib>=3.2.2',
          'seaborn>=0.9',
          'scikit-image>=0.16',
          'tqdm>=4.51',
          'nevergrad==0.4.3.post4',
          'Ipython',
          'joblib>=1',
          'loky',
      ],
      extras_require={
          'extra': ['lightgbm>3',
                    'sweetviz>2',
                    'nilearn>=0.7',
                    'python-docx',
                    'mvlearn',
                    'imblearn',
                    'networkx',
                    'xgboost>=1.3'],
       },
      test_suite='pytest',
      tests_require=['pytest', 'coverage'],
      zip_safe=False)
