# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

master_doc = 'index'

# -- Project information -----------------------------------------------------

project = 'BPt'
copyright = '2020, sahahn'
author = 'sahahn'

# The full version, including alpha/beta/rc tags
release = '2.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.napoleon', 'sphinx.ext.intersphinx']


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


master_doc = "index"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'asteroid_sphinx_theme'
html_theme_path = ["_themes", ]


html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
    'logo': 'logo/logo.png'
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []
logo = 'logo/logo.png'
html_logo = 'logo/logo.png'
autoclass_content = 'both'

intersphinx_mapping =\
    {'deslib': ('http://deslib.readthedocs.io/en/latest', None),
     'sklearn': ('http://scikit-learn.org/stable', None),
     'lightgbm': ('https://lightgbm.readthedocs.io/en/latest', None),
     'xgboost': ('https://xgboost.readthedocs.io/en/latest', None),
     'imblearn': ('https://imbalanced-learn.readthedocs.io/en/stable', None),
     'category_encoders': ('http://contrib.scikit-learn.org/category_encoders/', None)     
     }
