# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

#import importlib.metadata as importlib_metadata
# from importlib.metadata import version

# -- Project information -----------------------------------------------------

project = 'wfi-reference-file-pipeline'
copyright = '2021, STScI'
author = 'Tyler Desjardins'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.imgmath',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx_automodapi.automodapi'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

html_logo = '_static/stsci_pri_combo_mark_white.png'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['wfi_reference_pipeline']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
pygments_style = 'sphinx'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = 'py:obj'

# The full version, including alpha/beta/rc tags.
# The full version, including alpha/beta/rc tags.
#release = importlib_metadata.version(project)
#release = version(project)
# The short X.Y version.
#ver = '.'.join(release.split('.')[:2])
