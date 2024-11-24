# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))  

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Melanoma binary segmentation'
copyright = '2024, Ishchenko Roman'
author = 'Ishchenko Roman'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [  
    'sphinx.ext.autodoc',  
    'sphinx.ext.napoleon',  
    'sphinx_autodoc_typehints',  
    # 'sphinx.ext.pngmath',
]

templates_path = ['_templates']
exclude_patterns = [
    '_build', 
    'Thumbs.db',
    '.DS_Store',
    '__pycache__',
    # '*.sh',
    # '*Docker',
    # '*txt',
    # "*.md",
    # "*.png",
    # "*.bmp",
]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'

templates_path = ['_templates']  
exclude_patterns = []  

html_theme = 'sphinx_rtd_theme'  
html_static_path = ['_static'] 