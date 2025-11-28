# Configuration file for the Sphinx documentation builder.

import os
import sys

# -- Path setup --------------------------------------------------------------

# Add the project root directory so autodoc can import the package
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'FLAG-X'
copyright = '2025'
author = 'Paul Martini'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",          # Autodoc for Python modules
    "sphinx.ext.autosummary",      # Auto-generate .rst stubs
    "sphinx.ext.napoleon",         # Google-style docstrings
]

# Show __init__ docstring of class as well
autoclass_content = "both"

# Order as in code
autodoc_member_order = 'bysource'

# Choose HTML theme
html_theme = "sphinx_rtd_theme"

# Napoleon settings
napoleon_google_docstring = True

autodoc_mock_imports = [
    "numpy",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "numba",
    "umap-learn",
    "umap",
    "anndata",
    "scanpy",
    "igraph",
    "leidenalg",
    "fcsparser",
    "readfcs",
    "pytometry",
    "tqdm",
    "requests",
    "somoclu",
    "click",
    "torch",
    "flowio",
    "flowutils",
    "typing_extensions",
    "sklearn",
    "scipy",
]

