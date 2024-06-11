from __future__ import annotations

import os
import sys
import importlib.metadata

# Add the current directory to the system path
sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'DAS4Whales'
copyright = "2024, Bouffaut & Goestchel"
author = 'Léa Bouffaut, Quentin Goestchel'
# version = release = importlib.metadata.version("package")

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

templates_path = ['_templates']
source_suffix = [".rst", ".md"]
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo' #'sphinx_rtd_theme'
myst_enable_extensions = [
    "colon_fence",
]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
]

always_document_param_types = True

# -- Extension configuration -------------------------------------------------

# Add any custom extension configuration here...
