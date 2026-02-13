# Configuration file for the Sphinx documentation builder.

import sys
from pathlib import Path

sys.path.insert(0, Path("../../src").resolve().as_posix())

try:
    from importlib.metadata import version as get_version  # python 3.8+
except ImportError:
    from importlib_metadata import version as get_version

try:
    release = get_version("legend_data_monitor")
except Exception:
    release = "0.0.0"

version = release

project = "legend-data-monitor"
copyright = "Copyright Holder"

extensions = [
    "sphinx.ext.githubpages",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
language = "python"

# Furo theme
html_theme = "furo"
html_theme_options = {
    "source_repository": "https://github.com/legend-exp/legend-data-monitor",
    "source_branch": "main",
    "source_directory": "docs/source",
}
html_title = f"{project} {version}"

# list here legend-optics dependencies that are not required for building docs and
# could be unmet at build time
autodoc_mock_imports = [
    # add new packages here
]
autodoc_default_options = {"ignore-module-all": True}

# sphinx-napoleon
# enforce consistent usage of NumPy-style docstrings
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_ivar = True

# intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("http://docs.scipy.org/doc/numpy", None),
    "scipy": ("http://docs.scipy.org/doc/scipy/reference", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "matplotlib": ("http://matplotlib.org/stable", None),
}  # add new intersphinx mappings here

# sphinx-autodoc
# Include __init__() docstring in class docstring
autoclass_content = "both"
autodoc_typehints = "both"
autodoc_typehints_description_target = "documented_params"
autodoc_typehints_format = "short"
