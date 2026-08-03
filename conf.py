# -*- coding: utf-8 -*-
"""Sphinx documentation configuration for retinoto_py.

Sources live at the repo root — Markdown files and Jupyter notebooks are all
built as first-class doc pages in a flat layout, with nothing nested under a
sub-directory. ReadTheDocs picks up this config via '.readthedocs.yaml'.
"""


# -- Project information ------------------------------------------------------
project = "retinoto_py"
copyright = "2025, Laurent U Perrinet"
author = "Laurent U Perrinet"

version_raw = 'unknown'
try:
    from importlib.metadata import version as pkg_version  # Python 3.8+
    version_raw = pkg_version("retinoto_py")
except Exception:
    pass

# RTD passes `_RTD_VERSION` to avoid caching stale values across branches.
import os
version = os.environ.get("_RTD_VERSION", version_raw)


# -- General Sphinx settings --------------------------------------------------
extensions = [
    "myst_parser",          # Markdown (and Markdown-like notebooks)
    "nbsphinx",             # Jupyter notebook execution + output embedding
]

exclude_patterns = [
    "_build",               # build outputs
    ".venv/",
    "cached_data/*",
    "!docs/**",             # leave any legacy docs/ tree out if it exists
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}


# Source files are at repo root. Notebooks (.ipynb) are added by nbsphinx.
myst_enable_extensions = [
    "attrs_inline",
    "colon_fence",
    "html_admonition",
]

rst_epilog = """
.. |VERSION| replace:: {version}
""".format(version=version or 'unknown')


# -- HTML output ------------------------------------------------------------------
templates_path = []  # no custom templates at root level

output_dir = "_build/html"

html_theme = "furo"  # clean monospace theme suited to API-style docs
html_static_path = []         # leave empty unless you need shared CSS/images
html_title = (
    f"retinoto_py | Foveated Retinotopy Documentation"
)
# html_short_title  = 'retinoto_py'

# Add GitHub / RTD links
html_context = {
    "docs_readthedocs_url": "https://retinoto-py.readthedocs.io/",
}
# html_favicon = ''            # optional, default Sphinx favicon is fine


# -- nbsphinx rendering -------------------------------------------------------
nbsphinx_execute = 'never'   # only run on RTD / local; skip during CI
# Set cache dir explicitly so the build isn't sensitive to cwd location (e.g. in GitHub Actions)
import sys, os  # noqa: E402 — top-level import above can still raise ImportError if Python is too old
nbsphinx_execute_arguments = [
    '--InlineMathPlugin.enable',
]


# -- intersphinx / Napoleon ---------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

# Napoleon auto-links numpy-style docstrings (useful for API pages added later)
napoleon_google_docstring = False
napoleon_numpy_docstring = True


# -- nbsphinx notebook processing ----------------------------------------------
# Don't run notebooks on every Sphinx invocation — only let them render.
# Useful when a notebook has heavy dependencies (e.g., torch/imagenet data).
nbsphinx_execute_kernel_timeout = -1  # infinite
nbsphinx_allow_errors = True           # don't fail doc build if cells error
