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


# -- Project information -----------------------------------------------------

project = 'DTW'
copyright = '2021, Christophe Godin'
author = 'Christophe Godin'

# The full version, including alpha/beta/rc tags
release = '0.0.1'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # 'myst_parser',  # MyST markdown parser (can be activated by 'myst_nb')
    'myst_nb',  # MyST jupyter notebooks parser
    'sphinx.ext.autodoc',  # Include documentation from docstrings
    'sphinx.ext.doctest',  # Test snippets in the documentation
    'sphinx.ext.intersphinx',  # Link to other projects’ documentation
    'sphinx.ext.napoleon',  # Support for NumPy and Google style docstrings
    'sphinx.ext.todo',  # Support for todo items
    'sphinx.ext.viewcode',  # Add links to highlighted source code
    'sphinxarg.ext',  # To include tables describing command-line arguments for executable scripts
]

# Optional MyST syntax:
# https://myst-parser.readthedocs.io/en/latest/using/syntax-optional.html#optional-myst-syntaxes
myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "colon_fence",
    "deflist"
]

# sphinx.ext.todo settings:
todo_include_todos = True

# sphinx.ext.napoleon settings:
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_keyword = True
napoleon_use_param = True
napoleon_use_rtype = False

# sphinx.ext.viewcode settings:
viewcode_follow_imported_members = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.md'

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'colorful'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Import Notebooks --------------------------------------------------------
import os

for file in os.listdir('.'):
    if '.ipynb' in file:
        os.remove(file)

nb_path = '../../notebooks'
notebooks = [nb for nb in os.listdir(nb_path) if nb.endswith('.ipynb') and nb.startswith("tutorial")]

for nbf in notebooks:
    print(f'importing notebook file: {nbf}')
    os.system(f'cp {nb_path}/{nbf} tutorials/{nbf}')


# -- Create tutorial page ----------------------------------------------------
tuto_header = """
% Update "fixed" table of contents on the left quick browse panel
```{eval-rst}
.. toctree::
  :maxdepth: 2

"""

with open("tutorials.md", "w+") as f:
    f.write("# Tutorials\n")
    # f.write(tuto_header)
    # for i in notebooks:
    #     f.write(f"   tutorials/{i}" + "\n")
    # f.write("```" + "\n")
    f.write("\n")
    f.write("Here is a list of tutorials built from the available notebooks:")
    f.write("\n")
    for i in notebooks:
        ref = i.replace("tutorial-", "")
        ref = ref.replace(".ipynb", "")
        ref = ref.replace('_', ' ')
        f.write(f" - [{ref.capitalize()}](tutorials/{i})" + "\n")

# Add this to render plotly figures:
html_js_files = ["https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = ['custom.css']
