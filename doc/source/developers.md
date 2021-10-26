# Developers instructions

## Developer dependencies
Make sure to install the dependencies from ``conda/env/dtw_dev.yaml``.
This can be done, from the root `dtw` directory, with:
```bash
conda update --file conda/env/dtw_dev.yaml
```

:::{note}
Do that from within your (active) environment, or specify its name with the `--name` option , like `--name dtw`. 
:::


## Notebooks
Creates jupyter notebooks to create examples and tutorials.

### Starting the notebook server
Once you have installed the [developer dependencies](#developer-dependencies), from the ``notebooks/`` directory, you can start the notebook server with:
```bash
jupyter notebook
```

### Conventions
To automatically add the notebooks to the documentation, follows this convention for file names:
  - tutorials notebooks starts with the `tutorial-` prefix
  - examples notebooks starts with the `example-` prefix

They will be automatically copied to `doc/tutorials/` and referenced.

### Clean-up before commits
To automatically clean the notebooks from their output (to avoid unnecessary commit of useless changes), we add a pre-commit action.
From the `dtw` root directory, add the notebook clean-up action:
```bash
echo "#!/bin/sh" > .git/hooks/pre-commit
echo "jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebooks/*.ipynb" >> .git/hooks/pre-commit
```
:::{warning}
The first line will erase any existing `pre-commit` file!
Only execute the second one if you already specified some actions!
:::


## Website & documentation

### Styles & guidelines
We use [Sphinx](https://www.sphinx-doc.org/en/master/index.html), and [MyST](https://myst-parser.readthedocs.io/en/latest/index.html) to generate this documentation. 
To write the **docstrings** we follow the [NumPy conventions](https://numpydoc.readthedocs.io/en/latest/format.html).

### Local build
Once you have installed the [developer dependencies](#developer-dependencies), from the `doc/` directory, you can build the documentation with:
```bash
make html
```

:::{note}
This will:
    - use sphinx to generates the rst files
    - copy the notebooks in the `notebooks/` directory (starting with `tutorial-`)
    - reference them in the `doc/source/tutorials.md` file (so do not edit it!)
    - execute them with `myst-nb` extension (create the outputs)
    - convert everything to markdown
:::

You will then obtain a `build/` directory (under `doc/`) with an `index.html` file to open with your browser!


## Publish a clone to ROMI GitHub

### The first time:
0. Create a GitHub empty repo
1. Clone this repository and go to the directory
2. Add a 'romi' remote pointing to the empty GitHub repo: `git remote add romi https://github.com/romi/dtw.git`

### To update GitHub master from Inria GitLab
3. Push your modifications to GitHub (from the repository root): `git push romi master`


## Conda packaging

### Requirement
Install `conda-build`, in the `base` environment, to be able to build conda packages:
```bash
conda deactivate
conda install conda-build
```

:::{warning}
For macOS, follow these [instructions](https://docs.conda.io/projects/conda-build/en/latest/resources/compiler-tools.html#macos-sdk) to install the required `macOS 10.9 SDK`.
:::

### Build a conda package
Using the given recipes in `dtw/conda/recipe`, it is easy to build the conda package:
```bash
cd dtw/conda/recipe
conda build .
```

:::{warning}
This should be done from the `base` environment!
:::

### Conda useful commands

#### Purge built packages:
```bash
conda build purge
```

#### Clean cache & unused packages:
```bash
conda clean --all
```
