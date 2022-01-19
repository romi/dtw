# Install

## Installing the conda package

:::{warning}
Not yet available!
:::


## Installing the module from sources

### Clone the sources
Start by cloning the sources with:
```bash
git clone https://gitlab.inria.fr/cgodin-dev/dtw.git
```

### Conda environment creation (optional)
If you don't have an existing conda environment, create one (named `dtw`) with:
```bash
conda env create -f environment.yml
```
You can find the official instructions on how to manually create an environment [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands).

### Install DTW library
Start by activating your (`dtw`) environment with:
```bash
conda activate dtw
```
Then install the package in develop mode with `pip`,
```bash
(dtw)> python -m pip install -e .
```

Note: `-e` is used here to install the code in "develop mode", this way you do not have to re_install the package every time you makes modifications to he sources.


## Testing the code
To run an example file, go in `data-analysis` directory and launch `ipython`.
```bash
cd data-analysis
ipython
```

Then you can run the `arabido-test.py` test file with:
```bash
ipython> %run arabido-test.py
```
