# Dynamic time warping algorithm(s)


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


## Publish master branch to ROMI GitHub

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

> :warning: For macOS, follow these [instructions](https://docs.conda.io/projects/conda-build/en/latest/resources/compiler-tools.html#macos-sdk) to install the required `macOS 10.9 SDK`.


### Build a conda package
Using the given recipes in `dtw/conda/recipe`, it is easy to build the conda package:
```bash
cd dtw/conda/recipe
conda build .
```
> :warning: This should be done from the `base` environment!


### Conda useful commands

#### Purge built packages:
```bash
conda build purge
```

#### Clean cache & unused packages:
```bash
conda clean --all
```
