# Dynamic time warping algorithm(s)

## Installing the module
Setup the right conda env:
```bash
conda activate my_env
```

Then,
```bash
my_env> python setup.py develop
```
or
```bash
ipython> run setup.py develop
```

Note: 'develop' is used here to use the code that is being developed rather than a python package.

## Running the code
To run an example file go in data-analysis dir and launch ipython. Then,
```bash
ipython> run arabido-test.py
```

## Editing the code and testing
If one modifies the code (example by modifying the dtw.py file), one should reinstall the modified files in python to take modifications into account.

This is done using:
```bash
my_env> python setup.py install
```
Create an `env.yaml` file to set conda dependencies.

## Publish master branch to GitHub
0. Create a GitHub empty repo
1. Clone this repository and go to the directory
2. Add a 'romi' remote pointing to the empty GitHub repo: ``git remote add romi https://github.com/romi/dtw.git``
3. Push your modifications to GitHub (from the repository root): ``git push romi master``
