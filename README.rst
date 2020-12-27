========================
dtw
========================

.. {# pkglts, doc

.. #}

Dynamic time warping algorithm(s)

. Installing the module

setup the right conda env:

> conda activate my_env

Then,

my_env> python setup.py develop --prefix=$CONDA_PREFIX
or
ipython> run setup.py develop --prefix=$CONDA_PREFIX

Note: 'develop' is used here to use the code that is being developed rather than a python package.

. Running the code

To run an example file go in data-analysis dir and launch ipython. Then,

ipython> run arabido-test.py

. Editing the code and testing
If one modifies the code (example by modifying the dtw.py file), one should reinstall the modified files in python to take modifications into account.
This is done using:

my_env> python setup.py install

Create an env.yaml file to set conda dependencies 
