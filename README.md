# Reor

[![Build Status](https://travis-ci.com/azedarach/reor.svg?branch=master)](https://travis-ci.com/azedarach/reor)

Experiments with different methods for
constructing reduced order models.

## Installation

### Requirements

The following packages are required:

  - Python (>= 3.6)
  - NumPy
  - SciPy
  - scikit-learn
  - cvxopt

To build from source, the following additional packages
are required:

  - setuptools

Additionally, some of the examples require Matplotlib (>= 3.0.3)
to be installed.

### Installation

To install from source, run:

    python setup.py install

For example, if installing into a custom conda environment, first create
the environment via

    conda create -n reor-env python=3.6
    conda activate reor-env

The package may then be installed using

    cd /path/to/reor/directory
    python setup.py install

Optionally, a set of unit tests may be run by executing

    python setup.py test
