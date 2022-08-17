<p align="center">
  <img src="https://github.com/MICS-Lab/scyan/raw/master/docs/assets/logo.png" alt="scyan_logo" width="500"/>
</p>

[![PyPI](https://img.shields.io/pypi/v/scyan.svg)](https://pypi.org/project/scyan)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://mics-lab.github.io/scyan/)
![Build](https://github.com/MICS-Lab/scyan/workflows/ci/badge.svg)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![Downloads](https://pepy.tech/badge/scyan)](https://pepy.tech/project/scyan)
[![License](https://img.shields.io/pypi/l/scyan.svg)](https://github.com/MICS-Lab/scyan/blob/master/LICENSE)
[![Imports: isort](https://img.shields.io/badge/imports-isort-blueviolet)](https://pycqa.github.io/isort/)
[![DOI](https://zenodo.org/badge/516048412.svg)](https://zenodo.org/badge/latestdoi/516048412)

Scyan stands for **S**ingle-cell **Cy**tometry **A**nnotation **N**etwork. Based on biological knowledge prior, it provides a fast cell population annotation without requiring any training label. Scyan is an interpretable model that also corrects batch-effect and can be used for debarcoding, cell sampling, and population discovery.

# Documentation

The [complete documentation can be found here](https://mics-lab.github.io/scyan/). It contains installation guidelines, tutorials, a description of the API, etc.

# Overview

Scyan is a Bayesian probabilistic model composed of a deep invertible neural network called a normalizing flow (the function $f_{\phi}$). It maps a latent distribution of cell expressions into the empirical distribution of cell expressions. This cell distribution is a mixture of gaussian-like distributions representing the sum of a cell-specific and a population-specific term. Also, interpretability and batch effect correction are based on the model latent space — more details in the article's Methods section.

<p align="center">
  <img src="https://github.com/MICS-Lab/scyan/raw/master/docs/assets/overview.png" alt="overview_image"/>
</p>

# Getting started

Scyan can be installed on every OS with `pip` or [`poetry`](https://python-poetry.org/docs/).

On macOS / Linux, `python>=3.8,<3.11` is required, while `python>=3.8,<3.10` is required on Windows. The preferred Python version is `3.9`.

## Install from PyPI (recommended)

```bash
pip install scyan
```

## Install locally (if you want to contribute)

> Advice (optional): We advise creating a new environment via a package manager (except if you use Poetry, which will automatically create the environment). For instance, you can create a new `conda` environment:
>
> ```bash
> conda create --name scyan python=3.9
> conda activate scyan
> ```

Clone the repository and move to its root:

```bash
git clone https://github.com/MICS-Lab/scyan.git
cd scyan
```

Choose one of the following, depending on your needs (it should take at most a few minutes):

```bash
pip install .                           # pip minimal installation (library only)
pip install -e '.[dev,docs,discovery]'  # pip installation in editable mode
poetry install -E 'dev docs discovery'  # poetry installation in editable mode
```

## Basic usage / Demo

```py
import scyan

adata, marker_pop_matrix = scyan.data.load("aml")

model = scyan.Scyan(adata, marker_pop_matrix)
model.fit()
model.predict()
```

This code should run in approximately 40 seconds (once the dataset is loaded).
For more usage demo, read the [tutorials](https://mics-lab.github.io/scyan/tutorials/usage/) or the complete [documentation](https://mics-lab.github.io/scyan/).

# Technical description

Scyan is a **Python** library based on:

- _Pytorch_, a deep learning framework
- _AnnData_, a data library that works nicely with single-cell data
- _Pytorch Lighning_, for model training
- _Hydra_, for project configuration (optional)
- _Weight & Biases_, for model monitoring (optional)

# Project layout

    .github/      # Github CI and templates
    config/       # Hydra configuration folder (optional use)
    data/         # Data folder containing adata files and csv tables
    docs/         # The folder used to build the documentation
    scripts/      # Scripts to reproduce the results from the article
    tests/        # Folder containing tests
    scyan/                    # Library source code
        data/                 # Folder with data-related functions and classes
            datasets.py       # Load and save datasets
            tensors.py        # Pytorch data-related classes for training
        module/               # Folder containing neural network modules
            coupling_layer.py # Coupling layer
            distribution.py   # Prior distribution (called U in the article)
            real_nvp.py       # Normalizing Flow
            scyan_module      # Core module
        plot/                 # Plotting tools
            ...
        tools/
            ...               # Preprocessing tools and more
        mmd.py                # Maximum Mean Discrepancy implementation
        model.py              # Scyan model class
        utils.py              # Misc functions
    .gitattributes
    .gitignore
    CONTRIBUTING.md   # To read before contributing
    LICENSE
    mkdocs.yml        # The docs configuration file
    poetry.lock
    pyproject.toml    # Dependencies, project metadata, and more
    README.md
    setup.py          # Setup file, see `pyproject.toml`
