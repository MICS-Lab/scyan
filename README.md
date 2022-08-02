<p align="center">
  <img src="https://github.com/MICS-Lab/scyan/raw/master/docs/assets/logo.png" alt="scyan_logo" width="500"/>
</p>

Scyan stands for **S**ingle-cell **Cy**tometry **A**nnotation **N**etwork. Based on biological knowledge prior, it provides a fast cell population annotation without requiring any training label. Scyan is an interpretable model that also corrects batch-effect and can be used for debarcoding / cell sampling / population discovery.

# Documentation

The [complete documentation can be found here](https://mics_biomathematics.pages.centralesupelec.fr/biomaths/scyan/). It contains installation guidelines, tutorials, a description of the API, etc.

# Overview

Scyan is a Bayesian probabilistic model composed of a deep invertible neural network called a normalizing flow (the function $f_{\phi}$). It maps a latent distribution of cell expressions into the empirical distribution of cell expressions. This cell distribution is a mixture of gaussian-like distributions representing the sum of a cell-specific and a population-specific term. Also, interpretability and batch effect correction are based on the model latent space — more details in the article's Methods section.

<p align="center">
  <img src="https://github.com/MICS-Lab/scyan/raw/master/docs/assets/overview.png" alt="overview_image"/>
</p>

# Getting started

Scyan can be installed on every OS with `pip` or [`poetry`](https://python-poetry.org/docs/).

On MacOS / Linux, `python>=3.8,<3.11` is required, while `python>=3.8,<3.10` is required on Windows. The preferred Python version is `3.9`.

## Install with PyPI

Available after publication

## Install locally

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

## Basic usage

```py
import scyan

adata, marker_pop_matrix = scyan.data.load("aml")

model = scyan.Scyan(adata, marker_pop_matrix)
model.fit()
model.predict()
```

This code should run in approximately 40 seconds (once the dataset was loaded).
For more usage demo, read the [tutorials](https://mics_biomathematics.pages.centralesupelec.fr/biomaths/scyan/tutorials/usage/) or the complete [documentation](https://mics_biomathematics.pages.centralesupelec.fr/biomaths/scyan/).

# Technical description

Scyan is a **Python** library based on:

- _Pytorch_, a deep learning framework
- _AnnData_, a data library that works nicely with single-cell data
- _Pytorch Lighning_, for model training
- _Hydra_, for project configuration (optional)
- _Weight & Biases_, for model monitoring (optional)

# Project layout

    config/       # Hydra configuration folder (optional use)
    data/         # Data folder containing adata files and csv tables
    docs/         # The folder used to build the documentation
    scripts/      # Scripts to reproduce the results from the article
    tests/        # Folder containing tests
    scyan/                    # Library source code
        data/                 # Folder with data-related functions and classes
            datasets.py       # Load and save datasets
            tensors.py        # Pytorch data related classes for training
        module/               # Folder containing neural network modules
            coupling_layer.py # Coupling layer
            distribution.py   # Prior distribution (called U in the article)
            real_nvp.py       # Normalizing Flow
            scyan_module      # Core module
        plot/                 # Plotting tools
            ...
        mmd.py                # Maximum Mean Discrepancy implementation
        model.py              # Scyan model class
        preprocess.py         # Preprocessing functions
        utils.py              # Misc functions
    .gitattributes
    .gitignore
    .gitlab-ci.yml    # CI that builds documentation
    CONTRIBUTING.md   # To read before contributing
    LICENSE
    mkdocs.yml        # The docs configuration file
    poetry.lock
    pyproject.toml    # Dependencies, project metadata, and more
    README.md
    setup.py          # Setup file, see `pyproject.toml`
