<p align="center">
  <img src="./docs/assets/logo.png" alt="scyan_logo" width="500"/>
</p>

Scyan stands for **S**ingle-cell **Cy**tometry **A**nnotation **N**etwork. Based on a biological knowledge prior, it provides a fast cell populations annotation without requiring any training label. Scyan is an interpretable model that also corrects batch-effect and can be used for debarcoding / cell sampling / population discovery.

# Documentation

The [complete documentation can be found here.](https://2017blampeyq.pages.centralesupelec.fr/scyan/) It contains installation guidelines, tutorials, a description of the API, ...

# Overview

Scyan is a Bayesian probabilistic model composed of a deep invertible neural network called a Normalizing Flow (see function `f`). This network maps a latent distribution of cell expressions into the empirical distribution of cell expressions. The latter latent cell distribution is a mixture of gaussian-like distributions representing the sum of a cell-specific term and a population-specific term. The latent space is used for interpretability and batch effect correction. More details on the methods section of the article.

<p align="center">
  <img src="./docs/assets/overview.png" alt="overview_image"/>
</p>

# Getting started

## Install with PyPI

Soon available

## Install locally

> Advice (optional): if using `pip`, we advise to create a new environment via a package manager. For instance, you can create a new conda environment:
>
> ```bash
> conda create --name scyan python=3.9
> conda activate scyan
> ```

Clone the repository and move to its root

```bash
git clone git@gitlab-research.centralesupelec.fr:mics_biomathematics/biomaths/scyan.git
cd scyan
```

You can install it with `pip` or `poetry`, choose one among the following:

```bash
pip install -e '.[dev,docs]'  # pip installation in development mode
pip install .                 # pip basic installation (package only)
poetry install -E 'dev docs'  # poetry installation in development mode
```

## Basic usage

```py
import scyan

adata, marker_pop_matrix = scyan.data.load("aml")

model = scyan.Scyan(adata, marker_pop_matrix)
model.fit()
model.predict()
```

For more details, read the [documentation](https://2017blampeyq.pages.centralesupelec.fr/scyan/).

# Technical description

Scyan is a **Python** library based on:

- _Pytorch_, a deep learning framework
- _AnnData_, a data library that works nicely with nice single-cell data
- _Pytorch Lighning_ for model training
- _Hydra_ for project configuration
- _Weight & Biases_ for model monitoring

# Project layout

    config/       # Hydra configuration folder (optional use)
    data/         # Data folder containg adata files and csv tables
    docs/         # The documentation folder
    scripts/      # Scripts to reproduce the results from the article
    tests/        # Testing the library
    scyan/                    # Library source code
        module/               # Folder containing neural network modules
            coupling_layer.py # Coupling layer
            distribution.py   # Prior distribution (called U in the article)
            real_nvp.py       # Normalizing Flow
            scyan_module      # Core module
        plot/                 # Plotting tools
            ...
        data/                 # Folder with data related functions and classes
            ...
        mmd.py                # Maximum Mean Discrepancy implementation
        model.py              # Scyan model class
        utils.py              # Misc functions
    .gitignore
    .gitlab-ci.yml    # CI that builds documentation
    CONTRIBUTING.md   # To read before contributing
    LICENSE
    mkdocs.yml        # The docs configuration file
    poetry.lock
    pyproject.toml    # Dependencies, project metadata and more
    README.md
    setup.py          # Setup file, see `pyproject.toml`
