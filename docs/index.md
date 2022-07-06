# Scyan documentation

<p align="center">
  <img src="./assets/logo.png" alt="scyan_logo" />
</p>

Scyan stands for **S**ingle-cell **Cy**tometry **A**nnotation **N**etwork. Based on a biological knowledge prior, it provides a fast cell populations annotation without requiring any training label. Scyan is an interpretable model that also corrects batch-effect and can be used for debarcoding / cell sampling / population discovery.

## Overview

Scyan is a Bayesian probabilistic model composed of a deep invertible neural network called a Normalizing Flow (the function $f_{\phi}$). This network maps a latent distribution of cell expressions into the empirical distribution of cell expressions. The latter latent cell distribution is a mixture of gaussian-like distributions representing the sum of a cell-specific term and a population-specific term. The latent space is used for interpretability and batch effect correction. More details on the methods section of the article.

<figure markdown>
  ![Image title](./assets/overview.png)
  <figcaption>a) Overview of the tasks that Scyan can perform. b) Overview of the model architecture. c) One coupling layer, i.e. the elementary unit that compose the Normalizing Flow.</figcaption>
</figure>

## Technical description

Scyan is a **Python** library based on:

- _Pytorch_, a deep learning framework
- _AnnData_, a data library that works nicely with nice single-cell data
- _Pytorch Lighning_ for model training
- _Hydra_ for project configuration
- _Weight & Biases_ for model monitoring

## Project layout

See [Scyan on Gitlab](https://gitlab-research.centralesupelec.fr/mics_biomathematics/biomaths/scyan)

    config/       # Hydra configuration folder (optional use)
    data/         # Data folder containg adata files and csv tables
    docs/         # The documentation folder
    scripts/      # Scripts to reproduce the results from the article
    scyan/                    # Library source code
        module/               # Folder containing neural network modules
            coupling_layer.py
            distribution.py   # Prior distribution (called U in the article)
            real_nvp.py       # Normalizing Flow
            scyan_module      # Core module
        plot/                 # Plotting tools
            ...
        data.py               # Data related functions and classes
        mmd.py                # Maximum Mean Discrepancy implementation
        model.py              # Scyan model class
        utils.py              # Misc functions
    .gitignore
    LICENSE
    mkdocs.yml    # The docs configuration file
    pyproject.toml
    README.md
    requirements.txt
