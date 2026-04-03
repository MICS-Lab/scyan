<p align="center">
  <img src="https://github.com/MICS-Lab/scyan/raw/main/docs/assets/logo.png" alt="scyan_logo" width="400"/>
</p>

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/scyan.svg)](https://pypi.org/project/scyan)
[![Downloads](https://static.pepy.tech/badge/scyan)](https://pepy.tech/project/scyan)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://mics-lab.github.io/scyan/)
![Build](https://github.com/MICS-Lab/scyan/workflows/ci/badge.svg)
[![License](https://img.shields.io/pypi/l/scyan.svg)](https://github.com/MICS-Lab/scyan/blob/main/LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

<p align="center"><i>
  🧬 <b>S</b>ingle-cell <b>Cy</b>tometry <b>A</b>nnotation <b>N</b>etwork
</i></p>

Based on biological knowledge prior, Scyan provides a fast cell population annotation without requiring any training label. It is an interpretable model that also corrects batch-effect and can be used for debarcoding, cell sampling, and population discovery.

# Documentation

The [complete documentation can be found here](https://mics-lab.github.io/scyan/). It contains installation guidelines, tutorials, a description of the API, etc.

# Overview

Scyan is a Bayesian probabilistic model composed of a deep invertible neural network called a normalizing flow (the function $f_{\phi}$). It maps a latent distribution of cell expressions into the empirical distribution of cell expressions. This cell distribution is a mixture of gaussian-like distributions representing the sum of a cell-specific and a population-specific term. Also, interpretability and batch effect correction are based on the model latent space — more details in the article's Methods section.

<p align="center">
  <img src="https://github.com/MICS-Lab/scyan/raw/main/docs/assets/overview.png" alt="overview_image"/>
</p>

# Getting started

Scyan can be installed on every OS with `pip` for `python>=3.11`.

## Install from PyPI (recommended)

```bash
pip install scyan
```

## Install locally (if you want to contribute)

> Advice (optional): We advise creating a new environment via a package manager. For instance, you can create a new `conda` environment:
>
> ```bash
> conda create --name scyan python=12
> conda activate scyan
> ```

Clone the repository and move to its root:

```bash
git clone https://github.com/MICS-Lab/scyan.git
cd scyan
```

Choose one of the following, depending on your needs (it should take at most a few minutes):

```bash
pip install .                            # pip minimal installation (library only)
pip install -e .                         # pip installation in editable mode
pip install -e '.[dev,hydra,discovery]'  # pip installation with all the extras
```

## Basic usage / Demo

```py
import scyan

adata, table = scyan.data.load("aml") # Automatic loading

model = scyan.Scyan(adata, table)
model.fit()
model.predict()
```

This code should run in approximately 40 seconds (once the dataset is loaded).
For more usage demo, read the [tutorials](https://mics-lab.github.io/scyan/tutorials/usage/) or the complete [documentation](https://mics-lab.github.io/scyan/).

# Technical description

Scyan is a **Python** library based on:

- [_AnnData_](https://anndata.readthedocs.io/en/latest/), a data library that works nicely with single-cell data
- [_Pytorch_](https://pytorch.org/), a deep learning framework
- [_Pytorch Lightning_](https://www.pytorchlightning.ai/), for model training

Optionally, it also supports:
- [_Hydra_](https://hydra.cc/docs/intro/), for project configuration
- [_Weight & Biases_](https://wandb.ai/site), for model monitoring

# Cite us

Our paper is published in ***Briefings in Bioinformatics*** and is available [here](https://doi.org/10.1093/bib/bbad260).
```txt
@article{10.1093/bib/bbad260,
    author = {Blampey, Quentin and Bercovici, Nadège and Dutertre, Charles-Antoine and Pic, Isabelle and Ribeiro, Joana Mourato and André, Fabrice and Cournède, Paul-Henry},
    title = "{A biology-driven deep generative model for cell-type annotation in cytometry}",
    journal = {Briefings in Bioinformatics},
    pages = {bbad260},
    year = {2023},
    month = {07},
    issn = {1477-4054},
    doi = {10.1093/bib/bbad260},
    url = {https://doi.org/10.1093/bib/bbad260},
    eprint = {https://academic.oup.com/bib/advance-article-pdf/doi/10.1093/bib/bbad260/50973199/bbad260.pdf},
}
```
