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

## Installation

Scyan can be installed on every OS with `pip` for `python>=3.11`:

```bash
pip install scyan
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
