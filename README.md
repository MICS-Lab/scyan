<img src="./docs/assets/logo.png" alt="scyan_logo" width="400"/><br />

Scyan (**S**ingle-cell **Cy**tometry **A**nnotation **N**etwork) is a flow-based deep generative network that annotates mass and spectral cytometry cells. It leverages expert knowledge to make predictions without any gating or labeling required.

# Model features overview

What can be done with Scyan?

# Getting started

### Install with pip

Comming soon

### Install locally

Clone the repository and then

```
pip install -r requirements.txt
```

## Basic usage

```python
import scyan

... # import data
model = scyan.Scyan(adata, marker_pop_matrix)
model.fit()
```

# Docs

- `mkdocs new [dir-name]` - Create a new project.
- `mkdocs serve` - Start the live-reloading docs server.
- `mkdocs build` - Build the documentation site.
- `mkdocs -h` - Print help message and exit.
