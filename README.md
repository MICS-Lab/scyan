<img src="./docs/_static/logo.png" alt="scyan_logo" width="400"/><br />

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

# Generate the docs

```bash
cd docs
sphinx-apidoc -o source ../scyan
make html
open build/html/index.html
```

Or autobuild: `sphinx-autobuild docs/source docs/_build/html`
