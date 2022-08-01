## Installation

Scyan can be installed on every OS with `pip` or [`poetry`](https://python-poetry.org/docs/).

On MacOS / Linux, `python>=3.8,<3.11` is required, while `python>=3.8,<3.10` is required on Windows. The preferred Python version is `3.9`.

!!! note "Advice (optional)"

    We advise creating a new environment via a package manager (except if you use Poetry, which will automatically create the environment).

    For instance, you can create a new `conda` environment:

    ```bash
    conda create --name scyan python=3.9
    conda activate scyan
    ```

Choose one of the following, depending on your needs (it should take at most a few minutes):

=== "With PyPI"

    ``` bash
    # Available after publication
    pip install scyan
    ```

=== "Local install (pip, library only)"

    ``` bash
    git clone https://github.com/MICS-Lab/scyan.git
    cd scyan

    pip install .
    ```

=== "Local install (pip, editable mode)"

    ``` bash
    git clone https://github.com/MICS-Lab/scyan.git
    cd scyan

    pip install -e '.[dev,docs,discovery]'
    ```

=== "Poetry (editable mode)"

    ``` bash
    git clone https://github.com/MICS-Lab/scyan.git
    cd scyan

    poetry install -E 'dev docs discovery'
    ```

## Usage

### Minimal example

```py
import scyan

adata, marker_pop_matrix = scyan.data.load("aml")

model = scyan.Scyan(adata, marker_pop_matrix)
model.fit()
model.predict()
```

### Inputs details

- `adata` is an [AnnData](https://anndata.readthedocs.io/en/latest/) object, whose variables (`adata.var`) corresponds to markers, and observations (`adata.obs`) to cells. `adata.X` is a matrix of size ($N$ cells, $M$ markers) representing cell-marker expressions after being **preprocessed** ([asinh](./api/asinh.md) or [logicle](./api/auto_logicle.md)) and [**standardized**](./api/scale.md).
- `marker_pop_matrix` is a [pandas DataFrame](https://pandas.pydata.org/) with $P$ rows (one per population) and $M$ columns (one per marker). Each value represents the knowledge about the expected expression, i.e. `-1` for negative expression, `1` for positive expression, or `NA` if we don't know. It can also be any float value such as `0` or `0.5` for mid and low expressions, respectively (use it only when necessary).

!!! note "Help to create the `adata` object and the `marker_pop_matrix`"

    Read the [preprocessing tutorial](./tutorials/preprocessing.ipynb) if you have a FCS file and want explanations to initialize `Scyan`.

!!! check

    Make sure every marker from the table (i.e. columns names of the DataFrame) is inside the data, i.e. in `adata.var_names`.

## Resources to guide you

- Read the tutorials (e.g. [how to prepare your data](./tutorials/preprocessing.ipynb) or [usage example with interpretability](./tutorials/usage.ipynb)).
- Read our [advice](./advanced/advice.md) to design the knowledge table.
- Read the API to know more about what you can do (e.g. [scyan.Scyan](./api/model.md)).
- [Save and load your own dataset](./advanced/data.md).
- [How to choose the model parameters if you don't want to use the default ones](./advanced/parameters.md).
