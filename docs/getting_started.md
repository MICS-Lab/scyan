## Installation

Scyan can be installed on every OS with `pip` or [`poetry`](https://python-poetry.org/docs/).

On macOS / Linux, `python>=3.8,<3.11` is required, while `python>=3.8,<3.10` is required on Windows. The preferred Python version is `3.9`.

!!! note "Advice (optional)"

    We advise creating a new environment via a package manager (except if you use Poetry, which will automatically create the environment).

    For instance, you can create a new `conda` environment:

    ```bash
    conda create --name scyan python=3.9
    conda activate scyan
    ```

Choose one of the following, depending on your needs (it should take at most a few minutes):

=== "From PyPI"

    ``` bash
    pip install scyan
    ```

=== "Local install (pip)"

    ``` bash
    git clone https://github.com/MICS-Lab/scyan.git
    cd scyan

    pip install .
    ```

=== "Local install (pip, dev mode)"

    ``` bash
    git clone https://github.com/MICS-Lab/scyan.git
    cd scyan

    pip install -e '.[dev,hydra,discovery]'
    ```

=== "Poetry (dev mode)"

    ``` bash
    git clone https://github.com/MICS-Lab/scyan.git
    cd scyan

    poetry install -E 'dev hydra discovery'
    ```

## Usage

### Minimal example

```py
import scyan

adata, table = scyan.data.load("aml") # Automatic loading

model = scyan.Scyan(adata, table)
model.fit()
model.predict()
```

This code should run in approximately 40 seconds (once the dataset is loaded).

### Inputs details

- `adata` is an [AnnData](https://anndata.readthedocs.io/en/latest/) object, whose variables (`adata.var`) corresponds to markers, and observations (`adata.obs`) to cells. `adata.X` is a matrix of size ($N$ cells, $M$ markers) representing cell-marker expressions after being **preprocessed** ([asinh][scyan.preprocess.asinh_transform] or [logicle][scyan.preprocess.auto_logicle_transform]) and [**standardized**][scyan.preprocess.scale].
- `table` is a [pandas DataFrame](https://pandas.pydata.org/) with $P$ rows (one per population) and $M$ columns (one per marker). Each value represents the knowledge about the expected expression, i.e. `-1` for negative expression, `1` for positive expression, or `NA` if we don't know. It can also be any float value such as `0` or `-0.5` for mid and low expressions, respectively (use it only when necessary).

!!! note "Help to create the `adata` object and the `table`"

    Read the [preprocessing tutorial](../tutorials/preprocessing) if you have an FCS file and want explanations to initialize `Scyan`. You can also look at [existing tables](https://github.com/MICS-Lab/scyan_data/blob/main/public_tables.md).

!!! check

    Make sure every marker from the table (i.e. columns names of the DataFrame) is inside the data, i.e. in `adata.var_names`.

## Resources to guide you

- Read the tutorials (e.g. [how to prepare your data](../tutorials/preprocessing) or [usage example with interpretability](../tutorials/usage)).
- Read our [advice](../advice/#advice-for-the-creation-of-the-table) to design the knowledge table.
- Read the API to know more about what you can do (e.g. [scyan.Scyan][]).
- [Save and load your own dataset](../advanced/data).
- [How to choose the model parameters if you don't want to use the default ones](../advanced/parameters).
