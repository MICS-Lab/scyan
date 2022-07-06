## Installation

### Via PyPI

Soon available

### Local installation

!!! note "Advice (optional)"

    If using `pip`, we advise to create a new environment via a package manager.
    For instance, you can create a new conda environment:

    ```bash
    conda create --name scyan python=3.9
    conda activate scyan
    ```

Scyan can be installed with `pip` or `poetry` after cloning the repository:

=== "With pip (dev mode)"

    ``` bash
    git clone git@gitlab-research.centralesupelec.fr:mics_biomathematics/biomaths/scyan.git
    cd scyan

    pip install -e '.[dev,docs]'
    ```

=== "With pip (package only)"

    ``` bash
    git clone git@gitlab-research.centralesupelec.fr:mics_biomathematics/biomaths/scyan.git
    cd scyan

    pip install .
    ```

=== "With poetry"

    ``` bash
    git clone git@gitlab-research.centralesupelec.fr:mics_biomathematics/biomaths/scyan.git
    cd scyan

    poetry install -E 'dev docs'
    ```

## Usage

### Example

```py
import scyan

adata, marker_pop_matrix = scyan.data.load("aml")

model = scyan.Scyan(adata, marker_pop_matrix)
model.fit()
model.predict()
```

### Inputs details

- `adata` is an [AnnData](https://anndata.readthedocs.io/en/latest/) object, whose variables (`adata.var`) corresponds to markers, and observations (`adata.obs`) to cells. `adata.X` is a matrix of size ($N$ cells, $M$ markers) representing cell expressions after being **preprocessed** (asinh or logicle) and **standardized**.
- `marker_pop_matrix` is a [pandas DataFrame](https://pandas.pydata.org/) with $P$ rows (one per population) and $M$ columns (one per marker). Each value represents the knowledge about the expected expression, i.e. `-1` for negative expression, `1` for positive expression, `NA` if we don't know, or any float value such as `0` for mid expressions.

!!! important

    Make sure every marker from the table (i.e. columns names of the DataFrame) is inside the data, i.e. in `adata.var_names`.

### Additional resources

- See the tutorials
- Find the right parameters
- Read the API
