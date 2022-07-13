# Add your own dataset

## Data folder description

If you cloned the repository, you have a folder named `./data/` at the root of the repository. If you made the install with PyPI, it is located in your home and is called `./scyan_data` (if not, consider running `scyan.data.load("bmmc")`, it will create it).

From now on, we will denote this folder by `<data-folder>`. It contains every data file used by the project and the tutorials. Every folder represents a project, and contains multiple files:

1. One or multiple `csv` file(s) containing the marker-population table (or expert knowledge).
2. One or multiple `h5ad` file(s) containing the marker expressions. They are gitignored but automatically loaded when running the project (because of the file size, we prefer not to store it on the main repository).

!!! note

    By default, `scyan.data.load("<project-name>")` loads the filenames `default.csv` and `default.h5ad`. If you don't want to use the default files, you can choose which `csv` or which `h5ad` you want to load (see [`scyan.data.load`](../api/load.md)).

## Creating your own data folder

Create a folder with the name of your project in the `<data-folder>` and add a `csv` and a `h5ad` file inside, see below for the instructions.

!!! note

    The name of the `csv` file and the `h5ad` file are the ones you will reuse when loading your dataset with `scyan.data.load(...)`, so we advise creating `default.csv` and `default.h5ad` first, but you can choose any filename as long as you keep the right extension.

### Expert knowledge

Create a `csv` file that contains biological knowledge (see `<data-folder>/aml/default.csv` for an example). Each row corresponds to a population and each column to a marker. Values inside the table can be:

- `-1` for negative expressions.
- `1` for positive expressions.
- Some float values such as `0` or `0.5` for mid and low expressions respectively (use it only when necessary).
- `NA` when you don't know or if it is not applicable.

### Cytometry data

Create a `h5ad` file containing your cytometry data (see `<data-folder>/aml/default.h5ad` for an example). Consider reading the [anndata documention](https://anndata.readthedocs.io/en/latest/) if you have never heard about `h5ad` and `anndata` before.

!!! tips

    You can load a FCS file and transform it into a `h5ad` file with:
    ```python
    adata = scyan.read_fcs("<path-to-fcs>")
    adata.write_h5ad("<path_to_h5ad>")
    ```
    See more details about [`scyan.read_fcs`](../api/read_fcs.md).

!!! check

    You can simply save your `adata` object with raw marker expressions. Be sure `adata.var_names` are the correct marker names (i.e. the ones in your `csv` file).

## Load your dataset

Congrats, you can now load your dataset with `scyan`:

```python
adata, marker_pop_matrix = scyan.data.load("<foldername-you-created>")
```
