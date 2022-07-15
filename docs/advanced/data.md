# Add your own dataset

## Prepare your objects

Prepare a data object and a table as described below.

!!! note

    You can have an example of a data object and of a knowledge-table by running:

    ```python
    adata, marker_pop_matrix = scyan.data.load("aml")
    ```

### Cytometry data

Create an `AnnData` object containing your cytometry data. Consider reading the [anndata documention](https://anndata.readthedocs.io/en/latest/) if you have never heard about `anndata` before.

!!! tips

    You can load a FCS file with:
    ```python
    adata = scyan.read_fcs("<path-to-fcs>")
    ```
    See more details about [`scyan.read_fcs`](../api/read_fcs.md).

Then, preprocess your `AnnData` object (see a [preprocessing tutorial](../tutorials/preprocessing.ipynb)).

### Expert knowledge

Create a `pd.DataFrame` that contains biological knowledge. Each row corresponds to a population and each column to a marker. Values inside the table can be:

- `-1` for negative expressions.
- `1` for positive expressions.
- Some float values such as `0` or `0.5` for mid and low expressions respectively (use it only when necessary).
- `NA` when you don't know or if it is not applicable.

!!! check

    Be sure the columns of your datafrale correspond to a marker name in `adata.var_names`.

## Save your data

Now that you have create an `adata` object and a `marker_pop_matrix` table, you can simply save them (for more details, see [scyan.data.add](../api/add.md)):

```python
scyan.data.add("<your-project-name>", adata, marker_pop_matrix)
```

## Load your dataset

Congrats, you can now load your dataset with `scyan`:

```python
adata, marker_pop_matrix = scyan.data.load("<your-project-name>")
```
