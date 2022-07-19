# Add your own dataset

## Prepare your Python objects

You have to prepare your cytometry data and create your knowledge table as described in the [preprocessing tutorial](../tutorials/preprocessing.ipynb).

!!! tips

    Read our [advice](../advanced/advice.md) to create this table. A great table leads to better predictions.

!!! info

    You can have an example of a data object and of a knowledge-table by running:

    ```python
    adata, marker_pop_matrix = scyan.data.load("aml")
    ```

## Save your dataset

Now that you have created an `adata` object and a `marker_pop_matrix` table, you can simply save them (for more details, see [scyan.data.add](../api/add.md)):

```python
scyan.data.add("<your-project-name>", adata, marker_pop_matrix)
```

## Load your dataset

Congrats, you can now load your dataset with `scyan`:

```python
adata, marker_pop_matrix = scyan.data.load("<your-project-name>")
```
