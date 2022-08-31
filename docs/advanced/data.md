# Add your own dataset

## Prepare your Python objects

You must prepare your cytometry data and create your knowledge table as described in the [preprocessing tutorial](/tutorials/preprocessing).

!!! tips

    Read our [advice](/advanced/advice) to create the knowledge table. A great table leads to better predictions.

!!! info

    If needed, you have an example of an `adata` object and a knowledge table if you run:

    ```python
    adata, marker_pop_matrix = scyan.data.load("aml")
    ```

## Save your dataset

Now that you have created an `adata` object and a `marker_pop_matrix` table, you can simply save them (for more details, see [scyan.data.add](/api/datasets/#scyan.data.add)):

```python
scyan.data.add("<your-project-name>", adata, marker_pop_matrix)
```

## Load your dataset

Congrats, you can now load your dataset with `scyan`:

```python
adata, marker_pop_matrix = scyan.data.load("<your-project-name>")
```
