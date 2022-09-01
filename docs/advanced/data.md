# Add your own dataset

Existing datasets and versions can be listed via [`scyan.data.list()`](/api/datasets/#scyan.data.list). By default, only public datasets are available, but you can add some if you follow the next steps.

## 1. Prepare your Python objects

You must prepare your `cytometry data` and create your `knowledge table` as described in the [preprocessing tutorial](/tutorials/preprocessing). You can also read our [advice](/advanced/advice) to create the knowledge table (a great table leads to better predictions!).

!!! info

    If needed, you have an example of an `adata` object and a knowledge table (a.k.a `marker_pop_matrix`) if you run:

    ```python
    adata, marker_pop_matrix = scyan.data.load("aml")
    ```

## 2. Save your dataset

Now that you have created an `adata` object and a `marker_pop_matrix` table, you can simply save them (for more details, see [scyan.data.add](/api/datasets/#scyan.data.add)):

```python
scyan.data.add("<your-project-name>", adata, marker_pop_matrix)
```

## 3. Load your dataset

Congrats, you can now load your dataset (for more details, see [scyan.data.load](/api/datasets/#scyan.data.load)):

```python
adata, marker_pop_matrix = scyan.data.load("<your-project-name>")
```
