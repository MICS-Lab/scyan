# Add your own dataset

Existing datasets and versions can be listed via [`scyan.data.list()`][scyan.data.list]. By default, only public datasets are available, but you can add some if you follow the next steps.

## 1. Prepare your Python objects

You must prepare your `cytometry data` and create your `knowledge table` as described in the [preprocessing tutorial](../../tutorials/preprocessing). You can also read our [advice](../../advice/#advice-for-the-creation-of-the-table) to create the knowledge table (a great table leads to better predictions!).

!!! info

    If needed, you have an example of an `adata` object and a knowledge `table` if you run:

    ```python
    adata, table = scyan.data.load("aml")
    ```

## 2. Save your dataset

Now that you have created an `adata` object and a `table`, you can simply save them (for more details, see [scyan.data.add][]):

```python
scyan.data.add("<your-project-name>", adata, table)
```

## 3. Load your dataset

Congrats, you can now load your dataset (for more details, see [scyan.data.load][]):

```python
adata, table = scyan.data.load("<your-project-name>")
```
