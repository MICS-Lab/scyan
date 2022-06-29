Scyan annotation model.

It inherites from Pytorch Lighning module and is a wrapper to the `ScyanModule` that contains the core logic (the loss implementation, the forward function, ...). While `ScyanModule` works on tensors, this class (`Scyan`) works directly on `AnnData` objects.

## Usage

```py
import scyan
from scyan import Scyan

adata, marker_pop_matrix = scyan.data.load("aml")
model = Scyan(adata, marker_pop_matrix)
```

## Class description

::: scyan.Scyan
