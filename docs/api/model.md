```py
# Usage example

import scyan
from scyan import Scyan

adata, marker_pop_matrix = scyan.data.load("aml")

model = Scyan(adata, marker_pop_matrix)
model.fit()
```

::: scyan.Scyan
