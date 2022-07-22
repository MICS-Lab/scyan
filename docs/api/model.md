```py
### Usage example
import scyan

adata, marker_pop_matrix = scyan.data.load("aml")

model = scyan.Scyan(adata, marker_pop_matrix)
model.fit()
model.predict()
```

!!! info "Notations"

    $N$ denotes the number of cells, $P$ the number of populations, $M$ the number of markers, and $B$ the size of a mini-batch (not the number of biological batches). You can find other definitions in the article.

::: scyan.Scyan
