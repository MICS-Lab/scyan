```py
### Usage example
import scyan

adata, table = scyan.data.load("aml")

model = scyan.Scyan(adata, table)
model.fit()
model.predict()
```

!!! info "Notations"

    $N$ denotes the number of cells, $P$ the number of populations, $M$ the number of markers, and $B$ the size of a mini-batch (not the number of biological batches). You can find other definitions in the article.

::: scyan.Scyan
    options:
      heading_level: 3
      group_by_category: False
      members:
        - __init__
        - fit
        - predict
        - predict_proba
        - batch_effect_correction
        - sample
        - pop_names
        - var_names
        - pops
        - level_names
