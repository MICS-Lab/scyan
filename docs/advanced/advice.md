!!! important

    The design of the knowledge table is essential for the annotations. The literature can help its creation, but we also provide some advice to enhance the table.

### General advice

- Check that all markers are working as intended. A marker that didn't work during the cytometer acquisition should be removed from the knowledge table, as it can mislead the model.
- It is not required to use all the panel markers. If some markers are not important for the annotation, it is better not to add them to the table.
- Try to provide at least one negative population and one positive population for each marker. For instance, if you provide only "NA" values and "1", then the model has no reference to understanding what is negative and what is positive (even though it should not cause any severe issue in most cases).

!!! note

    However, you can create a column full of NA for a marker that you want to appear in the latent space. It will not be helpful for the annotations, but it can help the population discovery.

- Don't hesitate to provide NA values when the expression is unknown (depending on your table, 70% of NA can be acceptable). Yet, the extreme case where one population has almost only NA can lead to overpredicting this population. Indeed, a population that has only NA means it can likely correspond to any cell (the population is not constrained).
- Sometimes, you will have a marker that is expressed (or not) by most of the population $A$ cells, but you know a few cells from the same population may not express this marker. In that case, we still advise adding the marker on the table, and **Scyan will be able to catch these rare exceptions**, as they look like population $A$ cells for most of the other markers of the panel. Consider the image below: let's say this is the manual gating that you did traditionally. Even though some cells of $A$ are `Marker1+`, you should still write that the population $A$ is `Marker1-`, to make a clear distinction with the population $B$. Also, if you are specifically interested in the cells from $A$ that are `Marker1+`, you can still create another population $A_2$ and write that $A_2$ is `Marker1+` while $A$ is `Marker1-`.

<p align="center">
  <img src="../../assets/example_scatterplot.png" alt="scatterplot" width="350px"/>
</p>

### What should I do if Scyan seems wrong?

- First thing to do is to check your table again. You may have made a typo that could confuse the model. Typically, if you have written `Marker+` for a population that is `Marker-` (or the opposite), it can perturb the prediction toward this population **and** toward other populations.
- Try using [`scyan.plot.probs_per_marker`](../api/probs_per_marker.md). Many markers may show up dark on the heatmap at places they shouldn't, but the errors may be due to only one marker. You can find which marker it is by checking actual marker expressions with `scanpy.pl.umap`, or with a [scatter plot](../api/scatter.md), and then update your table or read some literature again.
- One reason for not predicting a population may be an unbalanced knowledge quantity between two related populations. For instance, having 10 values inside the table for `CD4 T CM` cells versus 5 values for `CD4 T EM` cells will probably make the model predict very few `CD4 T CM` cells. Indeed, `CD4 T CM` has many constraints compared to `CD4 T EM`, which becomes the "easy prediction" (indeed, very few constraints are applied to this population). In that case, read the advice related to the scatter plot above again.

!!! info "Example about how Scyan handles NA"

    If a population $A$ is labeled CD25+, then a cell that is CD25+ provides more confidence to the model towards this population than for a population $B$ for which CD25 is labeled "NA". Yet, if a cell is CD25-, then it will provide more confidence to the population $B$ than for the population $A$. Indeed, CD25- is strong evidence that the cell is not $A$, while it does not penalizes the population $B$ too much.

!!! note "If you still can't make it work"

    You can create an issue or ask for help ([quentin.blampey@gmail.com](mailto:quentin.blampey@gmail.com))
