::: scyan.tools.umap
    options:
      show_root_heading: true

::: scyan.tools.leiden
    options:
      show_root_heading: true

::: scyan.tools.subcluster
    options:
      show_root_heading: true

::: scyan.tools.palette_level
    options:
      show_root_heading: true

::: scyan.tools.cell_type_ratios
    options:
      show_root_heading: true

::: scyan.tools.mean_intensities
    options:
      show_root_heading: true

::: scyan.tools.PolygonGatingUMAP
    options:
      show_root_heading: true
      members:
        - __init__
        - select
        - save_selection
        - extract_adata
  
::: scyan.tools.PolygonGatingScatter
    options:
      show_root_heading: true
      members:
        - __init__
        - select
        - save_selection
        - extract_adata
