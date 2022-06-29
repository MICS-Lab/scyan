# Scyan documentation

## Project layout

    config/       # Hydra configuration folder (optional use)
    data/         # Data folder containg adata files and csv tables
    docs/         # The documentation folder
    scripts/      # Scripts to reproduce the results from the article
    scyan/                    # Library source code
        module/               # Folder containing neural network modules
            coupling_layer.py
            distribution.py   # Prior distribution (called U in the article)
            real_nvp.py       # Normalizing Flow
            scyan_module      # Core module
        plot/                 # Plotting tools
            ...
        data.py               # Data related functions and classes
        mmd.py                # Maximum Mean Discrepancy implementation
        model.py              # Scyan model class
        utils.py              # Misc functions
    .gitignore
    LICENSE
    mkdocs.yml    # The docs configuration file
    pyproject.toml
    README.md
    requirements.txt
