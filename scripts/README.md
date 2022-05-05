# Hydra and Weight & Biases scripts

These scripts are designed to be used together with Hydra and Weight & Biases.

## Important notes

- For a **normal usage** of Scyan, do not use these scripts, please refer to the documentation instead.
- These scripts can be used to reproduce the results from the article.

## Scripts available

1. `run.py`: runs Scyan, with the possibility of hyperparameter search with Hydra and Optuna.
2. `timing.py`: used to time Scyan on different dataset sizes.
3. `testing.py`: tests Scyan stability over multiple run with a different seed.

## Usage

### Examples

At the root of the repository, run the following command lines (you can change the script name to the scripts listed above).

```bash
# Simple run with default config
python -m scripts.run

# Using the debug trainer for a quick debugging
python -m scripts.run trainer=debug

# Use the GPU trainer and enable wandb and saves umap after training
python -m scripts.run trainer=gpu wandb.mode=online wandb.save_umap=true

# Hyperoptimization (-m option) with wandb enabled and working on the BMMC dataset
python -m scripts.run -m wandb.mode=online project=bmmc
```

### Hydra config

The scripts above use the Hydra config files (see the `config` folder at the root of the project).

You can read their [documentation](https://hydra.cc/docs/intro/) to learn more about Hydra.

### Weight & Biases

Weight & Biases (a.k.a. wandb) can be setup for experiment tracking to improve Scyan model management.

For instance, it logs all the metrics over the different epochs and saves different figures online.

You can read their [documentation](https://docs.wandb.ai/) to learn more about Weight & Biases.
