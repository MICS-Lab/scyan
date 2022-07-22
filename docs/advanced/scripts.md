# Running scripts

!!! caution

    You can run `Scyan` without using these scripts. Use it **only if** you want to use Hydra and/or Weight and Biases. Before continuing, ensure that you have [configured your project](./hydra_wandb.md).

## Usage examples

At the root of the repository, run the following command lines.
Note that `<project-name>` can be `aml`, `bmmc` or your own project (the one you have configured).

```bash
# One run with the default config
python -m scripts.run project=<project-name> n_run=1

# Running hyperparameter optimization (-m option)
python -m scripts.run -m project=<project-name>

# Using the debug trainer for a quick debugging
python -m scripts.run project=<project-name> trainer=debug

# Use the GPU trainer and enable wandb and save umap after training
python -m scripts.run project=<project-name> trainer=gpu wandb.mode=online wandb.save_umap=true

# Change the model parameters
python -m scripts.run project=<project-name> model.temperature=0.5 model.prior_std=0.3
```

## Reproduce the article results

The hyperparameters were obtained by (unsupervised) hyperparameter optimization (see [Hydra configuration](./hydra_wandb.md)).

```bash
# Testing on BMMC
python -m scripts.run project=bmmc model.hidden_size=32 model.n_hidden_layers=8 model.n_layers=8 model.prior_std=0.35 model.temperature=2.5 model.modulo_temp=2 callbacks.early_stopping.patience=2 callbacks.early_stopping.min_delta=2

# Testing on AML
python -m scripts.run project=aml model.hidden_size=32 model.n_hidden_layers=7 model.n_layers=7 model.prior_std=0.2 model.temperature=0.75 model.modulo_temp=3 model.lr=0.0003 model.batch_size=8192 callbacks.early_stopping.patience=3 callbacks.early_stopping.min_delta=0.5
```
