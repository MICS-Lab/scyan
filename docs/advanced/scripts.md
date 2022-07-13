# Running scripts

!!! caution

    You can run `Scyan` without using these scripts. Use it **only if** you want to use Hydra and/or Weight and Biases. Before continuing, also make sure that you have [configured your project](./hydra_wandb.md).

## Usage

### Examples

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
