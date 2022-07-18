# Configuration and monitoring

If needed, you can use [Hydra](https://hydra.cc/docs/intro/) to manage your configuration. It allows to easily run the scripts or run hyperparameter optimization. You can also monitor your jobs with [Weight & Biases](https://wandb.ai/site)

For that, clone the repository and make an editable install of the project (see [Getting Started](https://mics_biomathematics.pages.centralesupelec.fr/biomaths/scyan/getting_started/)). Then, you have to follow the step listed below.

## Create a new project configuration

Create a new project at `config/project/<your-project-name>.yaml`.
Provide a `name`. It should be the one where you store your data (see how to [create your dataset](./data.md)). We advise to have `name = <your-project-name> = <your-dataset-name>`.

Add optionally:

- `size` or `table` if you don't want to use the default table or anndata files from your dataset.
- `batch_key` (and eventually `batch_ref`) if you want to correct the batch effect.
- You can add some `continuous_covariate_keys` and `categorical_covariate_keys` (as a list of items).
- `wandb_project_name`, the name of your Weight and Biases project for model monitoring. It will log all the metrics over the different epochs and saves different figures online.

## Other configuration

Update `config/config.yaml` for some basic configuration. The most important config variables are:

- `n_run`: the number of runs per trial (we advise choosing at least 5 to smooth the results at each trial).
- `save_predictions` can be set to `True` to save scyan predictions at each run. They will be saved in the log directory created by Hydra.
- `wandb.mode = online` if you want to use Weight and Biases.

!!! Tips

    Every config parameter can be updated directly via the command line (see [Hydra docs](https://hydra.cc/docs/intro/)). Thus, you don't have to change the parameters in the config files directly for every run.

## (Optional) Hyperparameter optimization configuration

Update `config/sweeper/optuna.yaml` to select the parameters you want to optimize and their ranges. In particular, select the right number of trials, or update it via the command line.

!!! check

    Now that you have configured your project, you can run the scripts (see [running scripts](./scripts.md)) by providing the argument `project=<your-project-name>`.
