We provide some help to choose [scyan.Scyan](api/model.md) parameters.

- `prior_std` is probably one of the most important parameters. Its default value should work for most of the usage, but if needed it can be changed. A low `prior_std` (about `0.15`) will help better separate the populations, but it may be too stringent and some small population may disappear. In contract, a high `prior_std` (about `0.4`) increases the chances to have a large diversity of population, but their separation may be less clear. For a project were population are easy to indentify, we thus recommend to lower the `prior_std`.
- `temperature` can help better capture small populations. For that, you can lower the temperature (to `0.5` for instance). If it is not enough, try using `modulo_temp = 3`.
- `batch_ref` is the reference batch we use to align distributions. By default, we use the batch where we have the most cells, but you can choose your own reference. For that, please choose a batch that is representative of the diversity of populations you want to annotate, it can help the batch effect correction.
- `continuous_covariate_keys` and `categorical_covariate_keys` can be provided to the model, if you have some. For instance, if you changed one antibody, you can add a categorical covariate telling which samples have been measured with which antibody. Any covariate may help the model annotations and batch effect correction.

## Hyperparameter search

You can also run an hyperparameter search with [Hydra](https://hydra.cc/docs/intro/). For that, clone the repository and make an editable install of the project (see [Getting Started](getting_started.md)). Then, you have to follow the step listed below.

### Select which parameter you want to optimize

Update `config/sweeper/optuna.yaml` to select the parameters you want to optimize, and their ranges. Especially, select the right number of trials.

### Create a new project configuration

Create a new project at `config/project/<your-project>.yaml`.
Project a `name`, and eventually a `wandb_project_name` if you want to use Weight and Biases. The name should be the one where you store your data, i.e. you need to place your data files at `data/<name>` with, inside, a `h5ad` file and a `csv` file. The `csv` if the knowledge table, and the `h5ad` is the anndata containing all the cells (with a batch observation if you want to correct batch effect). You also need to add `size` to the config, which is the prefix of your `h5ad` file, e.g. `default` if your file is called `default.h5ad`.

And optionally:

- You can indicate `batch_key` (and eventually `batch_ref`) if you want to correct batch effect.
- You can add some categorical and continuous covariates names.

### Other configuration

Update `config/config.yaml` for some basic configuration. They can be changed via the command line, or directly by editing the config file. The most important config variables are:

- `n_run`: the number of run per trial (we advise to choose at least 5 to smooth the results at each trial).
- `save_predictions` can be set to `True` to save scyan predictions at each run.
- `wandb.mode = online` if you want to use Weight and Biases.

## Run hyperoptimization

Then, simply run one of the following:

```bash
python -m scripts.run project=<your-project> # no hyperoptimization
python -m scripts.run -m project=<your-project> # normal hyperoptimization
python -m scripts.run -m project=<your-project> n_run=5 save_predictions=true # command line usage to set n_run to 5 and save predictions
```
