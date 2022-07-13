We provide some help to choose [scyan.Scyan](../api/model.md) parameters. We listed below the most important ones.

- `prior_std` is probably one of the most important parameters. Its default value should work for most of the usage, but if needed it can be changed. A low `prior_std` (about `0.15`) will help better separate the populations, but it may be too stringent and some small populations may disappear. In contrast, a high `prior_std` (about `0.4`) increases the chances to have a large diversity of populations, but their separation may be less clear. For a project where populations are easy to identify, we thus recommend lowering the `prior_std`.
- `temperature` can help better capture small populations. For that, you can lower the temperature (to `0.5` for instance). If it is not enough, try using `modulo_temp = 3`.
- `batch_ref` is the reference batch we use to align distributions. By default, we use the batch where we have the most cells, but you can choose your own reference. For that, please choose a batch that is representative of the diversity of populations you want to annotate, it can help the batch effect correction.
- `continuous_covariate_keys` and `categorical_covariate_keys` can be provided to the model if you have some. For instance, if you changed one antibody, you can add a categorical covariate telling which samples have been measured with which antibody. Any covariate may help the model annotations and batch effect correction.

## Hyperparameter search

If you want to automate the choice of the hyperparameters, you can also run a hyperparameter optimization with [Hydra](https://hydra.cc/docs/intro/). For that, clone the repository and make an editable install of the project (see [Getting Started](../getting_started.md)). Then, [configure your project](./hydra_wandb.md) and follow the instructions.
