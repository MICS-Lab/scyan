!!! note

    If not done yet, you may be interested in reading our [advice to improve your knowledge table](../../advanced/advice). The default `Scyan` parameters should work well for most cases, so you should first check your knowledge table.

## Main parameters

We provide some help to choose [scyan.Scyan](../../api/model) parameters. We listed below the most important ones.

- `prior_std` is probably one of the most important parameters. Its default value should work for most of the usage, but it can be changed if needed. A low `prior_std` (about `0.15`) will help better separate the populations, but it may be too stringent, and some small populations may disappear. In contrast, a high `prior_std` (about `0.4`) increases the chances of having a large diversity of populations, but their separation may be less clear. For a project where populations are easy to identify, we thus recommend lowering the `prior_std`.
- Reducing the `temperature` can help better capture small populations. For instance, you can lower the temperature to `0.5`. If it is not enough, try using `modulo_temp = 3`.
- `batch_ref` is the reference batch we use to align distributions. By default, we use the batch where we have the most cells, but you can choose your own reference. For that, please choose a batch that is representative of the diversity of populations you want to annotate; it can help the batch effect correction.
- `continuous_covariate_keys` and `categorical_covariate_keys` can be provided to the model if you have some. For instance, if you changed one antibody, you can add a categorical covariate telling which samples have been measured with which antibody. Any covariate may help the model annotations and batch effect correction.

## Hyperparameter search

If you want to automate the choice of the hyperparameters, you can also run a hyperparameter optimization with [Hydra](https://hydra.cc/docs/intro/). See how to [configure your project and run Hydra](../../advanced/hydra_wandb).
