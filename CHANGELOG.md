## [1.7.0] - 2026-04-07

- Move scyan to our new GitHub organization: [prism-oncology](https://github.com/prism-oncology) (WIP)
- Move to `uv` and `ruff`
- Upgrade dependencies to now use python>=3.11 and more recent deps versions

## [1.6.2] - 2024-05-22

Hotfix: use get_cmap from matplotlib.pyplot (#35)

## [1.6.1] - 2024-04-08

Rename `correct_spillover` -> `compensate`

## [1.6.0] - 2024-04-08

### Changed
- More dependencies version support
- Use `"cpu"` accelerator by default

### Added
- Spillover matrix reading + function to apply it in `scyan.preprocess`

### Fixed
- Issue when running `.predict` after training on GPU (#34)

## [1.5.4] - 2024-01-08

### Added
- `CHANGELOG.md` file
- Fix NA presence in `refine_fit` when not all pops are predicted [#27](https://github.com/prism-oncology/scyan/issues/27)
- Deploy on PyPI when pushing a new version (i.e., a tag that starts with `v`)
