# Contributing to Scyan

Contribution are welcome as we aim to continue improving the library. For instance, you can contribute by:

- Reporting a bug
- Reporting some difficulties to use the library
- Discussing the current state of the code
- Submitting a fix
- Proposing new features

## Opening an issue

If you have any difficulty to install the library, if you discovered a bug, or if you need a new feature, then you can **open an issue**.
We will try to discuss it with you and resolve your issue as soon as possible.
Especially, if you have any question on the usage of the library or difficulties to have satisfying results, don't hesitate to ask.

## Contributing to the code

1. Install the library in development mode (see the documentation) using `poetry` ideally.
2. Create your own personnal branch.
3. Make sure you read the coding guidelines below.
4. Run the tests via `poetry run pytest`.
5. If needed, you can update the documentation. To do so, update the files in `./docs/` and run `poetry run mkdocs serve` to see your changes.
6. Create a pull request and wait for discussion and validation.

## Coding guidelines

- Use the `black` formatter and `isort`. Their usage should be automatic as they are in the `pyproject.toml` file. Depending on your IDE, you can choose to format your code on save.
- Follow the [PEP8](https://peps.python.org/pep-0008/) style guide. In particular, use snake_case notations (and PascalCase for classes).
- Provide meaningful name to all your variables and functions.
- Use relative imports to `scyan`.
- Document your functions and type your function inputs / outputs.
- Create your functions in the intented file, or create one if needed. See the project layout.
- Try as much as possible to follow the same coding style as the rest of the library.
