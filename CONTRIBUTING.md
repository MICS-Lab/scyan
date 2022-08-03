# Contributing to Scyan

Contributions are welcome as we aim to continue improving the library. For instance, you can contribute by:

- Reporting a bug
- Reporting some difficulties in using the library
- Discussing the current state of the code
- Submitting a fix
- Proposing new features

## Opening an issue

If you have difficulty installing the library, if you discovered a bug, or need a new feature, then you can **open an issue**.
We will try to discuss it with you and resolve your issue as soon as possible.
Especially if you have any questions about the usage of the library or difficulties having satisfying results, don't hesitate to ask.

## Contributing to the code

1. Install the library in editable mode (see [Getting Started](https://mics-lab.github.io/scyan/getting_started/)). Using `poetry` is recommended.
2. Create your personal branch from `dev`.
3. Make sure you read the coding guidelines below.
4. Implement your changes.
5. Run the tests via `pytest` (or `poetry run pytest`).
6. If needed, you can update the documentation. To do so, update the files in `./docs/` and run `mkdocs serve` (or `poetry run mkdocs serve`) to see your changes.
7. Create a pull request with explanations about your developed features. Then, wait for discussion and validation of your pull request.

## Coding guidelines

- Use the `black` formatter and `isort`. Their usage should be automatic as they are in the `pyproject.toml` file. Depending on your IDE, you can choose to format your code on save.
- Follow the [PEP8](https://peps.python.org/pep-0008/) style guide. In particular, use snake_case notations (and PascalCase for classes).
- Provide meaningful names to all your variables and functions.
- Use relative imports to `scyan`.
- Document your functions and type your function inputs/outputs.
- Create your functions in the intended file, or create one if needed. See the project layout.
- Try as much as possible to follow the same coding style as the rest of the library.
