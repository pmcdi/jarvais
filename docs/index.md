# Welcome to jarvAIs

**j**ust **a** **r**eally **v**ersatile **AI** **s**ervice

jarvAIs is a Python package designed to automate and enhance machine learning workflows. The primary goal of this project is to reduce redundancy in repetitive tasks, improve consistency, and elevate the quality of standardized processes in oncology research.

## Installation

```bash
$ pip install jarvais
```

### (recommended) Create new `pixi` environment for a project

```bash
mkdir my_project
cd my_project
pixi init
pixi add --pypi jarvais
```

### (recommended) Create new conda virtual environment

```bash
conda create -n jarvais python=3.11
conda activate jarvais
pip install jarvais
```

## Modules

This package consists of 3 different modules:

- [**Analyzer**](get_started/analyzer.md): A module that analyzes and processes data, providing valuable insights for downstream tasks.
- [**Trainer**](get_started/trainer.md): A module for training machine learning models, designed to be flexible and efficient.
- [**Explainer**](get_started/explainer.md): A module that explains model predictions, offering interpretability and transparency in decision-making.

## Contributing

Please use the following angular commit message format:

```text
<type>(optional scope): short summary in present tense

(optional body: explains motivation for the change)

(optional footer: note BREAKING CHANGES here, and issues to be closed)

```

`<type>` refers to the kind of change made and is usually one of:

- `feat`: A new feature.
- `fix`: A bug fix.
- `docs`: Documentation changes.
- `style`: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc).
- `refactor`: A code change that neither fixes a bug nor adds a feature.
- `perf`: A code change that improves performance.
- `test`: Changes to the test framework.
- `build`: Changes to the build process or tools.

`scope` is an optional keyword that provides context for where the change was made. It can be anything relevant to your package or development workflow (e.g., it could be the module or function - name affected by the change).

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

