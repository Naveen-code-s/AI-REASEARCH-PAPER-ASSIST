# AI Research Paper Assistant

A Python project for helping users work with research papers. The repository includes an application entry point and modules for citation handling, shared context, and configuration.

## Repository contents

- `app.py` — application entry point.
- `citation_engine.py` — citation-related processing.
- `citation_finder.py` — citation lookup functionality.
- `global_context.py` — shared application context.
- `config.py` — application configuration.
- `AI REASEARCH PAPER ASSIST` and `AI RESEARCH PAPER` — Git submodules included in the repository.

> The repository does not currently show a dependency manifest or a documented run command. Add those details after confirming the app's framework and required packages.

## Getting the source

Clone the repository and initialize its submodules:

```bash
git clone --recurse-submodules https://github.com/Naveen-code-s/AI-REASEARCH-PAPER-ASSIST.git
cd AI-REASEARCH-PAPER-ASSIST
```

If you already cloned without submodules, run:

```bash
git submodule update --init --recursive
```

## Setup

Use a Python virtual environment, then install the dependencies required by the project. Once the dependency list is added to the repository, install it with:

```bash
pip install -r requirements.txt
```

Configure any API keys or other secrets required by `config.py` through environment variables or a local, ignored secrets file. Do not commit secrets.

## Run

The correct start command depends on the framework used by `app.py`. Document and verify it here so contributors can launch the assistant reliably.

## Development notes

- Add a `requirements.txt` (or `pyproject.toml`) to make setup reproducible.
- Document supported paper formats, citation sources, and any external services.
- Include a small sample or screenshot so visitors can see what the assistant does.

## License

No license file is currently documented. Add a `LICENSE` file if you want others to know how they may use, modify, or distribute this project.
