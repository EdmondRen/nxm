# Repository Guidelines

## Project Structure & Module Organization

This repository is a small, notebook-driven research project for multi-channel, multi-template event reconstruction.

- `_config.py` contains detector constants, channel groupings, calibration weights, dataset series, and local data paths.
- `notebooks/0_do_pca.ipynb` develops PCA-based pulse templates; `notebooks/1.nxm_dev.ipynb` develops the NxM reconstruction workflow. Keep numbered prefixes when adding notebooks so their execution order stays clear.
- `doc/nxm.md` records the mathematical model and derivation.
- `data` is intentionally ignored and may be a local symlink to large experiment data. Do not commit generated datasets or machine-specific artifacts.

## Build, Test, and Development Commands

There is no package build, dependency manifest, or automated test suite yet. Work from the repository root in an environment with Jupyter and the scientific Python dependencies used by the notebooks (`numpy`, `scipy`, `matplotlib`, `scikit-learn`, `joblib`, and `pycrp`).

- `jupyter lab` opens the notebooks for interactive development.
- `python -m py_compile _config.py` performs a quick syntax check of the shared configuration.
- `jupyter nbconvert --to notebook --execute notebooks/1.nxm_dev.ipynb --output /tmp/nxm_dev.ipynb` smoke-tests a notebook without replacing the tracked copy. It requires valid data paths and locally available project dependencies.

## Coding Style & Naming Conventions

Use four-space indentation and PEP 8 conventions for Python. Prefer `snake_case` for functions and variables, `UPPER_CASE` for true constants, and descriptive names for detector quantities. Keep reusable configuration or helpers in Python modules rather than duplicating cells. Add units or provenance comments for physical constants and calibration values. In Markdown, keep equations and terminology consistent with `doc/nxm.md`.

## Testing Guidelines

For every change, run the syntax check and execute the affected notebook from a clean kernel. Verify that numerical outputs and plots remain physically sensible. If reusable Python logic grows, add `pytest` tests under `tests/` using names such as `test_config.py` and `test_<behavior>()`. Avoid tests that depend on unavailable institutional data; use small synthetic arrays or fixtures.

## Commit & Pull Request Guidelines

The history currently contains only `first commit`, so no established convention exists. Use short, imperative subjects such as `Add PCA template validation`. Keep configuration, notebook, and documentation changes focused. Pull requests should explain the scientific or implementation change, list validation performed, note required external data or environment assumptions, and include representative plots when results change. Never commit credentials, private paths, or large notebook-generated data.



## Documentation and Plan History

- All generated or modified code must include explanatory comments or docstrings following Google style. Python docstrings should use `Args`, `Returns`, and `Raises` sections where applicable; use the corresponding structured comment style for other languages.
- In plan mode, save the complete plan in `plan_history/` before beginning execution. Name the file `YYYYMMDD-HHMMSS-short-description.md` using local repository time.
- Maintain `plan_history/README.md` as a brief cumulative index. Add one dated entry for every newly generated plan, including a link to its full archived plan.