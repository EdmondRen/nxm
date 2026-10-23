# Implement Frequency-Domain NxM Reconstruction

## Summary

Implement the regularized MAP workflow from `doc/nxm.md` as a standalone NumPy/SciPy module in `src/nxm.py`. Profile the PCA shape coefficients analytically and use bounded Powell optimization for nonnegative channel amplitudes and per-channel delays.

## Public API and Core Implementation

- Add `NxMReconstructor` accepting mean templates `(n_channels, n_samples)`, PCA components `(n_channels, n_modes, n_samples)`, PCA variances `(n_channels, n_modes)`, and raw RFFT noise covariance `(n_frequencies, n_channels, n_channels)`.
- Support optional channel names, a frequency-bin limit, and a relative CSD eigenvalue floor of `1e-6`; default to the full available spectrum.
- Define delays as fractional sample offsets with phase `exp(-2 pi i k t / M)`, use one-sided RFFT weights, and omit DC.
- Hermitian-symmetrize and condition the CSD, validate all public inputs, and cache frequency-domain templates and inverse CSD matrices.
- Implement matched-filter initialization, analytic profiling of PCA coefficients, bounded Powell fitting, and time-domain reconstruction.
- Return an `NxMFitResult` dataclass containing fitted parameters, objective components, optimizer diagnostics, and boundary flags.
- Add `estimate_csd` for synchronized noise arrays shaped `(n_events, n_channels, n_samples)` using the raw covariance convention `E[X_j X_k*]`.
- Keep the module independent of `pycrp`, use only NumPy and SciPy, and add Google-style docstrings.
- Update `doc/nxm.md` with the implemented normalization, delay units, and class entry point.

## Automated Tests

- Add synthetic pytest coverage for validation errors, CSD estimation and conditioning, profiled solves, fractional-delay sign, reconstruction, off-diagonal noise weighting, pulse recovery, noise-only fitting, and optimizer diagnostics.
- At high signal-to-noise ratio require amplitude error below 2%, delay error below 0.15 sample, and PCA-coefficient error below 0.1.
- Run `pytest -q`, compile `src/nxm.py`, and execute both notebooks from clean kernels.

## Validation Notebooks

- Add `notebooks/2_validate_nxm_synthetic.ipynb` with captured pytest results, seeded correlated noise-only fits, simulated pulse injections, metrics, plots, and stated numerical acceptance checks.
- Add `notebooks/3_validate_nxm_held_out.ipynb` using the existing 12-channel templates, CSD, training traces, and the `23231216_233807` series. Exclude the one waveform found in the training set, fit the 26 held-out pulses using 4,000 bins, and show objective, parameter, pull, residual, convergence, and boundary diagnostics.
- Retain concise executed outputs. If ignored real-data artifacts are absent, skip the held-out section with setup guidance instead of failing execution.

## Assumptions

- All channels use the same configurable number of PCA modes; current artifacts use three.
- Fit inputs are baseline-subtracted; only `estimate_csd` performs requested detrending.
- Amplitudes are independent and nonnegative, while delays are independent but bounded near initialization.
- Inputs use raw FFT covariance, not power-density normalization.
- Held-out data provide qualitative validation; quantitative recovery accuracy comes from simulated injections.
