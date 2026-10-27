# Adopt One-Sided CSD Density and Further Accelerate Fits

## Summary

Change `NxMReconstructor` to accept standard one-sided RFFT cross-spectral
density at `fs=1`, and further optimize fixed-delay fits with lazy DOF
diagnostics and an analytic amplitude gradient.

## CSD Convention

- Define the input density as
  $S_f=m_f\mathbb E[X_fX_f^\dagger]/N$, where $m_f=2$ for interior bins and
  $m_f=1$ for DC and Nyquist.
- Make `estimate_csd()` return this convention.
- Separate RFFT multiplicities used for DOF from likelihood weights
  $m_f^2/N$ used with inverse one-sided PSD.
- Use the likelihood weights in profiling, initialization, cached sufficient
  statistics, and local Jacobian calculations.
- Document raw-covariance and legacy-artifact migration rules.

## Additional Optimization

- Calculate effective or nominal DOF only when its public diagnostic is called.
- Populate `n_fitted_parameters` with the inexpensive active-column count.
- Replace repeated small `einsum` operations with cached matrix-vector forms.
- Add an analytic fixed-delay amplitude gradient and select L-BFGS-B
  automatically for fixed-delay fits while retaining Powell for fitted delays.
- Respect explicit optimizer methods and provide method-specific defaults.

## Validation and Migration

- Test one-sided density estimation, endpoint scaling, converted likelihood
  equivalence, reduced chi-square, and correlated CSD behavior.
- Check the analytic gradient against finite differences and compare optimized
  results with Powell.
- Migrate synthetic and data-backed notebooks to PSD inputs and compatible
  optimizer options without overwriting unrelated notebook work.
- Run syntax checks, the full test suite, clean-kernel notebook smoke tests,
  and a representative performance benchmark.

## Assumptions

- Sampling frequency is fixed at `fs=1`.
- Stored `csd_matrices` artifacts are one-sided spectral sums and are converted
  to density by dividing by trace length.
- Raw covariance inputs are no longer supported by the constructor.
