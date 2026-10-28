# Revise the NxM Reconstruction Note

## Summary

Rewrite `doc/nxm.md` into a cohesive scientific note centered on the regularized pulse model. Present the earlier linear NxM formulation as useful prior work whose assumptions become limiting in the low-SNR, scale-invariant regime considered here. Keep the mathematical narrative in the main body and move implementation, optimization, API, and operational guidance into appendices.

## Main-Body Revisions

- Retain the current title and rewrite the introduction to state the problem, modeling assumptions, and progression from single-channel shape variation to correlated multi-channel reconstruction.
- Rename “Why the previous method fails” to a neutral heading such as “Motivation for a regularized formulation.”
  - Refer generically to the [earlier NxM work](https://arxiv.org/abs/2508.20090), emphasizing its demonstrated high-SNR application.
  - Explain three limitations specifically for the present regime: PCA is an affine reconstruction with an explicit mean; amplitude variation can dominate a truncated PCA basis; and unconstrained extra coefficients can absorb low-SNR noise.
  - Clarify that basis rotations are harmless when the complete span is retained and that the expected reduction in minimized chi-square from additional parameters is ordinary statistical behavior, not an error in the earlier method.
  - Explain that amplitude-normalized template training addresses the first two issues, while the Gaussian PCA prior addresses the third.
- Expand section 1.3 into a mathematical template-generation procedure:
  - Start with aligned, baseline-subtracted, high-SNR training pulses.
  - Alternate between noise-weighted matched-filter amplitude estimates and rebuilding the mean template from amplitude-normalized pulses.
  - Fix the arbitrary amplitude/template scale after each update and rescale amplitudes consistently.
  - Show that defining the updated template as the mean normalized pulse makes the mean residual zero for that update; after amplitude re-estimation, iteration drives it back to an arbitrarily small convergence tolerance.
  - Define the final normalized residuals and apply PCA to them, yielding unit-norm directions and variances used in the scaled basis $\mathbf B$.
- Polish the remaining derivation:
  - Standardize notation, grammar, equation formatting, and terminology.
  - Correct duplicate section numbering and undefined or inconsistent symbols.
  - Keep the single-channel MAP objective, analytic profiling of shape coefficients, shared time shift, multi-channel stacked model, and general frequency-domain likelihood in the main body.
  - Make the transition from single-channel to multi-channel reconstruction explicit and state the one-channel and diagonal-CSD limiting cases.

## Appendices and Interfaces

- Replace the current “Processing procedure” with organized appendices that preserve the existing technical material:
  - **Practical workflow:** training-event selection, alignment, residual preprocessing, PCA artifact metadata, synchronized noise selection, initialization, bounded fitting, and validation.
  - **Spectral conventions and CSD conditioning:** one-sided RFFT multiplicities, the exact CSD-density and likelihood factors, DC treatment, legacy artifact conversion, Hermitian symmetrization, and eigenvalue flooring.
  - **Numerical implementation and usage:** fixed- and free-delay sufficient statistics, caching, optimizer selection, result fields, failure diagnostics, and the roles of `estimate_csd`, `NxMReconstructor`, `initialize`, `fit`, and `reduced_chi_square`.
  - **Goodness-of-fit diagnostics:** nominal and effective degrees of freedom, active-bound treatment, and interpretation limits for reduced chi-square.
- Remove source-file names, optimizer defaults, cache behavior, direct method calls, and artifact-conversion instructions from the main derivation.
- Make no changes to Python APIs, notebooks, numerical behavior, or data formats.

## Validation

- Inspect the rendered Markdown for heading order, equation rendering, links, and appendix navigation.
- Check every equation for consistent dimensions and symbols, including $A$, $\mathbf z$, $\mathbf B$, $D$, $H$, $\mathbf g$, and the delay convention.
- Verify the template-iteration explanation against `notebooks/0_do_pca.ipynb` and all implementation claims against `src/nxm.py` and `tests/test_nxm.py`.
- Confirm the model reduces correctly to the single-channel case and to independent channel fits for diagonal noise covariance.
- Run `python -m py_compile _config.py` and `python -m pytest`; no notebook execution is required because no notebook will be modified.

## Assumptions

- The comparison with prior work remains generic in prose, with an explicit arXiv link but no repeated paper title or author list.
- All current implementation and usage information is retained, but reorganized into appendices.
- Scale-invariant pulse-shape variation, nonnegative amplitudes, bounded delays, per-channel PCA bases, and independent standard-normal PCA priors remain the documented model assumptions.
- The only tracked changes will be `doc/nxm.md` and the two required plan-history files.
