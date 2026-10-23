# Complete the NxM Reconstruction Documentation

## Summary

Rewrite and complete `doc/nxm.md` as a concise mathematical and procedural guide for the intended regularized MAP workflow. Preserve the existing notation style, correct grammar and inconsistent equations, and use the current notebooks as implementation context without documenting exploratory notebook details.

## Documentation Changes

- Refine the introduction and single-channel model:
  - Retain $M$ samples, mean template $\mathbf s_0$, PCA basis $\mathbf B$, amplitude $A$, and standardized coefficients $\mathbf z$.
  - Correct the PCA prior precision from the undefined $C^{-1}$ to $I$.
  - Apply the time shift $t_0$ to both $\mathbf s_0$ and $\mathbf B$.
  - Define the profiled objective obtained by substituting the conditional solution $\widehat{\mathbf z}(A,t_0)$.
- Add the multi-channel derivation:
  - Use $J$ channels, independent nonnegative amplitudes $A_j$, bounded per-channel delays $t_j$, and independent per-channel PCA priors.
  - Stack channel data and define the block-diagonal shape matrix $D(\mathbf A,\mathbf t)=\operatorname{blockdiag}(A_j\mathbf B_j(t_j))$.
  - Express the joint MAP objective with the full cross-channel noise covariance and derive $\widehat{\mathbf z}=(D^TWD+I)^{-1}D^TW(\mathbf x-\boldsymbol\mu)$.
  - Explain that noise correlation couples the fitted coefficients even though their PCA priors are independent.
- Present the frequency-domain form:
  - Define the channel vector and Hermitian CSD matrix at each frequency.
  - Represent delays through Fourier phase factors and use the CSD-weighted residual sum plus $\mathbf z^T\mathbf z$.
  - State that FFT, template, and CSD normalization must be consistent; baseline-subtract traces and omit the DC contribution.
  - Hermitian-symmetrize each CSD and floor small eigenvalues before inversion.
- Complete the processing procedure: template generation, CSD estimation, linear optimal-filter initialization, bounded profiled MAP optimization, recorded diagnostics, and validation.
- Use a fixed configurable PCA count with a current default of three modes per channel.

## Interfaces and Compatibility

- Make no code or public API changes.
- Describe the existing linear optimal filter only as an initializer; use the regularized amplitude-times-shape MAP model for final reconstruction.
- Keep detector-specific delay bounds, filters, frequency limits, and CSD eigenvalue floor configurable rather than presenting them as universal constants.

## Validation

- Check dimensions and scalar/vector notation in every equation.
- Confirm the multi-channel model reduces to the single-channel model for one channel and to independent fits for diagonal noise covariance.
- Ensure all placeholder sections are replaced, spelling and grammar are corrected, and Markdown equations render correctly.
- Keep the result concise and avoid implementation claims not supported by the notebooks or `pycrp`.

## Assumptions

- The regularized MAP model is the final reconstruction; linear optimal filtering is used only for initialization.
- Channel amplitudes are independent and nonnegative.
- Arrival times are independent per channel but bounded near their initial estimates.
- PCA bases and standard-normal priors are independent per channel; cross-channel dependence enters through the noise CSD.
