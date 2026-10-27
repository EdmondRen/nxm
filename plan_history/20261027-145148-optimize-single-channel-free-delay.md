# Optimize Free-Delay 1xM Reconstruction

## Summary

Add an automatic, exact single-channel fast path for `n_channels == 1` and
`fix_delay=False`. Preserve fractional-sample delays and existing results while
replacing repeated frequency-domain matrix construction with cached sufficient
statistics and analytic gradients. Do not cache matrices for every time
offset: in 1xM the template Gram matrices are delay-invariant, while trace
correlations are trace-dependent.

## Implementation Changes

- Cache the single-channel mean norm, PCA Gram matrix, mean/PCA cross term,
  and weighted conjugate template spectra when constructing the reconstructor.
- Cache trace-dependent spectral correlations once per fit, then evaluate the
  exact profiled objective and its amplitude and delay gradients from a single
  phase vector at each fractional delay.
- Select L-BFGS-B automatically for free-delay single-channel fits while
  honoring explicit optimizer methods. Continue using the existing fixed-delay
  and general multi-channel paths.
- Accelerate built-in 1xM initialization by obtaining all integer-delay
  matched-filter responses with one inverse FFT and by reusing the trace FFT
  already calculated by `fit()`.
- Document the delay-invariant single-channel reduction and optimizer choice.
  Do not modify the manually edited validation notebook.

## Public Interface

- Add no public arguments, result fields, or types.
- Change `method=None` from Powell to L-BFGS-B only for free-delay one-channel
  fits. Explicit `method="Powell"` retains Powell semantics while using the
  faster objective evaluator.
- Preserve exact fractional delays, bounds, frequency cutoffs, and
  reconstruction output.

## Validation

- Compare the specialized objective, coefficients, and analytic gradients
  against the general frequency-domain implementation and finite differences.
- Verify automatic routing, optimizer equivalence, recovery, boundary cases,
  and exact inverse-FFT initialization for odd/even sample counts and truncated
  frequency ranges.
- Run the syntax check, complete pytest suite, clean-kernel notebook smoke
  test to `/tmp`, and a representative single-threaded performance benchmark.

## Assumptions

- Exact fractional delays are required; quantized delay caches are excluded.
- The optimization activates automatically from `n_channels == 1`.
- Circular shifts and the existing one-sided CSD weighting remain unchanged.
- Existing uncommitted notebook edits are preserved.
