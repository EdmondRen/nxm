# Optimize Fixed-Delay Fits and Correct Reduced Chi-Square

## Summary

Implement an automatic fixed-delay path that precomputes frequency-domain
sufficient statistics once per fit. Revise reduced chi-square to use local
effective degrees of freedom by default, while retaining a nominal rank-based
option for comparison.

## Fixed-Delay Optimization

- Activate the fast path automatically whenever `fix_delay=True`; preserve the
  existing general path when delays are fitted.
- At fixed delays, construct shifted mean templates $M$ and PCA basis $B$, then
  precompute $K=\operatorname{Re}(B^\dagger W B)$,
  $u=\operatorname{Re}(B^\dagger W x)$,
  $L=\operatorname{Re}(B^\dagger W M)$,
  $E=\operatorname{Re}(M^\dagger W M)$,
  $d=\operatorname{Re}(M^\dagger W x)$, and
  $c=\operatorname{Re}(x^\dagger W x)$, including the RFFT weights.
- For amplitudes $A$, repeat amplitudes across modes as $R$ and evaluate
  $H=I+\operatorname{diag}(R)K\operatorname{diag}(R)$,
  $g=R\odot(u-LA)$, $\hat z=H^{-1}g$,
  $Q_0=c-2d^TA+A^TEA$, $Q=Q_0-g^T\hat z$,
  $Q_{\rm prior}=\hat z^T\hat z$, and
  $Q_{\rm residual}=Q-Q_{\rm prior}$.
- Avoid constructing a model spectrum during optimizer evaluations. Handle
  zero PCA modes and preserve roundoff guards.

## Reduced Chi-Square

- Provide `degrees_of_freedom(result, method="effective")` and
  `reduced_chi_square(result, dof_method="effective")`, with a `"nominal"`
  alternative.
- Count real data constraints as $N_{\rm data}=J\sum_f w_f$.
- Record whether delays were fitted. Construct the local model Jacobian from
  amplitude, active delay, and PCA derivatives, excluding fixed or active-bound
  outer parameters.
- Form $G=J^\dagger WJ$ and prior precision $P$, then compute
  $T=(G+P)^+G$ and
  $\nu_{\rm eff}=N_{\rm data}-2\operatorname{tr}(T)+\operatorname{tr}(T^2)$.
- Define nominal residual DOF as
  $N_{\rm data}-\operatorname{rank}(G)$ and reject invalid nonpositive values.
- Document the local-linear approximation and the raw-RFFT CSD convention.

## Validation

- Compare the fast and reference profile results over correlated synthetic
  problems, fractional delays, and zero/nonzero PCA modes.
- Test partial/full bandwidth constraint counts, regularized effective DOF,
  degenerate Jacobians, fixed delays, and active bounds.
- Validate reduced chi-square on seeded independent noise ensembles.
- Benchmark a 12-channel, 3-mode, 4,000-bin fixed-delay problem, targeting at
  least a fivefold objective-evaluation speedup without making timing a flaky
  unit-test requirement.
- Run the syntax check and complete pytest suite.

## Assumptions

- Noise CSD input is raw NumPy RFFT covariance.
- Effective DOF is the default; nominal rank-based DOF remains available.
- Bound-active amplitude or delay parameters are locally fixed.
- Acceleration is internal and requires no caller changes.
