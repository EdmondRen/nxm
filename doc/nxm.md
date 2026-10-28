# Amplitude-dependent regularization for multi-channel, multi-template event reconstruction

Transition-edge-sensor (TES) pulse shapes vary with event position and energy
partition, while detector noise can be correlated across readout channels. A
single fixed template cannot represent all of this variation, but adding
independent templates without regularization makes a low-signal-to-noise-ratio
(SNR) fit sensitive to noise fluctuations. This note develops a reconstruction
model that separates the overall pulse amplitude from approximately
scale-invariant shape variation. Principal component analysis (PCA) describes
the distribution of the shape variation, and the full noise cross-spectral
density (CSD) accounts for correlations between channels.

The derivation proceeds from a single channel to a joint multi-channel fit.
Implementation conventions and practical usage are collected in the
appendices so that the main text remains focused on the statistical model.

## 1. Single-channel, multi-template reconstruction

A multi-template matched filter provides an empirical description of pulse
shape variation without requiring an analytic detector-response model. It is
therefore attractive near threshold and can be evaluated efficiently. The
important question is not whether PCA components can serve as templates, but
how the training pulses are normalized and how the corresponding coefficients
are constrained during reconstruction.

### 1.1 Motivation for a regularized formulation

The [earlier NxM formulation](https://arxiv.org/abs/2508.20090) fits a pulse as
a linear combination of empirically derived PCA templates,

$$
\mathbf x \approx \sum_{k=0}^{K} a_k\mathbf p_k+\boldsymbol\epsilon,
$$

where the coefficients $a_k$ are fitted independently. This approach is
effective when the pulse SNR is high enough to constrain all retained
coefficients, as demonstrated in that work. The present application extends
the same idea to a regime in which the reconstruction must remain stable at low SNR. Three
features then require additional care.

First, ordinary PCA describes an affine rather than a purely linear
reconstruction. For a training pulse $\mathbf x_i$, the rank-$K$ PCA
reconstruction is

$$
\widehat{\mathbf x}_i
=\overline{\mathbf x}
+P_KP_K^T(\mathbf x_i-\overline{\mathbf x})
=\overline{\mathbf x}
+\sum_{k=1}^{K}d_{ik}\mathbf p_k,
$$

where $\overline{\mathbf x}$ is the training-sample mean and the columns of
$P_K$ are the retained principal directions. Using only the principal
directions as linear templates is not, in general, the same truncated model:
the mean must either be included explicitly or already lie in the retained
template span. 

Second, PCA of unnormalized pulses captures amplitude variation as well as
shape variation. If

$$
\mathbf x_i=A_i(\mathbf s_0+\delta\mathbf s_i)+\boldsymbol\epsilon_i,
$$

then variation in $A_i$ contributes a covariance component along the mean
pulse $\mathbf s_0$. Correlations between $A_i$ and
$\delta\mathbf s_i$ also rotate the principal directions into mixtures of the
mean and the physical shape variations. Such a rotation does not change a
complete template span, but the directions that explain the most total
variance need not be the best low-rank description of the normalized pulse
manifold.

Third, an unconstrained fit does not use the PCA eigenvalues or any other
information about how frequently each shape occurs in the training
population. Every additional independent coefficient can follow a component
of the noise. Under a Gaussian-noise model, adding $K$ unconstrained
directions reduces the expected minimized chi-square by $K$ even when their
true coefficients are zero. This is the expected behavior of a more flexible
model, but it makes the coefficients and reconstructed pulse unstable when
the data do not constrain them.

Amplitude-normalized template generation addresses the affine-mean and
amplitude-mixing issues. A probabilistic model for the PCA coefficients then
regularizes shape directions according to their observed variation.

### 1.2 Pulse model for scale-invariant shape variation

Let $\mathbf x\in\mathbb R^M$ be a baseline-subtracted pulse with additive
zero-mean Gaussian noise $\boldsymbol\epsilon$ and covariance $N$. Define the
noise precision as $W=N^{-1}$. We assume that the pulse shape is approximately
independent of its overall scale over the reconstruction range; detector
saturation and other explicit energy-dependent changes are outside this
model.

After amplitude normalization, let $\mathbf s_0$ be the mean unit-amplitude
pulse and let $\mathbf s_k$ be the $k^{th}$ unit-norm PCA direction of the shape
residuals, with variance $\lambda_k$. The pulse model is

$$
\mathbf x
=A\left(\mathbf s_0+\sum_{k=1}^{K}c_k\mathbf s_k\right)
+\boldsymbol\epsilon,
\qquad
c_k\sim\mathcal N(0,\lambda_k),
$$

where $A\geq0$ is the pulse amplitude. Define standardized shape coefficients
and a variance-scaled template matrix by

$$
c_k=\sqrt{\lambda_k}\,z_k,
\qquad
\mathbf z\sim\mathcal N(\mathbf 0,I_K),
\qquad
\mathbf B=
\begin{pmatrix}
\sqrt{\lambda_1}\mathbf s_1 & \cdots &
\sqrt{\lambda_K}\mathbf s_K
\end{pmatrix}.
$$

The model becomes

$$
\boxed{
\mathbf x(A,\mathbf z)
=A(\mathbf s_0+\mathbf B\mathbf z)+\boldsymbol\epsilon,
\qquad
\mathbf z\sim\mathcal N(\mathbf 0,I_K).
}
$$

The physical shape perturbation $\mathbf B\mathbf z$ therefore scales with
the pulse amplitude. Equivalently, the prior width of the $k$th absolute shape
coefficient $Ac_k$ is $A\sqrt{\lambda_k}$, so the allowed absolute variation
contracts as $A$ decreases. This is the desired behavior near threshold.

### 1.3 Template generation for this pulse model

The training procedure must estimate the mean pulse and remove its amplitude
before PCA. Consider $n$ clean, high-SNR training pulses $\mathbf x_i$ that
have been baseline-subtracted and aligned to a common start time. Begin with a
mean-template estimate $\mathbf s_0^{(0)}$. At iteration $q$, estimate each
amplitude with the current template using the noise-weighted matched-filter
projection

$$
\widetilde A_i^{(q)}
=\frac{(\mathbf s_0^{(q)})^T W\mathbf x_i}
{(\mathbf s_0^{(q)})^T W\mathbf s_0^{(q)}}.
$$

Next form the mean of the amplitude-normalized pulses,

$$
\mathbf m^{(q)}
=\frac{1}{n}\sum_{i=1}^{n}
\frac{\mathbf x_i}{\widetilde A_i^{(q)}}.
$$

The decomposition into amplitude and template has an arbitrary overall scale.
Let $\gamma^{(q)}$ impose a fixed convention, such as unit peak height, and
update

$$
\mathbf s_0^{(q+1)}=\frac{\mathbf m^{(q)}}{\gamma^{(q)}},
\qquad
A_i^{(q+1)}=\gamma^{(q)}\widetilde A_i^{(q)}.
$$

With these paired updates, the average normalized residual is zero by
construction:

$$
\frac{1}{n}\sum_{i=1}^{n}
\left(
\frac{\mathbf x_i}{A_i^{(q+1)}}-\mathbf s_0^{(q+1)}
\right)=\mathbf 0.
$$

Re-estimating the amplitudes with the updated template perturbs this identity,
so the two steps are repeated until both the template and the average residual
are stable. In finite-precision data the final average residual is not exactly
zero, but it can be made negligible compared with the individual shape
residuals and detector noise.

Using the converged amplitudes and mean template, define

$$
\mathbf r_i=\frac{\mathbf x_i}{A_i}-\mathbf s_0.
$$

PCA is applied to the $\mathbf r_i$, not to the original pulses. Because their
mean is negligible, the PCA directions describe scale-independent departures
from $\mathbf s_0$ rather than compensating for a missing mean or a broad
amplitude distribution. The resulting directions $\mathbf s_k$ and variances
$\lambda_k$ define $\mathbf B$ in section 1.2. Practical event selection,
residual preprocessing, and convergence checks are described in Appendix A.

### 1.4 Maximum a posteriori objective

The negative twice log posterior, up to a parameter-independent constant, is

$$
Q(A,\mathbf z)
=\left[\mathbf x-A(\mathbf s_0+\mathbf B\mathbf z)\right]^T
W\left[\mathbf x-A(\mathbf s_0+\mathbf B\mathbf z)\right]
+\mathbf z^T\mathbf z.
$$

The first term is the noise-weighted residual and the second is the PCA prior.
The product $A\mathbf z$ makes the joint problem nonlinear, but the conditional
solution for $\mathbf z$ is analytic at fixed $A$:

$$
\boxed{
\widehat{\mathbf z}(A)
=\left(A^2\mathbf B^TW\mathbf B+I_K\right)^{-1}
A\mathbf B^TW(\mathbf x-A\mathbf s_0).
}
$$

Substituting this solution into $Q$ profiles out the shape coefficients and
leaves only the amplitude to be optimized numerically.

### 1.5 Unknown pulse start time

Let $t$ be a template delay in samples, with positive $t$ shifting the
template later in the trace. The mean template and every shape template must
receive the same shift. The pulse model is

$$
\mathbf x(A,t,\mathbf z)
=A\left[\mathbf s_0(t)+\mathbf B(t)\mathbf z\right]
+\boldsymbol\epsilon,
$$

and the conditional solution becomes

```math
\boxed{
\widehat{\mathbf z}(A,t)
=\left[A^2\mathbf B(t)^TW\mathbf B(t)+I_K\right]^{-1}
A\mathbf B(t)^TW\left[\mathbf x-A\mathbf s_0(t)\right].
}
```

The profiled objective

$$
Q_{\mathrm{prof}}(A,t)
=Q\left(A,t,\widehat{\mathbf z}(A,t)\right)
$$

is minimized over $A\geq0$ and an allowed interval in $t$. This construction
reduces directly to a regularized one-channel, multi-template matched filter.

## 2. Multi-channel, multi-template reconstruction

### 2.1 Joint time-domain model

For $J$ channels, let channel $j$ have data $\mathbf x_j$, amplitude
$A_j\geq0$, delay $t_j$, mean template $\mathbf s_{0,j}$, scaled PCA matrix
$\mathbf B_j$, and standardized coefficients $\mathbf z_j$. Stack the data and
shape coefficients as

```math
\mathbf x=
\begin{pmatrix}
\mathbf x_1\\ \vdots\\ \mathbf x_J
\end{pmatrix},
\qquad
\mathbf z=
\begin{pmatrix}
\mathbf z_1\\ \vdots\\ \mathbf z_J
\end{pmatrix}.
```

Define the mean signal and the amplitude-dependent shape matrix by

```math
\boldsymbol\mu(\mathbf A,\mathbf t)=
\begin{pmatrix}
A_1\mathbf s_{0,1}(t_1)\\ \vdots\\
A_J\mathbf s_{0,J}(t_J)
\end{pmatrix},
```

$$
D(\mathbf A,\mathbf t)
=\mathrm{blockdiag}\left[
A_1\mathbf B_1(t_1),\ldots,A_J\mathbf B_J(t_J)
\right].
$$

The full covariance $N\in\mathbb R^{JM\times JM}$ contains both within-channel
and cross-channel noise covariance. With $W=N^{-1}$ and independent
standard-normal PCA priors, the joint objective is

$$
Q(\mathbf A,\mathbf t,\mathbf z)
=\left(\mathbf x-\boldsymbol\mu-D\mathbf z\right)^T
W\left(\mathbf x-\boldsymbol\mu-D\mathbf z\right)
+\mathbf z^T\mathbf z.
$$

For fixed $\mathbf A$ and $\mathbf t$, the conditional solution is

$$
\boxed{
\widehat{\mathbf z}
=\left(D^TWD+I_{JK}\right)^{-1}
D^TW\left(\mathbf x-\boldsymbol\mu\right).
}
$$

The prior models the shape distribution of each channel independently. The
off-diagonal blocks of $W$ nevertheless couple all fitted channels through
their correlated noise. For $J=1$, this expression reduces to the
single-channel solution. If $W$ is block diagonal and all parameters and
priors are channel-specific, the objective separates into independent
single-channel fits.

### 2.2 Frequency-domain form for stationary noise

For stationary noise, the covariance is diagonal in frequency and dense only
across channels. At frequency bin $f_l$, collect the channel Fourier
coefficients in $\widetilde{\mathbf x}_l\in\mathbb C^J$ and let $S_l$ be the
$J\times J$ CSD matrix. A delay is applied through

$$
\widetilde{\mathbf s}_j(f_l;t_j)
=\widetilde{\mathbf s}_j(f_l)
e^{-2\pi i f_l t_j}.
$$

Let $\widetilde{\boldsymbol\mu}_l$ and $\widetilde D_l$ denote the
frequency-domain forms of $\boldsymbol\mu$ and $D$. For

$$
\widetilde{\mathbf r}_l
=\widetilde{\mathbf x}_l
-\widetilde{\boldsymbol\mu}_l
-\widetilde D_l\mathbf z,
$$

the objective is

$$
Q
=\sum_l \eta_l\,
\widetilde{\mathbf r}_l^\dagger S_l^{-1}
\widetilde{\mathbf r}_l
+\mathbf z^T\mathbf z,
$$

where $\eta_l$ contains the chosen Fourier and one-sided-spectrum
normalization. Equivalently, define

$$
H
=I_{JK}+\sum_l\eta_l\mathrm{Re}\left(
\widetilde D_l^\dagger S_l^{-1}\widetilde D_l
\right),
$$

$$
\mathbf g
=\sum_l\eta_l\mathrm{Re}\left[
\widetilde D_l^\dagger S_l^{-1}
(\widetilde{\mathbf x}_l-\widetilde{\boldsymbol\mu}_l)
\right].
$$

Then $\widehat{\mathbf z}=H^{-1}\mathbf g$. Data, templates, and CSDs must
use the same sampling, trace length, channel ordering, and Fourier convention.
The exact convention used by the accompanying implementation is specified in
Appendix B.

## 3. Interpretation and scope

The identity term in the profiled system is the mechanism that makes the
regularization amplitude-dependent. At high SNR, the data term
$A^2\mathbf B^TW\mathbf B$ dominates and the measured pulse can determine its
shape coefficients. As $A\rightarrow0$, the identity term dominates and
$\widehat{\mathbf z}\rightarrow\mathbf0$, so unsupported shape variation is
suppressed and the model approaches the mean-template fit. The same behavior
holds channel by channel in the joint model while the inverse CSD continues to
couple the channels through their noise.

This construction assumes that amplitude-normalized shape variations are
represented by the training population and are approximately independent of
amplitude. Saturation, energy-dependent time constants, nonstationary noise,
or new pulse populations require an expanded model or separately trained
templates; regularization cannot compensate for systematic template
mismatch.

## Appendix A. Practical template and fit workflow

### A.1 Training-pulse preparation

Select clean, high-SNR pulses that span the intended reconstruction range.
Sampling approximately uniformly in amplitude prevents densely populated
energy regions from dominating the learned shape distribution. Subtract each
channel baseline, estimate the pulse start time, and align all training pulses
to a common reference before estimating the mean template.

Iterate the amplitude and mean-template updates from section 1.3 until the
mean normalized residual is negligible. Events with failed timing or
amplitude estimates, amplitudes close to zero, or clear pileup should not enter
the PCA sample.

The current training notebook low-pass filters the residuals and multiplies
them by a smooth pulse-region window before PCA. This prevents baseline noise
far from the pulse and frequencies outside the detector bandwidth from
defining the leading modes. These operations are part of the template
definition and must be recorded with the artifact.

Fit PCA separately for each channel and retain a fixed, configurable number
of modes; the current workflow uses three. Store the mean template, unit-norm
PCA directions, PCA variances, template start time, sampling information,
filter and window definitions, trace length, and channel order.

### A.2 Noise CSD preparation

Estimate the CSD from synchronized noise-only traces with the same channel
order, sampling rate, and trace length as the signal data. Apply the same
baseline or detrending convention used for reconstruction. Average many noise
traces, verify Hermitian symmetry, and inspect the conditioned eigenvalue
spectrum for poorly estimated channel combinations.

### A.3 Initialization and profiled fit

Use a mean-template optimal filter to obtain initial amplitudes and integer
delays for every channel. The initializer treats channels separately for
robustness; the final objective uses the full inverse CSD. Clip initial
amplitudes to the nonnegative domain and restrict each final delay to a window
around its initial value so that weak channels cannot move to unrelated noise
fluctuations.

For every trial set $(\mathbf A,\mathbf t)$, construct
$\widetilde{\boldsymbol\mu}_l$ and $\widetilde D_l$, solve
$H\widehat{\mathbf z}=\mathbf g$, and evaluate
$Q_{\mathrm{prof}}=Q(\mathbf A,\mathbf t,\widehat{\mathbf z})$. Minimize the
profiled objective with $A_j\geq0$ and the configured delay bounds.

Record the fitted amplitudes, delays, standardized PCA coefficients, residual
and prior contributions, total objective, optimizer status, and active bounds.
Large $|z_k|$, structured whitened residuals, repeated boundary hits, or an
optimizer failure indicate template mismatch or an unreliable fit. Validate
the complete workflow with synthetic injections, noise-only traces, and
held-out measured pulses before using it for physics reconstruction.

## Appendix B. Spectral normalization and CSD conditioning

The implementation uses NumPy's unnormalized real FFT and one-sided CSD power
density with a normalized sampling frequency $f_s=1$. Let

$$
C_l
=\mathbb E\left[
\widetilde{\mathbf x}_l\widetilde{\mathbf x}_l^\dagger
\right]
$$

be the raw RFFT covariance for a trace of length $M$. Define the one-sided
multiplicity

$$
m_l=
\begin{cases}
1, & l=0,\\
1, & l=M/2 \text{ for even }M,\\
2, & \text{otherwise}.
\end{cases}
$$

The required CSD input is

$$
S_l=\frac{m_l}{M}C_l,
$$

and the likelihood multiplier used with $S_l^{-1}$ is

$$
\eta_l=\frac{m_l^2}{M}.
$$

An interior complex RFFT coefficient represents two real constraints, whereas
DC and Nyquist represent one each. Baseline-subtracted fits normally assign
zero likelihood weight to DC. Older `csd_matrices` artifacts in this repository
already contain the one-sided multiplicity but not the density normalization;
divide them by the trace length before reconstruction.

Before inversion, each estimated CSD is symmetrized as

$$
S_l\longleftarrow\frac{S_l+S_l^\dagger}{2}.
$$

If

$$
S_l=U_l\mathrm{diag}(d_{lq})U_l^\dagger,
$$

replace its eigenvalues by

$$
d_{lq}\longleftarrow
\max\left(d_{lq},\epsilon_{\mathrm{CSD}}d_{l,\max}\right),
\qquad \epsilon_{\mathrm{CSD}}>0.
$$

This floor prevents a poorly estimated low-variance channel combination from
dominating the fit. Reduced chi-square values near one require the same CSD
normalization, frequency selection, and DC convention as the likelihood.

## Appendix C. Numerical implementation and usage

The implementation is provided by `src/nxm.py`:

- `estimate_csd` estimates a Hermitian one-sided CSD density from synchronized
  noise traces.
- `NxMReconstructor` validates and stores the mean templates, PCA directions,
  PCA variances, CSD, channel order, selected frequency range, and CSD floor.
- `initialize` obtains nonnegative mean-template amplitude estimates and
  integer delays.
- `fit` profiles the PCA coefficients and optimizes the remaining amplitudes
  and, when requested, delays.
- `reduced_chi_square` evaluates the residual objective with either effective
  or nominal residual degrees of freedom.

For stationary single-channel noise, a common delay multiplies the mean and
all PCA spectra by the same unit-magnitude phase. The weighted template
products $\mathbf s_0^TW\mathbf s_0$, $\mathbf B^TW\mathbf B$, and
$\mathbf B^TW\mathbf s_0$ are therefore delay-independent. The reconstructor
caches these template terms and, for each trace, caches the weighted
trace-template spectra. A single phase vector then supplies the correlations
and their analytic delay derivatives at any fractional delay. Integer-delay
initialization evaluates all circular shifts with one inverse FFT.

When all delays are fixed, the implementation also precomputes the shifted
template, data, and inverse-CSD contractions. Trial amplitudes can then be
evaluated using only small channel-and-mode matrices. This cached objective is
algebraically identical to the direct frequency-bin calculation and provides
an analytic amplitude gradient.

With no explicit optimizer selection, fixed-delay fits and free-delay
single-channel fits use bounded L-BFGS-B with analytic gradients. Free-delay
multi-channel fits use Powell. An explicitly requested SciPy optimization
method is honored; analytic gradients are supplied when both the cached
objective and the selected method support them.

The returned fit result contains amplitudes, delays, standardized shape
coefficients, residual and prior objectives, total objective, convergence
status and message, evaluation counts, bound flags, and whether delays were
fitted. Degrees-of-freedom diagnostics are evaluated only when requested.

## Appendix D. Goodness-of-fit diagnostics

For the selected frequency bins, the number of real data constraints is

$$
N_{\mathrm{data}}=J\sum_l m_l,
$$

with zero contribution from an omitted DC bin. Parameters fixed by the user or
held at active bounds are excluded from the local active model.

The nominal residual degrees of freedom subtract the numerical rank of the
active local Jacobian from $N_{\mathrm{data}}$. The default effective
calculation accounts for the Gaussian PCA prior. If

$$
G=J_\theta^TWJ_\theta
$$

is the local parameter Gram matrix and $P$ is the diagonal prior precision,
with unit entries for standardized PCA coefficients and zero entries for
unregularized outer parameters, define

$$
T=(G+P)^+G.
$$

The effective residual degrees of freedom are

$$
\boxed{
\nu_{\mathrm{eff}}
=N_{\mathrm{data}}-2\mathrm{tr}(T)
+\mathrm{tr}(T^2).
}
$$

The reported reduced chi-square is the residual likelihood contribution,
excluding the PCA-prior penalty, divided by the selected residual degrees of
freedom. This is a local-linear diagnostic for a nonlinear fit. It cannot
correct a mismatched pulse model, a nonstationary noise distribution, or an
incorrectly normalized CSD, so it should be interpreted together with the
whitened residuals, standardized PCA coefficients, convergence status, and
parameter-bound flags.
