# Amplitude-dependent regularization for multi-channel, multi-template event reconstruction

TES pulse shapes vary with event position and energy partition, while detector noise is correlated across channels. This workflow models scale-invariant pulse-shape variation with principal components and handles channel correlations with the full noise cross-spectral density (CSD).


## 1. Single-channel, multi-template reconstruction

We first deal with pulse shape variation in a single channel by using a multi-template matched filter. The advantage over an analytical pulse shape fit is the robustness around threshold, and fast computing time.

Multi-template matched filter is well described in textbooks and we will skip the introduction. However, simply using PCA components as templates will give the wrong reconstruction. Cares must be taken in modeling the pulse and generating templates. An **unconstrained** fit works well at high signal-to-noise ratio [arxiv 2508.20090], but at low SNR the additional coefficients can follow noise fluctuations rather than genuine pulse-shape variations.


### 1.1 Why the previous method fails

The pulse model in the previous work is
$$
\mathbf x_i=
A_i\mathbf s_0+\delta\mathbf s_0 + \boldsymbol\epsilon
$$

where $\mathbf s_0$ is the average pulse shape, and $\delta\mathbf s_0$ describe pulse shape variation.
The fit model of the unconstrained workflow is 

$$
\mathbf x
\approx
\sum_{k=0}^K a_k\mathbf s_k,
$$
in which $\mathbf s_k$ are the pulse shape templates, $a_k$ are the amplitudes of each template. $\mathbf s_1,...\mathbf s_K$ are supposed to describe the pulse shape variation $\delta\mathbf s_0$. There are three major issues with this unconstrained fit workflow. 

First, directly using PCA components as templates is a misconception. Standard PCA minimizes $\sum_i\|\mathbf x_i-\hat{\mathbf x}_i\|^2$ and gives the optimal affine rank-$K$ reconstruction:

$$
\hat{\mathbf x}=
\bar{\mathbf x}
+
P_KP_K^T(\mathbf x-\bar{\mathbf x}) 
=\bar{\mathbf x}
+
\sum_{k=1}^K a_{ik}\mathbf s_k.
$$

$\bar{\mathbf x}$ is the mean value of training samples. Without including $\bar{\mathbf x}$ in the fit model, then that is not the normal PCA reconstruction at all. One consequence is that the principle components will need to account for the missing $\bar{\mathbf x}$



Second, the amplitude variation will mix the averaged pulse into the PCs.  If $A_i$ varies appreciably across the PCA training sample, direct PCA sees both amplitude variation and shape variation at once after subtracting $\bar{\mathbf x}$. The first PC will often be close to $\mathbf s_0$, but not exactly. If amplitude variation is correlated with pulse-shape variation, PCA produces rotated combinations such as So the first few PCs can easily be mixtures such as
$$
\mathbf p_1 \sim a_1\mathbf s_0+b_1\mathbf s_1+b_2\mathbf s_2, 
$$
$$
\mathbf p_2 \sim a_2\mathbf s_0+c_1\mathbf s_1+c_2\mathbf s_2,
$$
etc., where $\mathbf q_k$ describe pulse-shape variations. 

Mathematically, if we retain the full subspace, that rotation should not matter. But with only a few components, it can matter a lot because the optimal truncated subspace for total variance may not be the optimal truncated subspace for representing the pulse manifold locally.

Third, there is no constraints on the amplitudes of each template. This will cause sever overfiting when the SNR is low. Each extra template introduces an additional degree of freedom. Even when the true pulse has no shape variation, noise will generally project onto the extra templates. Under the Gaussian-noise model, adding $K$ linearly independent unconstrained templates reduces the expected minimized $\chi^2$ by $K$ relative to the model without those templates, even when their true coefficients are zero. 

A PCA basis identifies directions of variation in the training pulse population, but an unregularized fit does not use the PCA eigenvalues or any other information about **how probable** the corresponding coefficients are. 

The third point needs modification to the pulse model, while the first two problems can be resolved by modifying the template generation procedure based on the new pulse model. 

### 1.2 Pulse model for scale-invariant pulse shape variation


We made the following assumptions to simplify the problem:
1. Pulse shape fluctuation is scale invariant. We ignore the detector saturation effect for now.
2. Noise is stationary but can be colored.

Let $\mathbf x\in\mathbb R^M$ be a baseline-subtracted pulse. The additive noise $\boldsymbol\epsilon$ has zero mean and covariance $N$, with $W=N^{-1}$. For approximately scale-invariant shape fluctuations, model the pulse as

$$
\mathbf x
=A\left(\mathbf s_0+\sum_{k=1}^{K}c_k\mathbf s_k\right)
+\boldsymbol\epsilon,
\qquad c_k\sim\mathcal N(0,\lambda_k),
$$

where $\mathbf s_0$ is the mean unit-amplitude pulse, $A$ is its amplitude, $\mathbf s_k$ is the $k$-th unit-norm PCA direction, and $\lambda_k$ is its explained variance. Define standardized coefficients and a variance-scaled template matrix by

$$
c_k=\sqrt{\lambda_k}\,z_k,
\qquad z_k\sim\mathcal N(0,1),
\qquad
\mathbf B=\left(\sqrt{\lambda_1}\mathbf s_1,\ldots,
\sqrt{\lambda_K}\mathbf s_K\right).
$$

The model becomes

$$
\boxed{
\mathbf x(A,\mathbf z)
=A\left(\mathbf s_0+\mathbf B\mathbf z\right)
+\boldsymbol\epsilon,
\qquad \mathbf z\sim\mathcal N(\mathbf 0,I_K).
}
$$

### 1.3 Template generation for this pulse model

PCA must therefore be applied to amplitude-normalized residuals, not directly to unnormalized pulses. Otherwise the leading components mostly reproduce amplitude variation.

### 1.4 Maximum a posteriori (MAP) objective

The negative twice log posterior, up to a constant, is

$$
Q(A,\mathbf z)
=\left[\mathbf x-A(\mathbf s_0+\mathbf B\mathbf z)\right]^T
W\left[\mathbf x-A(\mathbf s_0+\mathbf B\mathbf z)\right]
+\mathbf z^T\mathbf z.
$$

The last term is the PCA prior. It prevents weakly constrained shape components from absorbing noise. The product $A\mathbf z$ makes the full problem nonlinear, thus there is no analytic solution for both A and $\mathbf z$. But for fixed $A$ the conditional solution is analytic:

$$
\widehat{\mathbf z}(A)
=\left(A^2\mathbf B^TW\mathbf B+I_K\right)^{-1}
A\mathbf B^TW(\mathbf x-A\mathbf s_0).
$$

### 1.5 Pulse start time

The above derivation assumes that templates and pulse data are already aligned in time. But in reality we don't know the pulse starting time.

Let $\mathbf s_0(t_0)$ and $\mathbf B(t_0)$ denote the mean and PCA templates shifted by $t_0$. Both terms must receive the same shift. Then the conditional solution is

$$
\boxed{
\widehat{\mathbf z}(A,t_0)
=\left[A^2\mathbf B(t_0)^TW\mathbf B(t_0)+I_K\right]^{-1}
A\mathbf B(t_0)^TW
\left[\mathbf x-A\mathbf s_0(t_0)\right].
}
$$

Substitution gives the profiled objective

$$
Q_{\mathrm{prof}}(A,t_0)
=Q\left(A,t_0,\widehat{\mathbf z}(A,t_0)\right),
$$

which is minimized numerically over $A\geq0$ and an allowed interval in $t_0$.

### 1.4 Single-channel sufficient statistics

For stationary single-channel noise, shifting the mean and every PCA template
by the same delay multiplies them by the same unit-magnitude Fourier phase.
Consequently, the weighted template products
$\mathbf s_0^T W\mathbf s_0$, $\mathbf B^T W\mathbf B$, and
$\mathbf B^T W\mathbf s_0$ do not depend on delay. The implementation caches
these quantities when constructing a one-channel reconstructor. For each trace
it then caches only the weighted trace-template spectra; one phase vector gives
the correlations and their exact derivatives at any fractional delay.

This avoids rebuilding frequency-summed matrices during every objective
evaluation without discretizing the delay. Free-delay one-channel fits select
L-BFGS-B and the analytic amplitude and delay gradients when `method=None`.
An explicitly selected optimizer is still honored. Integer-delay matched-filter
initialization obtains all circular delay responses with one inverse FFT.

## 2. Multi-channel, multi-template reconstruction

### 2.1 Joint time-domain model

For $J$ channels, let channel $j$ have data $\mathbf x_j$, amplitude $A_j\geq0$, delay $t_j$, mean template $\mathbf s_{0,j}$, PCA matrix $\mathbf B_j$, and standardized coefficients $\mathbf z_j$. Stack the channel data and coefficients as

```math
\mathbf x=
\begin{pmatrix}\mathbf x_1\\ \vdots\\ \mathbf x_J\end{pmatrix},
\qquad
\mathbf z=
\begin{pmatrix}\mathbf z_1\\ \vdots\\ \mathbf z_J\end{pmatrix}.
```

Define

```math
\boldsymbol\mu(\mathbf A,\mathbf t)=
\begin{pmatrix}
A_1\mathbf s_{0,1}(t_1)\\ \vdots\\
A_J\mathbf s_{0,J}(t_J)
\end{pmatrix},
\qquad
D(\mathbf A,\mathbf t)=
\mathrm{blockdiag}\left[
A_1\mathbf B_1(t_1),\ldots,A_J\mathbf B_J(t_J)
\right].
```

The full noise covariance $N\in\mathbb R^{JM\times JM}$ contains both within-channel and cross-channel covariance. With $W=N^{-1}$ and independent standard-normal PCA priors, the joint objective is

$$
Q(\mathbf A,\mathbf t,\mathbf z)
=\left(\mathbf x-\boldsymbol\mu-D\mathbf z\right)^T
W\left(\mathbf x-\boldsymbol\mu-D\mathbf z\right)
+\mathbf z^T\mathbf z.
$$

For fixed $\mathbf A$ and $\mathbf t$,

$$
\boxed{
\widehat{\mathbf z}
=\left(D^TWD+I_{JK}\right)^{-1}
D^TW\left(\mathbf x-\boldsymbol\mu\right).
}
$$

The prior treats shape variation in each channel independently. Off-diagonal blocks of $W$ still couple the fitted channels through their correlated noise.

### 2.2 Frequency-domain form

For stationary noise, evaluate the objective efficiently in the frequency domain. At frequency bin $f_l$, collect the channel Fourier coefficients in $\widetilde{\mathbf x}_l\in\mathbb C^J$ and let $S_l$ be the $J\times J$ noise CSD matrix. A delay is applied through

$$
\widetilde{\mathbf s}_j(f_l;t_j)
=\widetilde{\mathbf s}_j(f_l)e^{-2\pi i f_l t_j}.
$$

Let $\widetilde{\boldsymbol\mu}_l$ and $\widetilde D_l$ be the frequency-domain forms of $\boldsymbol\mu$ and $D$. For residual

$$
\widetilde{\mathbf r}_l
=\widetilde{\mathbf x}_l-\widetilde{\boldsymbol\mu}_l
-\widetilde D_l\mathbf z,
$$

the objective is

$$
Q=\sum_l w_l\
\widetilde{\mathbf r}_l^\dagger S_l^{-1}
\widetilde{\mathbf r}_l+\mathbf z^T\mathbf z,
$$

where $w_l$ contains the chosen FFT and one-sided-spectrum normalization. Equivalently,

```math
H=I_{JK}+\sum_l w_l\mathrm{Re}
\left(\widetilde D_l^\dagger S_l^{-1}\widetilde D_l\right),
\qquad
\mathbf g=\sum_l w_l\mathrm{Re}
\left[\widetilde D_l^\dagger S_l^{-1}
(\widetilde{\mathbf x}_l-\widetilde{\boldsymbol\mu}_l)\right],
```

and $\widehat{\mathbf z}=H^{-1}\mathbf g$. Data, templates, and CSDs must use identical FFT normalization, sampling rate, trace length, and channel ordering. After baseline subtraction, omit the DC bin.

The implementation in `src/nxm.py` uses one-sided CSD power density at sampling
frequency $f_s=1$. If
$C_l=\mathbb E[\widetilde{\mathbf x}_l\widetilde{\mathbf x}_l^\dagger]$
is the raw NumPy RFFT covariance and $m_l$ is 2 for an interior bin and 1 for
DC or Nyquist, the required input is

$$
S_l=\frac{m_l}{N}C_l.
$$

Thus the likelihood multiplier used with $S_l^{-1}$ is $m_l^2/N$. DC receives
zero likelihood weight when omitted. The real-data DOF count still uses
$m_l$, because an interior complex coefficient represents two real
constraints. Delays are fractional sample offsets relative to the unshifted
templates.

Legacy raw-covariance inputs must be multiplied by $m_l/N$. The repository's
older `csd_matrices` artifacts already contain the one-sided multiplicity but
not the density normalization, so divide those arrays by the trace length.

For a result returned by `fit`, call
`reconstructor.reduced_chi_square(result)`. The implementation derives the
number of real constraints from the frequency bins actually selected: an
interior RFFT bin counts twice, DC and Nyquist count once, and the count is
multiplied by the number of channels. It excludes fixed delays and outer
parameters at active bounds from the local model Jacobian.

The default effective degrees of freedom account for the Gaussian PCA prior.
If $G=J_\theta^T WJ_\theta$ is the local parameter Gram matrix and $P$ has unit
entries for standardized PCA coefficients and zero entries for unregularized
outer parameters, define

$$
T=(G+P)^+G,
\qquad
\nu_{\mathrm{eff}}
=N_{\mathrm{data}}-2\mathrm{tr}(T)+\mathrm{tr}(T^2).
$$

Then `reduced_chi_square(result)` returns
$Q_{\mathrm{residual}}/\nu_{\mathrm{eff}}$. This is a local-linear effective
DOF calculation for the nonlinear fit. For comparison,
`reduced_chi_square(result, dof_method="nominal")` subtracts the numerical rank
of the active Jacobian instead. Neither convention can compensate for an
incorrectly normalized CSD: values near one require the one-sided density
convention returned by `estimate_csd`.

Before inversion, symmetrize each estimated CSD as $(S_l+S_l^\dagger)/2$. If $S_l=U\mathrm{diag}(d_q)U^\dagger$, replace

$$
d_q\longrightarrow\max(d_q,\epsilon_{\mathrm{CSD}}d_{\max})
$$

with a configured floor $\epsilon_{\mathrm{CSD}}>0$. This prevents poorly estimated noise modes from dominating the fit.

## 3. Processing procedure

### 3.1 Template generation

1. Select clean, high-signal-to-noise pulses spanning the reconstruction range. Sampling approximately uniformly in amplitude prevents dense energy regions from dominating the PCA.
2. Subtract each channel baseline, estimate its arrival time, and align the training pulses.
3. Build a mean template and estimate each pulse amplitude with a single-template optimal filter. Alternate these two steps until the mean normalized residual is negligible.
4. Form residuals $\mathbf r=\mathbf x/A-\mathbf s_0$. Apply the configured low-pass filter and a smooth pulse-region window so baseline noise does not define the PCA modes.
5. Fit PCA separately for each channel. Retain a fixed, configurable number $K$ of modes; the current default is $K=3$. Save $\mathbf s_0$, $\mathbf s_k$, $\lambda_k$, timing, sampling, filtering, and channel-order metadata.

### 3.2 Noise CSD generation

Use synchronized noise-only traces with the same sampling rate and length as the signal traces. Apply the same baseline or detrending convention, compute each auto- and cross-spectrum, and average over many traces. Check that every $S_l$ is Hermitian and positive definite after eigenvalue flooring.

### 3.3 Initial amplitudes and delays

Run a linear mean-template optimal filter to obtain initial $A_j$ and $t_j$ for every channel. Clip the starting amplitudes to the nonnegative domain. Restrict each $t_j$ to a configured window around its initial estimate; this prevents low-signal-to-noise channels from moving to unrelated noise fluctuations. The linear multi-template fit may also be used as an initializer, but not as the final estimator because it lacks the PCA prior and scale-invariant parameterization.

### 3.4 Joint profiled fit

For each trial $`(\mathbf A,\mathbf t)`$, construct $`\widetilde{\boldsymbol\mu}_l`$ and $`\widetilde D_l`$, solve $`H\widehat{\mathbf z}=\mathbf g`$, and evaluate $`Q_{\mathrm{prof}}=Q(\mathbf A,\mathbf t,\widehat{\mathbf z})`$. Minimize this objective with bounds $`A_j\geq 0`$ and the configured timing intervals.


When delays are fixed, their phases and all frequency-summed template, data,
and CSD contractions are constant during the outer fit. The implementation
precomputes these sufficient statistics once and evaluates subsequent trial
amplitudes using only small channel/mode matrices. This is algebraically
equivalent to the frequency-bin calculation; caching shifted templates alone
would leave the dominant frequency contractions inside every objective call.
The cached objective also supplies an analytic amplitude gradient. Unless an
optimizer is selected explicitly, fixed-delay fits use bounded L-BFGS-B while
fits with free delays continue to use Powell. Degrees-of-freedom diagnostics
are computed lazily so their local-Jacobian contraction does not slow fits that
do not request reduced chi-square.

Store $\widehat{\mathbf A}$, $\widehat{\mathbf t}$, $\widehat{\mathbf z}$, the noise-residual term, the PCA-prior term, the total objective, optimizer status, and any active bounds. Large $|z_k|$, structured whitened residuals, or repeated boundary hits indicate template mismatch or a failed fit. Validate the workflow with held-out pulses, noise-only traces, and simulated pulse injections before applying it to physics data.


