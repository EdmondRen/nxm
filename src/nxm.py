"""Frequency-domain multi-channel, multi-template pulse reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, signal


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class NxMProfileResult:
    """Result of profiling the PCA coefficients at fixed amplitudes and delays.

    Attributes:
        shape_coefficients: Standardized PCA coefficients with shape
            ``(n_channels, n_modes)``.
        residual_objective: CSD-weighted residual contribution to the
            negative twice log posterior.
        prior_objective: Standard-normal PCA prior contribution.
        objective: Sum of the residual and prior contributions.
    """

    shape_coefficients: FloatArray
    residual_objective: float
    prior_objective: float
    objective: float


@dataclass(frozen=True)
class NxMFitResult:
    """Result of a profiled multi-channel pulse fit.

    Attributes:
        amplitudes: Nonnegative amplitudes, one per channel.
        delays: Per-channel template shifts in samples.
        shape_coefficients: Standardized PCA coefficients with shape
            ``(n_channels, n_modes)``.
        residual_objective: CSD-weighted residual contribution.
        prior_objective: Standard-normal PCA prior contribution.
        objective: Total profiled objective.
        success: Whether the outer optimizer reported convergence.
        message: Optimizer termination message.
        nfev: Number of profiled objective evaluations.
        nit: Number of outer optimizer iterations, when reported.
        amplitude_at_bound: Flags for amplitudes at their zero lower bound.
        delay_at_bound: Flags for delays at either configured bound.
    """

    amplitudes: FloatArray
    delays: FloatArray
    shape_coefficients: FloatArray
    residual_objective: float
    prior_objective: float
    objective: float
    success: bool
    message: str
    nfev: int
    nit: int
    amplitude_at_bound: NDArray[np.bool_]
    delay_at_bound: NDArray[np.bool_]


def estimate_csd(
    noise_traces: ArrayLike,
    detrend: str | bool | None = "constant",
) -> ComplexArray:
    """Estimate raw RFFT covariance from synchronized noise traces.

    The returned matrix uses the convention
    ``S[f, j, k] = mean(X[event, j, f] * conj(X[event, k, f]))``.
    It is an FFT-coefficient covariance, not a power spectral density.

    Args:
        noise_traces: Real array with shape
            ``(n_events, n_channels, n_samples)``.
        detrend: ``"constant"``, ``"linear"``, or ``None``/``False``.

    Returns:
        Complex CSD array with shape
        ``(n_samples // 2 + 1, n_channels, n_channels)``.

    Raises:
        ValueError: If the input shape, values, or detrending mode is invalid.
    """

    traces = np.asarray(noise_traces, dtype=np.float64)
    if traces.ndim != 3:
        raise ValueError(
            "noise_traces must have shape (n_events, n_channels, n_samples)"
        )
    if min(traces.shape) <= 0:
        raise ValueError("noise_traces dimensions must all be positive")
    if not np.all(np.isfinite(traces)):
        raise ValueError("noise_traces must contain only finite values")

    if detrend is False or detrend is None:
        prepared = traces
    elif detrend in {"constant", "linear"}:
        prepared = signal.detrend(traces, axis=-1, type=str(detrend))
    else:
        raise ValueError("detrend must be 'constant', 'linear', False, or None")

    spectra = np.fft.rfft(prepared, axis=-1)
    csd = np.einsum(
        "ejf,ekf->fjk",
        spectra,
        spectra.conj(),
        optimize=True,
    ) / traces.shape[0]
    # Remove roundoff-level violations of Hermitian symmetry.
    return np.asarray(
        0.5 * (csd + np.swapaxes(csd.conj(), -1, -2)),
        dtype=np.complex128,
    )


class NxMReconstructor:
    """Regularized multi-channel, multi-template frequency-domain fitter.

    PCA components are scaled by the square root of their variances, so the
    fitted shape coefficients have independent standard-normal priors. Noise
    CSDs must contain raw NumPy RFFT-coefficient covariance rather than power
    density.

    Args:
        mean_templates: Mean pulse templates with shape
            ``(n_channels, n_samples)``.
        pca_components: Unit-norm PCA directions with shape
            ``(n_channels, n_modes, n_samples)``.
        pca_variances: PCA variances with shape
            ``(n_channels, n_modes)``.
        noise_csd: Raw RFFT covariance with shape
            ``(n_frequencies, n_channels, n_channels)``.
        channel_names: Optional unique names in array channel order.
        frequency_limit: Number of lowest RFFT bins to use. ``None`` uses all
            bins available in both templates and the CSD.
        csd_eigenvalue_floor: Minimum CSD eigenvalue as a fraction of the
            largest eigenvalue in each frequency bin.
        omit_dc: Whether to give the DC bin zero weight. This should normally
            remain enabled for baseline-subtracted traces.

    Raises:
        ValueError: If shapes, values, or configuration are inconsistent.
    """

    def __init__(
        self,
        mean_templates: ArrayLike,
        pca_components: ArrayLike,
        pca_variances: ArrayLike,
        noise_csd: ArrayLike,
        *,
        channel_names: Sequence[str] | None = None,
        frequency_limit: int | None = None,
        csd_eigenvalue_floor: float = 1e-6,
        omit_dc: bool = True,
    ) -> None:
        means = np.asarray(mean_templates, dtype=np.float64)
        components = np.asarray(pca_components, dtype=np.float64)
        variances = np.asarray(pca_variances, dtype=np.float64)
        csd = np.asarray(noise_csd, dtype=np.complex128)

        if means.ndim != 2:
            raise ValueError("mean_templates must have shape (n_channels, n_samples)")
        if components.ndim != 3:
            raise ValueError(
                "pca_components must have shape (n_channels, n_modes, n_samples)"
            )
        if variances.ndim != 2:
            raise ValueError(
                "pca_variances must have shape (n_channels, n_modes)"
            )
        if csd.ndim != 3 or csd.shape[1] != csd.shape[2]:
            raise ValueError(
                "noise_csd must have shape (n_frequencies, n_channels, n_channels)"
            )

        self.n_channels, self.n_samples = means.shape
        if self.n_channels == 0 or self.n_samples < 2:
            raise ValueError("at least one channel and two samples are required")
        if components.shape[0] != self.n_channels:
            raise ValueError("PCA component channel count does not match templates")
        self.n_modes = components.shape[1]
        if components.shape[2] != self.n_samples:
            raise ValueError("PCA component sample count does not match templates")
        if variances.shape != (self.n_channels, self.n_modes):
            raise ValueError("PCA variance shape does not match PCA components")
        if csd.shape[1:] != (self.n_channels, self.n_channels):
            raise ValueError("CSD channel count does not match templates")
        if not all(
            np.all(np.isfinite(array)) for array in (means, components, variances)
        ) or not np.all(np.isfinite(csd)):
            raise ValueError("templates, PCA data, and CSD must be finite")
        if np.any(variances < 0.0):
            raise ValueError("pca_variances must be nonnegative")
        if not np.isfinite(csd_eigenvalue_floor) or csd_eigenvalue_floor <= 0.0:
            raise ValueError("csd_eigenvalue_floor must be finite and positive")

        if channel_names is None:
            names = tuple(f"channel_{index}" for index in range(self.n_channels))
        else:
            names = tuple(str(name) for name in channel_names)
            if len(names) != self.n_channels:
                raise ValueError("channel_names length does not match templates")
            if len(set(names)) != len(names):
                raise ValueError("channel_names must be unique")
        self.channel_names = names

        full_frequency_count = self.n_samples // 2 + 1
        available_frequency_count = min(full_frequency_count, csd.shape[0])
        if frequency_limit is None:
            selected_frequency_count = available_frequency_count
        elif isinstance(frequency_limit, (int, np.integer)):
            selected_frequency_count = int(frequency_limit)
            if not 1 <= selected_frequency_count <= available_frequency_count:
                raise ValueError(
                    "frequency_limit must be between 1 and the available RFFT bins"
                )
        else:
            raise ValueError("frequency_limit must be an integer or None")
        if omit_dc and selected_frequency_count < 2:
            raise ValueError(
                "frequency_limit must include a non-DC bin when omit_dc is True"
            )

        self.frequency_limit = selected_frequency_count
        self.csd_eigenvalue_floor = float(csd_eigenvalue_floor)
        self.omit_dc = bool(omit_dc)
        self._full_frequency_count = full_frequency_count
        self._frequency_indices = np.arange(selected_frequency_count, dtype=float)
        self._angular_frequencies = (
            2.0 * np.pi * self._frequency_indices / self.n_samples
        )

        self._weights = np.full(selected_frequency_count, 2.0, dtype=float)
        if not self.omit_dc:
            self._weights[0] = 1.0
        else:
            self._weights[0] = 0.0
        if (
            self.n_samples % 2 == 0
            and selected_frequency_count == full_frequency_count
        ):
            self._weights[-1] = 1.0

        self.mean_templates = means.copy()
        self.pca_components = components.copy()
        self.pca_variances = variances.copy()
        self.scaled_pca_templates = components * np.sqrt(variances)[:, :, None]

        self._mean_fft_full = np.fft.rfft(self.mean_templates, axis=-1).T
        basis_fft = np.fft.rfft(self.scaled_pca_templates, axis=-1)
        self._basis_fft_full = np.transpose(basis_fft, (2, 0, 1))
        self._mean_fft = self._mean_fft_full[:selected_frequency_count]
        self._basis_fft = self._basis_fft_full[:selected_frequency_count]

        selected_csd = csd[:selected_frequency_count].copy()
        selected_csd = 0.5 * (
            selected_csd + np.swapaxes(selected_csd.conj(), -1, -2)
        )
        self.noise_csd, self.inverse_noise_csd = self._condition_csd(selected_csd)

    def _condition_csd(self, csd: ComplexArray) -> tuple[ComplexArray, ComplexArray]:
        """Condition and invert every frequency bin used by the likelihood."""

        conditioned = np.zeros_like(csd)
        inverse = np.zeros_like(csd)
        start = 1 if self.omit_dc else 0

        if start == 1:
            conditioned[0] = csd[0]
        if start >= len(csd):
            return conditioned, inverse

        eigenvalues, eigenvectors = np.linalg.eigh(csd[start:])
        largest = eigenvalues[:, -1]
        if np.any(~np.isfinite(largest)) or np.any(largest <= 0.0):
            raise ValueError("each fitted CSD bin must have a positive eigenvalue")
        floors = self.csd_eigenvalue_floor * largest[:, None]
        clipped = np.maximum(eigenvalues, floors)
        eigenvectors_h = np.swapaxes(eigenvectors.conj(), -1, -2)
        conditioned[start:] = (
            eigenvectors * clipped[:, None, :]
        ) @ eigenvectors_h
        inverse[start:] = (
            eigenvectors * (1.0 / clipped)[:, None, :]
        ) @ eigenvectors_h
        return conditioned, inverse

    def _validate_traces(self, traces: ArrayLike) -> FloatArray:
        """Return validated time-domain traces."""

        values = np.asarray(traces, dtype=np.float64)
        if values.shape != (self.n_channels, self.n_samples):
            raise ValueError(
                "traces must have shape "
                f"({self.n_channels}, {self.n_samples})"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("traces must contain only finite values")
        return values

    def _validate_parameters(
        self,
        amplitudes: ArrayLike,
        delays: ArrayLike,
    ) -> tuple[FloatArray, FloatArray]:
        """Return validated amplitudes and delays."""

        amplitude_values = np.asarray(amplitudes, dtype=np.float64)
        delay_values = np.asarray(delays, dtype=np.float64)
        expected_shape = (self.n_channels,)
        if amplitude_values.shape != expected_shape:
            raise ValueError(f"amplitudes must have shape {expected_shape}")
        if delay_values.shape != expected_shape:
            raise ValueError(f"delays must have shape {expected_shape}")
        if not np.all(np.isfinite(amplitude_values)) or not np.all(
            np.isfinite(delay_values)
        ):
            raise ValueError("amplitudes and delays must be finite")
        # if np.any(amplitude_values < 0.0):
        #     raise ValueError("amplitudes must be nonnegative")
        return amplitude_values, delay_values

    def _normalize_delay_bounds(
        self,
        delay_bounds: ArrayLike,
    ) -> FloatArray:
        """Broadcast and validate delay bounds to channel-by-two shape."""

        bounds = np.asarray(delay_bounds, dtype=np.float64)
        if bounds.shape == (2,):
            bounds = np.broadcast_to(bounds, (self.n_channels, 2)).copy()
        elif bounds.shape != (self.n_channels, 2):
            raise ValueError(
                "delay bounds must have shape (2,) or (n_channels, 2)"
            )
        if not np.all(np.isfinite(bounds)):
            raise ValueError("delay bounds must be finite")
        if np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError("each delay lower bound must not exceed its upper bound")
        return bounds

    def _phase(self, delays: FloatArray, *, full: bool = False) -> ComplexArray:
        """Return per-frequency, per-channel shift phases."""

        if full:
            angular = (
                2.0 * np.pi
                * np.arange(self._full_frequency_count, dtype=float)
                / self.n_samples
            )
        else:
            angular = self._angular_frequencies
        return np.exp(-1j * angular[:, None] * delays[None, :])

    def _profile_fft(
        self,
        trace_fft: ComplexArray,
        amplitudes: FloatArray,
        delays: FloatArray,
        debug = False
    ) -> tuple[NxMProfileResult, ComplexArray]:
        """
        Profile shape coefficients for precomputed trace spectra.
        
        trace_fft: shape (n_frequencies, n_channels)
        amplitudes: shape (n_channels,)
        delays: shape (n_channels,)
        
        """

        phase = self._phase(delays)
        shifted_mean = self._mean_fft * phase
        residual_without_shape = trace_fft - amplitudes[None, :] * shifted_mean
        if debug:
            import matplotlib.pyplot as plt
            plt.plot(np.fft.irfft(residual_without_shape.T[0]))
            plt.plot(np.fft.irfft(trace_fft.T[0]))
            plt.plot(np.fft.irfft((amplitudes[None, :] * shifted_mean).T[0]))
            plt.show()

        if self.n_modes:
            shifted_basis = self._basis_fft * phase[:, :, None]
            design = shifted_basis * amplitudes[None, :, None]
            hessian_blocks = np.einsum(
                "f,fjk,fjm,fml->jkml",
                self._weights,
                design.conj(),
                self.inverse_noise_csd,
                design,
                optimize=True,
            ).real
            hessian = hessian_blocks.reshape(
                self.n_channels * self.n_modes,
                self.n_channels * self.n_modes,
            )
            hessian += np.eye(hessian.shape[0])
            gradient = np.einsum(
                "f,fjk,fjm,fm->jk",
                self._weights,
                design.conj(),
                self.inverse_noise_csd,
                residual_without_shape,
                optimize=True,
            ).real
            coefficients = np.linalg.solve(hessian, gradient.ravel()).reshape(
                self.n_channels, self.n_modes
            )
            fitted_shape = np.einsum(
                "fjk,jk->fj", shifted_basis, coefficients, optimize=True
            )
            model_fft = amplitudes[None, :] * (shifted_mean + fitted_shape)
        else:
            coefficients = np.empty((self.n_channels, 0), dtype=float)
            model_fft = amplitudes[None, :] * shifted_mean

        residual = trace_fft - model_fft
        weighted_residual = np.einsum(
            "fjm,fm->fj",
            self.inverse_noise_csd,
            residual,
            optimize=True,
        )
        residual_objective = float(
            np.sum(
                self._weights[:, None]
                * np.real(residual.conj() * weighted_residual)
            )
        )
        # Guard only against roundoff; a material negative value signals an
        # invalid covariance and should not be hidden.
        if residual_objective < 0.0 and residual_objective > -1e-9:
            residual_objective = 0.0
        prior_objective = float(np.sum(coefficients**2))
        profile = NxMProfileResult(
            shape_coefficients=coefficients,
            residual_objective=residual_objective,
            prior_objective=prior_objective,
            objective=residual_objective + prior_objective,
        )
        return profile, model_fft

    def profile(
        self,
        traces: ArrayLike,
        amplitudes: ArrayLike,
        delays: ArrayLike,
    ) -> NxMProfileResult:
        """Profile PCA coefficients at fixed channel amplitudes and delays.

        Args:
            traces: Baseline-subtracted traces with shape
                ``(n_channels, n_samples)``.
            amplitudes: Nonnegative channel amplitudes.
            delays: Channel template shifts in samples.

        Returns:
            Profiled coefficients and objective components.

        Raises:
            ValueError: If traces or parameters are invalid.
        """

        trace_values = self._validate_traces(traces)
        amplitude_values, delay_values = self._validate_parameters(
            amplitudes, delays
        )
        trace_fft = np.fft.rfft(trace_values, axis=-1).T[: self.frequency_limit]
        profile, _ = self._profile_fft(trace_fft, amplitude_values, delay_values)
        return profile

    def initialize(
        self,
        traces: ArrayLike,
        delay_bounds: ArrayLike = (-150.0, 150.0),
    ) -> tuple[FloatArray, FloatArray]:
        """Estimate amplitudes and integer delays with mean-template filters.

        Cross-channel CSD terms are intentionally omitted during this robust
        initialization. The final fit uses the complete inverse CSD.

        Args:
            traces: Baseline-subtracted traces with shape
                ``(n_channels, n_samples)``.
            delay_bounds: A shared ``(lower, upper)`` pair or one pair per
                channel, measured in samples.

        Returns:
            Tuple of nonnegative amplitudes and integer-valued delay estimates.

        Raises:
            ValueError: If traces or bounds are invalid.
        """

        trace_values = self._validate_traces(traces)
        bounds = self._normalize_delay_bounds(delay_bounds)
        trace_fft = np.fft.rfft(trace_values, axis=-1).T[: self.frequency_limit]
        amplitudes = np.zeros(self.n_channels, dtype=float)
        delays = np.zeros(self.n_channels, dtype=float)

        for channel in range(self.n_channels):
            lower, upper = bounds[channel]
            integer_delays = np.arange(np.ceil(lower), np.floor(upper) + 1.0)
            if integer_delays.size == 0:
                integer_delays = np.array([(lower + upper) / 2.0])
            phase = np.exp(
                -1j
                * self._angular_frequencies[:, None]
                * integer_delays[None, :]
            )
            shifted_template = self._mean_fft[:, channel, None] * phase
            diagonal_csd = self.noise_csd[:, channel, channel].real
            precision = np.zeros_like(diagonal_csd)
            active = self._weights > 0.0
            precision[active] = 1.0 / diagonal_csd[active]
            weighted = self._weights * precision
            denominator = float(
                np.sum(weighted * np.abs(self._mean_fft[:, channel]) ** 2)
            )
            if not np.isfinite(denominator) or denominator <= 0.0:
                raise ValueError(
                    f"mean template for channel {channel} has zero weighted norm"
                )
            responses = np.sum(
                weighted[:, None]
                * np.real(shifted_template.conj() * trace_fft[:, channel, None]),
                axis=0,
            )
            positive_responses = np.maximum(responses, 0.0)
            scores = positive_responses**2 / denominator
            best = int(np.argmax(scores))
            if scores[best] == 0.0:
                target = np.clip(0.0, lower, upper)
                best = int(np.argmin(np.abs(integer_delays - target)))
            amplitudes[channel] = positive_responses[best] / denominator
            delays[channel] = integer_delays[best]

        return amplitudes, delays

    def fit(
        self,
        traces: ArrayLike,
        *,
        fix_delay: bool = False,
        initial_amplitudes: ArrayLike | None = None,
        initial_delays: ArrayLike | None = None,
        initialization_bounds: ArrayLike = (-150.0, 150.0),
        delay_bounds: ArrayLike | None = None,
        delay_half_width: float | ArrayLike = 10.0,
        optimizer_options: dict[str, float | int | bool] | None = None,
        method = "Powell"
    ) -> NxMFitResult:
        """Fit amplitudes and delays while profiling PCA coefficients.

        
        Missing initial values are filled by :meth:`initialize`. When explicit
        final delay bounds are absent, each delay is restricted to
        ``initial_delay +/- delay_half_width``.

        Args:
            traces: Baseline-subtracted traces with shape
                ``(n_channels, n_samples)``.
            fix_delay: If ``fix_delay`` is True, the delays are fixed at their initial values. Default is ``False``.
            initial_amplitudes: Optional nonnegative starting amplitudes.
            initial_delays: Optional starting delays in samples.
            initialization_bounds: Search bounds used by :meth:`initialize`.
            delay_bounds: Optional final shared or per-channel bounds.
            delay_half_width: Final half-width around each initial delay when
                ``delay_bounds`` is not supplied.
            optimizer_options: Options passed to SciPy's Powell optimizer.

        Returns:
            Fitted parameters and diagnostics.

        Raises:
            ValueError: If traces, parameters, or bounds are invalid.
        """

        trace_values = self._validate_traces(traces)
        trace_fft = np.fft.rfft(trace_values, axis=-1).T[: self.frequency_limit]

        ## Get initial amplitudes and delays, either from user input or by estimating them.
        if initial_amplitudes is None or initial_delays is None:
            estimated_amplitudes, estimated_delays = self.initialize(
                trace_values, delay_bounds=initialization_bounds
            )
        else:
            estimated_amplitudes = np.empty(0)
            estimated_delays = np.empty(0)

        amplitudes = (
            estimated_amplitudes
            if initial_amplitudes is None
            else np.asarray(initial_amplitudes, dtype=float)
        )
        delays = (
            estimated_delays
            if initial_delays is None
            else np.asarray(initial_delays, dtype=float)
        )
        amplitudes, delays = self._validate_parameters(amplitudes, delays)

        ## Get delay bounds for the final fit, either from user input or by constructing them from the initial delays and half-width.
        if delay_bounds is None:
            half_width = np.asarray(delay_half_width, dtype=float)
            if half_width.ndim == 0:
                half_width = np.full(self.n_channels, float(half_width))
            if half_width.shape != (self.n_channels,):
                raise ValueError(
                    "delay_half_width must be scalar or have shape (n_channels,)"
                )
            if not np.all(np.isfinite(half_width)) or np.any(half_width < 0.0):
                raise ValueError("delay_half_width must be finite and nonnegative")
            final_delay_bounds = np.column_stack(
                (delays - half_width, delays + half_width)
            )
        else:
            final_delay_bounds = self._normalize_delay_bounds(delay_bounds)

        ## Clip the delays to the final bounds.
        delays = np.clip(
            delays, final_delay_bounds[:, 0], final_delay_bounds[:, 1]
        )
        if not fix_delay:
            start = np.concatenate((amplitudes, delays))
            bounds = [(0.0, None)] * self.n_channels + [
                tuple(pair) for pair in final_delay_bounds
            ]
        else:
            start = amplitudes
            bounds = [(0.0, None)] * self.n_channels

        def objective(parameters: FloatArray) -> float:
            if not fix_delay:
                profile, _ = self._profile_fft(
                    trace_fft,
                    parameters[: self.n_channels],
                    parameters[self.n_channels :],
                )
            else:
                profile, _ = self._profile_fft(
                    trace_fft,
                    parameters[: self.n_channels],
                    delays,
                )
            return profile.objective

        options: dict[str, float | int | bool] = {
            "xtol": 1e-3,
            "ftol": 1e-6,
            "maxiter": 500,
        }
        if optimizer_options is not None:
            options.update(optimizer_options)
        optimized = optimize.minimize(
            objective,
            start,
            method=method,
            bounds=bounds,
            options=options,
        )

        fitted_amplitudes = np.asarray(
            optimized.x[: self.n_channels], dtype=float
        )
        fitted_delays = np.asarray(optimized.x[self.n_channels :], dtype=float) if not fix_delay else delays
        fitted_profile, _ = self._profile_fft(
            trace_fft, fitted_amplitudes, fitted_delays
        )
        amplitude_tolerance = max(
            1e-8 * max(1.0, float(np.max(fitted_amplitudes, initial=0.0))),
            float(options.get("xtol", 1e-3)),
        )
        delay_tolerance = max(1e-3, float(options.get("xtol", 1e-3)))
        amplitude_at_bound = fitted_amplitudes <= amplitude_tolerance
        delay_at_bound = np.logical_or(
            np.abs(fitted_delays - final_delay_bounds[:, 0]) <= delay_tolerance,
            np.abs(fitted_delays - final_delay_bounds[:, 1]) <= delay_tolerance,
        )

        return NxMFitResult(
            amplitudes=fitted_amplitudes,
            delays=fitted_delays,
            shape_coefficients=fitted_profile.shape_coefficients,
            residual_objective=fitted_profile.residual_objective,
            prior_objective=fitted_profile.prior_objective,
            objective=fitted_profile.objective,
            success=bool(optimized.success),
            message=str(optimized.message),
            nfev=int(optimized.nfev),
            nit=int(getattr(optimized, "nit", -1)),
            amplitude_at_bound=amplitude_at_bound,
            delay_at_bound=delay_at_bound,
        )

    def reconstruct(self, result: NxMFitResult) -> FloatArray:
        """Return full-band time-domain waveforms for a fit result.

        Args:
            result: Result returned by :meth:`fit`.

        Returns:
            Reconstructed channel waveforms with shape
            ``(n_channels, n_samples)``.

        Raises:
            ValueError: If result parameter shapes are incompatible.
        """

        amplitudes, delays = self._validate_parameters(
            result.amplitudes, result.delays
        )
        coefficients = np.asarray(result.shape_coefficients, dtype=float)
        if coefficients.shape != (self.n_channels, self.n_modes):
            raise ValueError("result shape coefficients do not match reconstructor")
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("result shape coefficients must be finite")

        phase = self._phase(delays, full=True)
        shifted_mean = self._mean_fft_full * phase
        shifted_basis = self._basis_fft_full * phase[:, :, None]
        fitted_shape = np.einsum(
            "fjk,jk->fj", shifted_basis, coefficients, optimize=True
        )
        model_fft = amplitudes[None, :] * (shifted_mean + fitted_shape)
        return np.fft.irfft(model_fft.T, n=self.n_samples, axis=-1)


__all__ = [
    "NxMFitResult",
    "NxMProfileResult",
    "NxMReconstructor",
    "estimate_csd",
]
