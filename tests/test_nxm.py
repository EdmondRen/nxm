"""Tests for the frequency-domain NxM reconstructor."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from nxm import NxMFitResult, NxMReconstructor, estimate_csd


def make_problem(
    *,
    n_channels: int = 2,
    n_modes: int = 1,
    n_samples: int = 128,
    noise_scale: float = 1e-4,
) -> tuple[NxMReconstructor, np.ndarray, np.ndarray, np.ndarray]:
    """Create a small deterministic reconstruction problem.

    Args:
        n_channels: Number of synthetic channels.
        n_modes: Number of PCA modes per channel.
        n_samples: Trace length.
        noise_scale: Scale of the raw RFFT covariance.

    Returns:
        Reconstructor, mean templates, PCA components, and variances.
    """

    sample = np.arange(n_samples, dtype=float)
    means = []
    components = []
    for channel in range(n_channels):
        center = 0.4 * n_samples + 2.0 * channel
        width = 4.5 + channel
        mean = np.exp(-0.5 * ((sample - center) / width) ** 2)
        channel_components = []
        if n_modes:
            scaled_sample = (sample - center) / width
            candidates = [
                (scaled_sample**2 - 0.5) * mean,
                (scaled_sample**4 - 3.0 * scaled_sample**2 + 0.75) * mean,
            ]
            # Remove amplitude- and delay-like directions so the synthetic
            # recovery test has identifiable shape and timing parameters.
            nuisance_directions = [mean, np.gradient(mean)]
            nuisance_directions = [
                direction / np.linalg.norm(direction)
                for direction in nuisance_directions
            ]
            for mode in range(n_modes):
                component = candidates[mode % len(candidates)].copy()
                for nuisance in nuisance_directions:
                    component -= np.dot(component, nuisance) * nuisance
                for previous in channel_components:
                    component -= np.dot(component, previous) * previous
                component /= np.linalg.norm(component)
                channel_components.append(component)
        means.append(mean)
        components.append(channel_components)

    means_array = np.asarray(means)
    if n_modes:
        components_array = np.asarray(components)
    else:
        components_array = np.empty((n_channels, 0, n_samples))
    variances = np.full((n_channels, n_modes), 0.04)
    correlation = np.full((n_channels, n_channels), 0.2)
    np.fill_diagonal(correlation, 1.0)
    csd = np.repeat(
        (noise_scale * correlation)[None, :, :],
        n_samples // 2 + 1,
        axis=0,
    ).astype(complex)
    reconstructor = NxMReconstructor(
        means_array,
        components_array,
        variances,
        csd,
        frequency_limit=min(50, n_samples // 2 + 1),
    )
    return reconstructor, means_array, components_array, variances


def synthesize(
    means: np.ndarray,
    components: np.ndarray,
    variances: np.ndarray,
    amplitudes: np.ndarray,
    delays: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Synthesize circularly shifted traces using the documented model."""

    n_samples = means.shape[-1]
    frequencies = np.arange(n_samples // 2 + 1, dtype=float)
    phase = np.exp(
        -2j * np.pi * frequencies[:, None] * delays[None, :] / n_samples
    )
    mean_fft = np.fft.rfft(means, axis=-1).T
    scaled = components * np.sqrt(variances)[:, :, None]
    basis_fft = np.transpose(np.fft.rfft(scaled, axis=-1), (2, 0, 1))
    shape_fft = np.einsum(
        "fjk,jk->fj", basis_fft * phase[:, :, None], coefficients
    )
    model_fft = amplitudes[None, :] * (mean_fft * phase + shape_fft)
    return np.fft.irfft(model_fft.T, n=n_samples, axis=-1)


def make_result(
    amplitudes: np.ndarray,
    delays: np.ndarray,
    coefficients: np.ndarray,
) -> NxMFitResult:
    """Build a fit result for reconstruction-only tests."""

    return NxMFitResult(
        amplitudes=amplitudes,
        delays=delays,
        shape_coefficients=coefficients,
        residual_objective=0.0,
        prior_objective=float(np.sum(coefficients**2)),
        objective=float(np.sum(coefficients**2)),
        success=True,
        message="synthetic",
        nfev=0,
        nit=0,
        amplitude_at_bound=np.zeros_like(amplitudes, dtype=bool),
        delay_at_bound=np.zeros_like(delays, dtype=bool),
    )


def test_estimate_csd_recovers_correlated_white_noise() -> None:
    """Raw RFFT covariance should approach n_samples times time covariance."""

    rng = np.random.default_rng(20261023)
    n_events, n_samples = 3000, 16
    covariance = np.array([[2.0, 0.6], [0.6, 1.0]])
    traces = rng.multivariate_normal(
        np.zeros(2), covariance, size=(n_events, n_samples)
    ).transpose(0, 2, 1)

    csd = estimate_csd(traces, detrend=None)

    np.testing.assert_allclose(csd, np.swapaxes(csd.conj(), -1, -2))
    np.testing.assert_allclose(
        np.mean(csd[1:-1], axis=0).real,
        n_samples * covariance,
        rtol=0.05,
        atol=0.15,
    )


def test_csd_eigenvalue_floor_conditions_singular_bins() -> None:
    """A rank-one CSD should become positive definite after flooring."""

    n_samples = 32
    means = np.ones((2, n_samples))
    components = np.empty((2, 0, n_samples))
    variances = np.empty((2, 0))
    singular = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=complex)
    csd = np.repeat(singular[None, :, :], n_samples // 2 + 1, axis=0)

    reconstructor = NxMReconstructor(
        means,
        components,
        variances,
        csd,
        csd_eigenvalue_floor=1e-4,
    )

    eigenvalues = np.linalg.eigvalsh(reconstructor.noise_csd[1:])
    assert np.all(eigenvalues > 0.0)
    assert np.all(np.isfinite(reconstructor.inverse_noise_csd))


def test_omitted_dc_requires_an_active_frequency_bin() -> None:
    """A DC-only likelihood is invalid when DC has zero weight."""

    with pytest.raises(ValueError, match="non-DC"):
        NxMReconstructor(
            np.ones((1, 8)),
            np.empty((1, 0, 8)),
            np.empty((1, 0)),
            np.ones((5, 1, 1), dtype=complex),
            frequency_limit=1,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("means", np.zeros((2, 3, 4)), "mean_templates"),
        ("components", np.zeros((3, 1, 32)), "channel count"),
        ("variances", -np.ones((2, 1)), "nonnegative"),
        ("csd", np.zeros((17, 2)), "noise_csd"),
    ],
)
def test_constructor_rejects_invalid_inputs(
    field: str, replacement: np.ndarray, message: str
) -> None:
    """Constructor failures should identify invalid public inputs."""

    means = np.ones((2, 32))
    components = np.ones((2, 1, 32))
    variances = np.ones((2, 1))
    csd = np.repeat(np.eye(2, dtype=complex)[None], 17, axis=0)
    values = {
        "means": means,
        "components": components,
        "variances": variances,
        "csd": csd,
    }
    values[field] = replacement

    with pytest.raises(ValueError, match=message):
        NxMReconstructor(
            values["means"],
            values["components"],
            values["variances"],
            values["csd"],
        )


def test_profile_recovers_known_coefficients_at_high_snr() -> None:
    """The analytic profile solve should recover injected PCA coefficients."""

    reconstructor, means, components, variances = make_problem(noise_scale=1e-7)
    amplitudes = np.array([2.0, 3.0])
    delays = np.array([2.25, -1.75])
    coefficients = np.array([[0.5], [-0.3]])
    traces = synthesize(
        means, components, variances, amplitudes, delays, coefficients
    )

    profile = reconstructor.profile(traces, amplitudes, delays)

    np.testing.assert_allclose(
        profile.shape_coefficients, coefficients, atol=1e-4
    )
    assert profile.objective == pytest.approx(
        profile.residual_objective + profile.prior_objective
    )


def test_reconstruct_uses_fractional_delay_with_documented_sign() -> None:
    """Reconstruction should reproduce independently synthesized waveforms."""

    reconstructor, means, components, variances = make_problem()
    amplitudes = np.array([1.5, 2.5])
    delays = np.array([1.4, -2.2])
    coefficients = np.array([[0.2], [-0.4]])
    expected = synthesize(
        means, components, variances, amplitudes, delays, coefficients
    )

    actual = reconstructor.reconstruct(
        make_result(amplitudes, delays, coefficients)
    )

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_profile_uses_off_diagonal_csd_terms() -> None:
    """The residual objective should equal a direct correlated-noise form."""

    reconstructor, _, _, _ = make_problem(n_modes=0)
    rng = np.random.default_rng(10)
    traces = rng.normal(size=(reconstructor.n_channels, reconstructor.n_samples))
    amplitudes = np.zeros(reconstructor.n_channels)
    delays = np.zeros(reconstructor.n_channels)

    profile = reconstructor.profile(traces, amplitudes, delays)
    spectrum = np.fft.rfft(traces, axis=-1).T[: reconstructor.frequency_limit]
    weighted = np.einsum(
        "fjm,fm->fj", reconstructor.inverse_noise_csd, spectrum
    )
    expected = np.sum(
        reconstructor._weights[:, None]
        * np.real(spectrum.conj() * weighted)
    )

    assert profile.residual_objective == pytest.approx(expected)


def test_end_to_end_fit_recovers_multichannel_injection() -> None:
    """Bounded Powell fitting should meet high-SNR recovery requirements."""

    reconstructor, means, components, variances = make_problem(noise_scale=1e-4)
    true_amplitudes = np.array([2.0, 3.0])
    true_delays = np.array([2.3, -1.7])
    true_coefficients = np.array([[0.5], [-0.3]])
    traces = synthesize(
        means,
        components,
        variances,
        true_amplitudes,
        true_delays,
        true_coefficients,
    )

    initial_amplitudes, initial_delays = reconstructor.initialize(
        traces, (-8.0, 8.0)
    )
    result = reconstructor.fit(
        traces,
        initial_amplitudes=initial_amplitudes,
        initial_delays=initial_delays,
        delay_half_width=3.0,
        optimizer_options={"xtol": 1e-5, "ftol": 1e-9, "maxiter": 150},
    )

    assert result.success, result.message
    np.testing.assert_allclose(result.amplitudes, true_amplitudes, rtol=0.02)
    np.testing.assert_allclose(result.delays, true_delays, atol=0.15)
    np.testing.assert_allclose(
        result.shape_coefficients, true_coefficients, atol=0.1
    )
    assert not np.any(result.delay_at_bound)


def test_noise_only_fit_is_finite_and_respects_bounds() -> None:
    """Noise-only inputs should return finite constrained estimates."""

    reconstructor, _, _, _ = make_problem(n_samples=64)
    rng = np.random.default_rng(11)
    traces = rng.normal(scale=1e-3, size=(2, 64))

    result = reconstructor.fit(
        traces,
        initialization_bounds=(-4.0, 4.0),
        delay_half_width=1.0,
        optimizer_options={"maxiter": 30},
    )

    assert np.all(np.isfinite(result.amplitudes))
    assert np.all(result.amplitudes >= 0.0)
    assert np.all(np.isfinite(result.delays))
    assert np.all(np.isfinite(result.shape_coefficients))
    assert np.isfinite(result.objective)


def test_fit_reports_optimizer_failure_and_parameter_boundaries() -> None:
    """Termination and active-bound diagnostics should be exposed."""

    reconstructor, _, _, _ = make_problem(n_modes=0, n_samples=64)
    traces = np.zeros((2, 64))
    result = reconstructor.fit(
        traces,
        initial_amplitudes=np.zeros(2),
        initial_delays=np.zeros(2),
        delay_bounds=np.zeros((2, 2)),
        optimizer_options={"maxiter": 0},
    )

    assert not result.success
    assert np.all(result.amplitude_at_bound)
    assert np.all(result.delay_at_bound)
    assert result.nfev >= 1


def test_reconstruct_rejects_incompatible_result() -> None:
    """Results from an incompatible model should fail clearly."""

    reconstructor, _, _, _ = make_problem()
    result = make_result(np.ones(2), np.zeros(2), np.zeros((2, 1)))

    with pytest.raises(ValueError, match="shape coefficients"):
        reconstructor.reconstruct(
            replace(result, shape_coefficients=np.zeros((2, 2)))
        )
