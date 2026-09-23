from math import log, pi, sqrt

import numpy as np
import pytest

from dispersion import pulse, ridb
from dispersion.pulse import Layer, Spectrum, simulate

SILICA = """
DATA:
  - type: formula 1
    wavelength_range: 0.21 6.7
    coefficients: 0 0.6961663 0.0684043 0.4079426 0.1162414 0.8974794 9.896161
"""


@pytest.fixture
def silica():
    return ridb.Material.from_yaml_text(SILICA, name="fused silica")


def gaussian_tl(center_nm, fwhm_nm):
    return 4 * log(2) / pulse.bandwidth_nm_to_omega(center_nm, fwhm_nm)


def test_transform_limit_gaussian():
    r = simulate(Spectrum(800, "gaussian", 10), {})
    assert r.fwhm_tl == pytest.approx(gaussian_tl(800, 10), rel=1e-3)
    assert r.fwhm_tl == pytest.approx(94.1, abs=0.2)  # 0.441 λ²/(cΔλ)
    assert r.fwhm_in == pytest.approx(r.fwhm_tl, rel=1e-6)


def test_sech2_time_bandwidth():
    r = simulate(Spectrum(800, "sech2", 10), {})
    dnu = pulse.bandwidth_nm_to_omega(800, 10) / (2 * pi)
    assert r.fwhm_tl * dnu == pytest.approx(0.3148, rel=2e-3)


@pytest.mark.parametrize("gdd", [500.0, -2000.0, 20000.0])
def test_pure_gdd_broadening(gdd):
    r = simulate(Spectrum(800, "gaussian", 30), {2: gdd})
    t0 = r.fwhm_tl
    expected = t0 * sqrt(1 + (4 * log(2) * gdd / t0**2) ** 2)
    assert r.fwhm_in == pytest.approx(expected, rel=2e-3)
    assert r.peak_in_rel_tl == pytest.approx(t0 / expected, rel=5e-3)


def test_positive_gdd_red_leads():
    r = simulate(Spectrum(800, "gaussian", 30), {2: 2000.0})
    # instantaneous frequency (ω0 - dΦ/dt with E ∝ e^{iΦ}) must rise with time
    E, t = r.E_in, r.t_in
    core = np.abs(E) ** 2 > 0.5 * (np.abs(E) ** 2).max()
    inst = -np.gradient(np.unwrap(np.angle(E)), t)
    slope = np.polyfit(t[core], inst[core], 1)[0]
    assert slope > 0


def test_silica_refractive_index_and_gdd(silica):
    assert silica.n(0.8) == pytest.approx(1.45332, abs=2e-5)
    gd, gdd, tod, fod = pulse.dispersion_coefficients(
        Layer(silica, 1.0).phase, pulse.wl_nm_to_omega(800)
    )
    assert gdd == pytest.approx(36.16, abs=0.05)  # fs²/mm
    assert tod == pytest.approx(27.4, abs=0.3)  # fs³/mm
    # group index of silica at 800 nm ≈ 1.4671
    assert gd / (1e3 / pulse.C_UM_PER_FS) == pytest.approx(1.4671, abs=2e-4)


def test_material_equals_taylor_when_higher_orders_small(silica):
    spec = Spectrum(800, "gaussian", 10)
    layer = Layer(silica, 20.0)
    r = simulate(spec, {}, [layer])
    _, gdd, tod, _ = r.layer_dispersion[0]
    r2 = simulate(spec, {2: gdd, 3: tod})
    assert r.fwhm_out == pytest.approx(r2.fwhm_in, rel=1e-3)


def test_precompensation_restores_tl(silica):
    spec = Spectrum(800, "gaussian", 40)
    r = simulate(spec, {}, [Layer(silica, 10.0)])
    _, gdd, tod, fod = r.total_dispersion
    r2 = simulate(spec, {2: -gdd, 3: -tod, 4: -fod}, [Layer(silica, 10.0)])
    assert r2.fwhm_out == pytest.approx(r2.fwhm_tl, rel=5e-3)


def test_energy_conserved_without_absorption(silica):
    r = simulate(Spectrum(800, "gaussian", 30), {3: 5000.0}, [Layer(silica, 50.0)])
    e_in = (np.abs(r.E_in) ** 2).sum()
    e_out = (np.abs(r.E_out) ** 2).sum()
    assert e_out == pytest.approx(e_in, rel=1e-9)
    assert r.transmission == pytest.approx(1.0)
    assert not np.isnan(r.fwhm_out)


def test_custom_spectrum_matches_analytic():
    wl = np.linspace(700, 900, 2001)
    w = pulse.wl_nm_to_omega(wl)
    width = pulse.bandwidth_nm_to_omega(800, 20)
    s_omega = np.exp(-4 * log(2) * ((w - pulse.wl_nm_to_omega(800)) / width) ** 2)
    spec = Spectrum(800, "custom", custom_wl_nm=wl, custom_intensity=s_omega, per_wavelength=False)
    r = simulate(spec, {2: 300.0})
    r_ref = simulate(Spectrum(800, "gaussian", 20), {2: 300.0})
    assert r.fwhm_in == pytest.approx(r_ref.fwhm_in, rel=2e-3)


def test_all_zero_coefficients(silica):
    r = simulate(Spectrum(800, "gaussian", 30), {2: 0.0, 3: 0.0}, [Layer(silica, 10.0)])
    assert r.fwhm_in == pytest.approx(r.fwhm_tl, rel=1e-6)


def spm_spec(b, mode="flat", seed=300.0, shape="sech2"):
    return Spectrum(1030, "spm", spm_seed_fs=seed, spm_seed_shape=shape, spm_b=b, spm_phase=mode)


def test_spm_without_nonlinearity_is_the_seed():
    assert simulate(spm_spec(0.0), {}).fwhm_tl == pytest.approx(300.0, rel=2e-3)
    assert simulate(spm_spec(0.0, shape="gaussian"), {}).fwhm_tl == pytest.approx(300.0, rel=2e-3)


@pytest.mark.parametrize("b", [3.0, 10.0, 30.0])
def test_spm_rms_broadening_gaussian(b):
    # Agrawal, Nonlinear Fiber Optics: σω/σω0 = sqrt(1 + 4/(3√3) φmax²) for a Gaussian
    def rms(dw, S):
        w = S / S.sum()
        return np.sqrt(((dw - (dw * w).sum()) ** 2 * w).sum())

    ratio = rms(*pulse.spm_spectrum(300.0, "gaussian", b)[:2]) / rms(*pulse.spm_spectrum(300.0, "gaussian", 0.0)[:2])
    assert ratio == pytest.approx(sqrt(1 + 4 / (3 * sqrt(3)) * b**2), rel=1e-3)


def test_spm_full_phase_keeps_temporal_intensity():
    r = simulate(spm_spec(20.0, "full"), {})
    assert r.fwhm_in == pytest.approx(300.0, rel=5e-3)  # SPM only changes the temporal phase
    assert r.fwhm_tl < 30


def test_spm_is_up_chirped_and_compressible():
    g = pulse.spm_compressor_gdd(300.0, "sech2", 20.0)
    assert g < 0  # positive chirp needs negative GDD
    full = simulate(spm_spec(20.0, "full"), {})
    compressed = simulate(spm_spec(20.0, "full"), {2: g})
    gdd_mode = simulate(spm_spec(20.0, "gdd"), {})
    assert compressed.peak_in_rel_tl == pytest.approx(gdd_mode.peak_in_rel_tl, rel=1e-3)
    assert 0.5 < gdd_mode.peak_in_rel_tl < 1.0 < gdd_mode.peak_in_rel_tl / full.peak_in_rel_tl


def test_spm_target_solver():
    b = pulse.spm_b_for_target(300.0, "sech2", 25.0)
    assert simulate(spm_spec(b), {}).fwhm_tl == pytest.approx(25.0, rel=5e-3)


def test_spm_through_glass_matches_equivalent_gdd(silica):
    # compressed SPM pulse + 10 mm silica ≈ same pulse with the silica GDD/TOD added
    b = pulse.spm_b_for_target(300.0, "sech2", 25.0)
    r = simulate(spm_spec(b), {}, [Layer(silica, 10.0)])
    _, gdd, tod, fod = r.layer_dispersion[0]
    r2 = simulate(spm_spec(b), {2: gdd, 3: tod, 4: fod})
    assert r.fwhm_out == pytest.approx(r2.fwhm_in, rel=1e-2)
    assert r.fwhm_out > r.fwhm_in


def test_fwhm_helper():
    x = np.linspace(-10, 10, 2001)
    assert pulse.fwhm(x, np.exp(-4 * log(2) * x**2 / 9)) == pytest.approx(3.0, rel=1e-4)
