"""Linear propagation of an ultrashort pulse through dispersive materials.

Conventions
-----------
* Units: time in fs, angular frequency in rad/fs, wavelength in µm internally
  (nm at the user-facing level), thickness in mm.
* Field: E(t) = ∫ Ẽ(ω) exp(-iωt) dω, and a spectral phase φ(ω) enters as
  Ẽ(ω)·exp(+iφ(ω)).  The group delay is then +dφ/dω, so positive GDD (normal
  material dispersion) delays the blue side → positive (up-)chirp.
* The input spectral phase is a Taylor series about ω0:
      φ(ω) = Σ_n  c_n / n! · (ω-ω0)^n      c2 = GDD [fs²], c3 = TOD [fs³], ...
* The material phase is the exact  φ_mat(ω) = ω·n(ω)·L / c  (all orders).
  Its constant and linear (group-delay) parts are removed for display, so each
  pulse is plotted in a frame moving with the group delay at ω0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from math import factorial, log, pi
from typing import Optional, Sequence

import numpy as np

C_UM_PER_FS = 0.299792458  # speed of light, µm/fs
FWHM_SECH2 = 2 * np.arccosh(np.sqrt(2))  # sech²(x) FWHM in units of x
MAX_POINTS = 2**22


# --------------------------------------------------------------------------
# Unit helpers
# --------------------------------------------------------------------------
def wl_nm_to_omega(wl_nm):
    return 2 * pi * C_UM_PER_FS / (np.asarray(wl_nm, dtype=float) * 1e-3)


def omega_to_wl_nm(omega):
    return 2 * pi * C_UM_PER_FS / np.asarray(omega, dtype=float) * 1e3


def bandwidth_nm_to_omega(center_nm: float, fwhm_nm: float) -> float:
    """FWHM in ω of a spectrum spanning center ± fwhm/2 in wavelength."""
    lo, hi = center_nm - fwhm_nm / 2, center_nm + fwhm_nm / 2
    if lo <= 0:
        raise ValueError("Bandwidth must be smaller than twice the centre wavelength.")
    return float(wl_nm_to_omega(lo) - wl_nm_to_omega(hi))


# --------------------------------------------------------------------------
# Self-phase modulation (e.g. multipass-cell broadening)
# --------------------------------------------------------------------------
SPM_PHASE_MODES = ("flat", "gdd", "full")


def _seed_intensity(t: np.ndarray, fwhm_fs: float, shape: str) -> np.ndarray:
    if shape == "gaussian":
        return np.exp(-4 * log(2) * (t / fwhm_fs) ** 2)
    return 1 / np.cosh(FWHM_SECH2 * t / fwhm_fs) ** 2


@lru_cache(maxsize=64)
def spm_spectrum(seed_fwhm_fs: float, seed_shape: str, b_integral: float):
    """Spectrum of a transform-limited seed after pure SPM with peak nonlinear phase B.

    E(t) = sqrt(I(t)) * exp(+i B I(t))  (n2 > 0: red-shifted leading edge, up-chirp).
    Returns (dw, S, phase): sorted detuning grid (rad/fs), intensity (peak 1) and the
    unwrapped spectral phase (rad) with constant and linear parts removed.
    """
    tau = float(seed_fwhm_fs)
    half = (8.0 if seed_shape == "gaussian" else 14.0) * tau
    tc = np.linspace(-half, half, 20001)
    slope = np.abs(np.gradient(_seed_intensity(tc, tau, seed_shape), tc)).max()
    w_max = b_integral * slope + 40 / tau  # max instantaneous-frequency shift + seed bandwidth
    dt = pi / (1.5 * w_max)
    n = int(2 ** np.ceil(np.log2(4 * 2 * half / dt)))  # 4x zero padding -> fine spectral grid
    t = (np.arange(n) - n // 2) * dt
    I = _seed_intensity(t, tau, seed_shape)
    E = np.sqrt(I) * np.exp(1j * b_integral * I)
    # Ẽ(ω) = ∫ E(t) e^{+iωt} dt  (inverse of this module's E(t) = ∫ Ẽ e^{-iωt} dω)
    Ew = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(E)))
    dw = np.fft.fftshift(2 * pi * np.fft.fftfreq(n, dt))
    S = np.abs(Ew) ** 2
    S /= S.max()
    idx = np.nonzero(S > 1e-14)[0]
    sl = slice(idx[0], idx[-1] + 1)
    phase = np.zeros(n)
    phase[sl] = np.unwrap(np.angle(Ew[sl]))
    keep = S > 1e-8
    lin = np.polyfit(dw[keep], phase[keep], 1, w=np.sqrt(S[keep]))
    phase[sl] -= np.polyval(lin, dw[sl])
    phase[: idx[0]] = phase[idx[0]]
    phase[idx[-1] + 1 :] = phase[idx[-1]]
    return dw, S, phase


def _peak_power(dw, S, phase, pad: int = 8) -> float:
    """Peak of |E(t)|^2 for spectral amplitude sqrt(S) and phase (zero-padded FFT)."""
    n = len(dw)
    field = np.zeros(pad * n, dtype=complex)
    field[:n] = np.sqrt(S) * np.exp(1j * phase)
    return float((np.abs(np.fft.fft(field)) ** 2).max())


def _tl_fwhm(dw, S, pad: int = 8) -> float:
    n = len(dw)
    step = dw[1] - dw[0]
    field = np.zeros(pad * n)
    field[:n] = np.sqrt(S)
    I = np.abs(np.fft.fftshift(np.fft.fft(field))) ** 2
    t = (np.arange(pad * n) - pad * n // 2) * (2 * pi / (pad * n * step))
    return fwhm(t, I)


@lru_cache(maxsize=64)
def spm_compressor_gdd(seed_fwhm_fs: float, seed_shape: str, b_integral: float) -> float:
    """GDD (fs^2, negative) of an ideal GDD-only compressor that maximises the peak power."""
    from scipy.optimize import minimize_scalar

    dw, S, ph = spm_spectrum(seed_fwhm_fs, seed_shape, b_integral)
    keep = S > 1e-4
    if b_integral <= 0 or keep.sum() < 5:
        return 0.0
    g0 = -2 * np.polyfit(dw[keep], ph[keep], 2, w=np.sqrt(S[keep]))[0]
    if g0 == 0:
        return 0.0
    res = minimize_scalar(
        lambda g: -_peak_power(dw, S, ph + 0.5 * g * dw**2),
        bounds=sorted((0.0, 2 * g0)), method="bounded", options={"xatol": abs(g0) * 1e-4},
    )
    return float(res.x)


def spm_tl_fwhm(seed_fwhm_fs: float, seed_shape: str, b_integral: float) -> float:
    dw, S, _ = spm_spectrum(seed_fwhm_fs, seed_shape, b_integral)
    return _tl_fwhm(dw, S)


def spm_b_for_target(seed_fwhm_fs: float, seed_shape: str, target_fwhm_fs: float) -> float:
    """B-integral whose SPM spectrum has a transform-limited FWHM of *target_fwhm_fs*."""
    if target_fwhm_fs >= spm_tl_fwhm(seed_fwhm_fs, seed_shape, 0.0):
        return 0.0
    lo, hi = 0.0, 1.0
    while spm_tl_fwhm(seed_fwhm_fs, seed_shape, hi) > target_fwhm_fs:
        lo, hi = hi, hi * 2
        if hi > 2000:
            raise ValueError("Target duration is too short for this seed.")
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if spm_tl_fwhm(seed_fwhm_fs, seed_shape, round(mid, 6)) > target_fwhm_fs:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-4 * hi:
            break
    return round(0.5 * (lo + hi), 4)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
@dataclass
class Spectrum:
    """Spectral intensity definition.

    shape: "gaussian", "sech2", "supergaussian", "custom" or "spm".
    For "custom", give wavelength (nm) and intensity arrays; set
    ``per_wavelength=True`` if the intensity is a density per unit wavelength
    (as from most spectrometers) so it is converted to per unit frequency.
    A measured spectral phase (rad, this module's sign convention) can be given as
    ``custom_phase_wl_nm`` / ``custom_phase``; with ``custom_phase_unwrap`` it is
    unwrapped along frequency first. Outside its range the phase is held constant.
    For "spm", a transform-limited seed (``spm_seed_fs``, ``spm_seed_shape``) is
    broadened by self-phase modulation with peak nonlinear phase ``spm_b`` (rad).
    ``spm_phase`` selects the pulse's spectral phase: "flat" (ideally compressed),
    "gdd" (residual after the best GDD-only compressor) or "full" (uncompressed).
    """

    center_nm: float
    shape: str = "gaussian"
    fwhm_nm: float = 30.0
    sg_order: int = 2
    custom_wl_nm: Optional[np.ndarray] = None
    custom_intensity: Optional[np.ndarray] = None
    per_wavelength: bool = True
    custom_phase_wl_nm: Optional[np.ndarray] = None
    custom_phase: Optional[np.ndarray] = None
    custom_phase_unwrap: bool = True
    spm_seed_fs: float = 300.0
    spm_seed_shape: str = "sech2"
    spm_b: float = 10.0
    spm_phase: str = "flat"

    def _spm(self):
        return spm_spectrum(float(self.spm_seed_fs), self.spm_seed_shape, float(self.spm_b))

    def _custom_phase_arrays(self):
        wl = np.asarray(self.custom_phase_wl_nm, dtype=float)
        ph = np.asarray(self.custom_phase, dtype=float)
        keep = np.isfinite(wl) & np.isfinite(ph) & (wl > 0)
        w = wl_nm_to_omega(wl[keep])
        order = np.argsort(w)
        w, ph = w[order], ph[keep][order]
        if self.custom_phase_unwrap:
            ph = np.unwrap(ph)
        return w - self.omega0, ph

    @property
    def has_measured_phase(self) -> bool:
        return self.shape == "custom" and self.custom_phase is not None and len(self.custom_phase) > 1

    def measured_phase_range(self) -> Optional[tuple[float, float]]:
        """Detuning interval covered by the measured phase, if any."""
        if not self.has_measured_phase:
            return None
        x, _ = self._custom_phase_arrays()
        return float(x[0]), float(x[-1])

    def phase(self, dw: np.ndarray) -> np.ndarray:
        """Intrinsic spectral phase of the source (rad): SPM or measured, else zero."""
        dw = np.asarray(dw, dtype=float)
        if self.has_measured_phase:
            x, ph = self._custom_phase_arrays()
            return np.interp(dw, x, ph)
        if self.shape != "spm" or self.spm_phase == "flat":
            return np.zeros_like(dw)
        x, _, ph = self._spm()
        out = np.interp(dw, x, ph)
        if self.spm_phase == "gdd":
            g = spm_compressor_gdd(float(self.spm_seed_fs), self.spm_seed_shape, float(self.spm_b))
            out = out + 0.5 * g * dw**2
        return out

    @property
    def omega0(self) -> float:
        return float(wl_nm_to_omega(self.center_nm))

    def _custom_arrays(self):
        wl = np.asarray(self.custom_wl_nm, dtype=float)
        s = np.clip(np.asarray(self.custom_intensity, dtype=float), 0, None)
        keep = np.isfinite(wl) & np.isfinite(s) & (wl > 0)
        wl, s = wl[keep], s[keep]
        w = wl_nm_to_omega(wl)
        if self.per_wavelength:
            s = s * wl**2  # S_ω ∝ S_λ · λ²
        order = np.argsort(w)
        return w[order] - self.omega0, s[order] / s.max()

    def intensity(self, dw: np.ndarray) -> np.ndarray:
        """Spectral intensity (peak ≈ 1) vs detuning dw = ω - ω0."""
        if self.shape == "custom":
            x, s = self._custom_arrays()
            return np.interp(dw, x, s, left=0.0, right=0.0)
        if self.shape == "spm":
            x, s, _ = self._spm()
            return np.interp(dw, x, s, left=0.0, right=0.0)
        width = bandwidth_nm_to_omega(self.center_nm, self.fwhm_nm)
        if self.shape == "gaussian":
            return np.exp(-4 * log(2) * (dw / width) ** 2)
        if self.shape == "supergaussian":
            return np.exp(-log(2) * np.abs(2 * dw / width) ** (2 * self.sg_order))
        if self.shape == "sech2":
            return 1 / np.cosh(FWHM_SECH2 * dw / width) ** 2
        raise ValueError(f"Unknown spectrum shape {self.shape!r}")

    def support(self) -> tuple[float, float]:
        """Detuning interval outside which the intensity is < ~1e-12."""
        if self.shape == "custom":
            x, s = self._custom_arrays()
            nz = np.nonzero(s > 0)[0]
            return float(x[nz[0]]), float(x[nz[-1]])
        if self.shape == "spm":
            x, s, _ = self._spm()
            nz = np.nonzero(s > 1e-12)[0]
            return float(x[nz[0]]), float(x[nz[-1]])
        width = bandwidth_nm_to_omega(self.center_nm, self.fwhm_nm)
        if self.shape == "gaussian":
            half = width * np.sqrt(log(1e12) / (4 * log(2)))
        elif self.shape == "supergaussian":
            half = width / 2 * (log(1e12) / log(2)) ** (1 / (2 * self.sg_order))
        else:
            half = width * 14.5 / FWHM_SECH2
        return -half, half


@dataclass
class Layer:
    material: object  # anything with n(wl_um), k(wl_um), n_range, name
    thickness_mm: float

    def phase(self, omega: np.ndarray) -> np.ndarray:
        wl = 2 * pi * C_UM_PER_FS / omega
        return omega / C_UM_PER_FS * self.material.n(wl) * self.thickness_mm * 1e3

    def log_amplitude(self, omega: np.ndarray) -> np.ndarray:
        wl = 2 * pi * C_UM_PER_FS / omega
        return -omega / C_UM_PER_FS * self.material.k(wl) * self.thickness_mm * 1e3


def taylor_phase(dw: np.ndarray, coeffs: dict[int, float]) -> np.ndarray:
    out = np.zeros_like(dw, dtype=float)
    for n, c in coeffs.items():
        if c:
            out += c / factorial(n) * dw**n
    return out


def fit_taylor(dw: np.ndarray, phase: np.ndarray, weight: np.ndarray, order: int = 4) -> list[float]:
    """Intensity-weighted polynomial fit of a spectral phase about dw = 0.

    Returns [GD (fs), GDD (fs²), TOD (fs³), FOD (fs⁴), ...] up to *order*.
    """
    ok = np.isfinite(phase) & (weight > 0)
    x, y, w = dw[ok], phase[ok], np.sqrt(weight[ok])
    if x.size <= order:
        return [0.0] * order
    half = max(abs(x.min()), abs(x.max()))
    p = np.polynomial.polynomial.Polynomial.fit(x, y, order, w=w, domain=[-half, half], window=[-1, 1])
    c = p.convert().coef
    return [float(c[n] * factorial(n)) if n < len(c) else 0.0 for n in range(1, order + 1)]


def dispersion_coefficients(phase_fn, omega0: float, orders: int = 4) -> list[float]:
    """[GD (fs), GDD (fs²), TOD (fs³), FOD (fs⁴)] of phase_fn at omega0 via a local polynomial fit."""
    h = 0.02 * omega0
    x = np.linspace(-h, h, 41)
    y = phase_fn(omega0 + x)
    y = y - y[20]
    p = np.polynomial.polynomial.Polynomial.fit(x, y, 8, domain=[-h, h], window=[-1, 1])
    c = p.convert().coef
    return [float(c[n] * factorial(n)) if n < len(c) else 0.0 for n in range(1, orders + 1)]


# --------------------------------------------------------------------------
# Pulse metrics
# --------------------------------------------------------------------------
def fwhm(x: np.ndarray, y: np.ndarray) -> float:
    """Full width between the outermost half-maximum crossings (linear interpolation)."""
    y = np.asarray(y, dtype=float)
    if y.max() <= 0:
        return float("nan")
    half = 0.5 * y.max()
    above = np.nonzero(y >= half)[0]
    i0, i1 = above[0], above[-1]
    if i0 == 0 or i1 == len(y) - 1:
        return float("nan")  # pulse not contained in the window
    left = x[i0 - 1] + (half - y[i0 - 1]) * (x[i0] - x[i0 - 1]) / (y[i0] - y[i0 - 1])
    right = x[i1] + (half - y[i1]) * (x[i1 + 1] - x[i1]) / (y[i1 + 1] - y[i1])
    return float(right - left)


def rms_width(x: np.ndarray, y: np.ndarray) -> float:
    w = y / y.sum()
    mean = (x * w).sum()
    return float(np.sqrt(((x - mean) ** 2 * w).sum()))


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
@dataclass
class Result:
    t: np.ndarray  # fs, time axis of the transform-limited pulse
    t_in: np.ndarray  # fs, input pulse axis (t=0 at the group delay of ω0)
    t_out: np.ndarray  # fs, output pulse axis (t=0 at the group delay of ω0)
    E_in: np.ndarray
    E_out: np.ndarray
    E_tl: np.ndarray
    dw: np.ndarray  # rad/fs, fftshifted
    omega0: float
    spectrum_in: np.ndarray  # |Ẽ|² input (peak 1)
    spectrum_out: np.ndarray  # after absorption
    phase_in: np.ndarray  # rad, linear part removed, NaN outside spectrum
    phase_out: np.ndarray
    fwhm_tl: float
    fwhm_in: float
    fwhm_out: float
    rms_in: float
    rms_out: float
    transmission: float
    peak_in_rel_tl: float  # peak power relative to transform limit (equal energy)
    peak_out_rel_tl: float  # includes absorption losses
    layer_dispersion: list[list[float]]  # per layer [GD, GDD, TOD, FOD]
    input_dispersion: list[float]  # [GD, GDD, TOD, FOD] of the input phase
    n_points: int
    dt: float
    warnings: list[str] = field(default_factory=list)

    @property
    def I_in(self):
        return np.abs(self.E_in) ** 2

    @property
    def I_out(self):
        return np.abs(self.E_out) ** 2

    @property
    def I_tl(self):
        return np.abs(self.E_tl) ** 2

    @property
    def wavelength_nm(self):
        with np.errstate(divide="ignore"):
            return omega_to_wl_nm(self.omega0 + self.dw)

    @property
    def total_dispersion(self) -> list[float]:
        tot = np.array(self.input_dispersion, dtype=float)
        for d in self.layer_dispersion:
            tot += d
        return tot.tolist()


def _time_window_and_step(spec, phase_fn, lo, hi) -> tuple[float, float, float]:
    """Pick (window T, step dt, char. time) from the spectral support and group-delay spread."""
    x = np.linspace(lo, hi, 16384)
    s = spec.intensity(x)
    sig = s > 1e-6 * s.max()
    # characteristic TL time from the rms spectral width
    w = s / s.sum()
    sigma_w = np.sqrt((((x - (x * w).sum()) ** 2) * w).sum())
    tau = 1 / max(sigma_w, 1e-12)
    # group delay spread of the phase (input or output) over the significant spectrum
    spread = 0.0
    for fn in phase_fn:
        g_lo, g_hi = _gd_range(x, fn(x), np.where(sig, s, 0.0))
        spread = max(spread, g_hi - g_lo)
    T = 1.3 * spread + 60 * tau
    dt_nyq = pi / (1.1 * max(abs(lo), abs(hi)))
    dt = min(dt_nyq, tau / 20)
    return T, dt, tau


def _gd_range(x: np.ndarray, phase: np.ndarray, weight: np.ndarray, q: float = 1e-4):
    """Intensity-weighted [q, 1-q] quantiles of the group delay dφ/dω.

    Robust against the spikes that π phase steps at spectral zeros produce.
    """
    gd = np.gradient(phase, x)
    ok = np.isfinite(gd) & (weight > 0)
    if ok.sum() < 3:
        return 0.0, 0.0
    g, w = gd[ok], weight[ok]
    order = np.argsort(g)
    cw = np.cumsum(w[order])
    cw /= cw[-1]
    lo = g[order][min(np.searchsorted(cw, q), len(g) - 1)]
    hi = g[order][min(np.searchsorted(cw, 1 - q), len(g) - 1)]
    return float(lo), float(hi)


def simulate(
    spec: Spectrum,
    phase_coeffs: dict[int, float],
    layers: Sequence[Layer] = (),
    include_absorption: bool = True,
    max_points: int = MAX_POINTS,
) -> Result:
    warnings: list[str] = []
    w0 = spec.omega0
    lo, hi = spec.support()
    hi_lim = 0.95 * w0  # never let ω go to or below zero
    lo = max(lo, -hi_lim)
    if spec.shape == "custom" and (lo > 0 or hi < 0):
        warnings.append("Centre wavelength lies outside the measured spectrum.")

    def mat_phase(dw):
        om = w0 + dw
        tot = np.zeros_like(dw)
        for L in layers:
            tot = tot + L.phase(om)
        return tot

    def in_phase(dw):
        return taylor_phase(dw, phase_coeffs) + spec.phase(dw)

    def out_phase(dw):
        return in_phase(dw) + mat_phase(dw)

    with np.errstate(all="ignore"):
        T, dt, tau = _time_window_and_step(spec, (in_phase, out_phase), lo, hi)
    N = int(2 ** np.ceil(np.log2(max(T / dt, 256))))
    if N > max_points:
        N = max_points
        dt = T / N
        if pi / dt < max(abs(lo), abs(hi)):
            warnings.append(
                "Grid size limit reached: the spectral window is truncated, results may be inaccurate."
            )

    dw = 2 * pi * np.fft.fftfreq(N, dt)
    S = spec.intensity(dw)
    S[(dw < -hi_lim)] = 0.0
    mask = S > 1e-14 * S.max()
    A = np.sqrt(S)
    rng = spec.measured_phase_range()
    if rng is not None:
        sig = dw[S > 1e-2 * S.max()]
        if sig.min() < rng[0] or sig.max() > rng[1]:
            to_nm = lambda x: float(omega_to_wl_nm(w0 + x))
            warnings.append(
                f"The measured phase covers {to_nm(rng[1]):.0f}–{to_nm(rng[0]):.0f} nm but the spectrum "
                f"(> 1 % of peak) spans {to_nm(sig.max()):.0f}–{to_nm(sig.min()):.0f} nm; "
                "the phase is held constant outside its range."
            )

    phi_in = np.zeros(N)
    phi_in[mask] = in_phase(dw[mask])
    phi_mat = np.zeros(N)
    log_amp = np.zeros(N)
    if layers:
        with np.errstate(all="ignore"):
            phi_mat[mask] = mat_phase(dw[mask])
            if include_absorption:
                log_amp[mask] = sum(L.log_amplitude(w0 + dw[mask]) for L in layers)
        bad = mask & ~np.isfinite(phi_mat)
        if bad.any():
            frac = S[bad].sum() / S[mask].sum()
            warnings.append(
                f"Refractive index is undefined (NaN) for {frac:.1%} of the spectral energy "
                "(e.g. a formula pole) — that part of the spectrum was discarded."
            )
            A[bad] = 0.0
            phi_mat[bad] = 0.0
            log_amp[bad] = 0.0
        # warn when the spectrum extends outside the material data range
        sig = S > 1e-3 * S.max()
        wl_sig = omega_to_wl_nm(w0 + dw[sig]) * 1e-3
        for L in layers:
            sigma = getattr(L.material, "n_sigma", 0.0)
            dphi = 2 * pi * L.thickness_mm * 1e3 * sigma / (spec.center_nm * 1e-3)
            if dphi > 0.3:
                warnings.append(
                    f"{L.material.name}: tabulated n is only given to ±{sigma * np.sqrt(3):.0e}, "
                    f"i.e. ~{dphi:.1f} rad of phase uncertainty over {L.thickness_mm:g} mm. "
                    "Dispersion from this data set is unreliable; prefer a formula-based entry."
                )
            elif str(getattr(L.material, "n_kind", "")).startswith("tabulated"):
                warnings.append(
                    f"{L.material.name}: n is tabulated, so GDD/TOD come from an interpolated "
                    "table and can be inaccurate. Compare with a formula-based entry if one exists."
                )
            a, b = L.material.n_range
            if wl_sig.min() < a or wl_sig.max() > b:
                warnings.append(
                    f"{L.material.name}: spectrum ({wl_sig.min()*1e3:.0f}–{wl_sig.max()*1e3:.0f} nm) "
                    f"extends outside the data range ({a*1e3:.0f}–{b*1e3:.0f} nm); values are extrapolated."
                )

    phi_out = phi_in + phi_mat
    A_out = A * np.exp(log_amp)

    def to_time(amp, phi, gd_ref):
        # remove the linear phase that centres the pulse in the window (middle of the
        # GD range); the returned shift puts t=0 at gd_ref, the group delay of ω0
        order = np.argsort(dw)
        g_lo, g_hi = _gd_range(dw[order], phi[order], (amp**2)[order])
        gd_mid = 0.5 * (g_lo + g_hi)
        E = np.fft.fftshift(np.fft.fft(amp * np.exp(1j * (phi - gd_mid * dw))))
        return E, gd_mid - gd_ref

    # group delay at ω0 of the smooth (Taylor + material) phase; a source phase such
    # as SPM has its linear part removed already, so it does not shift the frame
    h = 1e-5 * w0
    with np.errstate(all="ignore"):
        gd_mat0 = float((mat_phase(np.array([h])) - mat_phase(np.array([-h])))[0] / (2 * h))
    if not np.isfinite(gd_mat0):
        gd_mat0 = 0.0

    t0 = (np.arange(N) - N // 2) * dt
    E_tl, _ = to_time(A, np.zeros(N), 0.0)
    E_in, sh_in = to_time(A, phi_in, 0.0)
    E_out, sh_out = to_time(A_out, phi_out, gd_mat0)
    # each pulse is centred in the window; its own axis puts t=0 at the GD of ω0
    t_in, t_out = t0 + sh_in, t0 + sh_out

    norm = np.abs(E_tl).max()
    E_tl, E_in, E_out = E_tl / norm, E_in / norm, E_out / norm
    I_tl, I_in, I_out = (np.abs(E) ** 2 for E in (E_tl, E_in, E_out))
    if I_in[0] > 1e-3 * I_in.max() or I_out[0] > 1e-3 * I_out.max():
        warnings.append("The pulse fills the time window; FWHM may be unreliable.")

    transmission = float((A_out**2).sum() / (A**2).sum())
    shift = np.fft.fftshift
    phase_disp_in = np.where(shift(mask), shift(phi_in), np.nan)
    phase_disp_out = np.where(shift(mask), shift(phi_out), np.nan)

    layer_disp = [dispersion_coefficients(L.phase, w0) for L in layers]
    in_disp = [0.0] + [float(phase_coeffs.get(n, 0.0)) for n in (2, 3, 4)]

    return Result(
        t=t0,
        t_in=t_in,
        t_out=t_out,
        E_in=E_in,
        E_out=E_out,
        E_tl=E_tl,
        dw=shift(dw),
        omega0=w0,
        spectrum_in=shift(A**2),
        spectrum_out=shift(A_out**2),
        phase_in=_remove_linear(shift(dw), phase_disp_in, shift(A**2)),
        phase_out=_remove_linear(shift(dw), phase_disp_out, shift(A**2)),
        fwhm_tl=fwhm(t0, I_tl),
        fwhm_in=fwhm(t0, I_in),
        fwhm_out=fwhm(t0, I_out),
        rms_in=rms_width(t0, I_in),
        rms_out=rms_width(t0, I_out),
        transmission=transmission,
        peak_in_rel_tl=float(I_in.max()),
        peak_out_rel_tl=float(I_out.max()),
        layer_dispersion=layer_disp,
        input_dispersion=in_disp,
        n_points=N,
        dt=dt,
        warnings=warnings,
    )


def _remove_linear(dw, phase, weight):
    """Remove constant + linear part (intensity-weighted fit) from a spectral phase for display."""
    ok = np.isfinite(phase) & (weight > 1e-4 * np.nanmax(weight))
    if ok.sum() < 3:
        return phase
    p = np.polyfit(dw[ok], phase[ok], 1, w=np.sqrt(weight[ok]))
    return phase - np.polyval(p, dw)
