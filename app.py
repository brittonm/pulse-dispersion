"""Streamlit front end: run with  `streamlit run app.py`."""

from __future__ import annotations

import io
from collections import OrderedDict
from math import pi

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dispersion import pulse, ridb
from dispersion.pulse import Layer, Spectrum

st.set_page_config(page_title="Pulse Dispersion", page_icon="〰️", layout="wide")

COLORS = {"in": "#1f77b4", "out": "#d62728", "tl": "#7f7f7f"}
DEFAULT_LAYER = ("main", "SiO2", "Malitson")
MAX_POINTS = 2**21  # keeps memory well inside Streamlit Community Cloud's ~1 GB limit


# --------------------------------------------------------------------------
# Database (cached)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Reading material catalog…")
def catalog_index(stamp: str | None):
    """shelf -> (shelf name, book -> (book name, group, [entries]))"""
    shelves: OrderedDict = OrderedDict()
    for e in ridb.load_catalog():
        books = shelves.setdefault(e.shelf, (e.shelf_name, OrderedDict()))[1]
        books.setdefault(e.book, (e.book_name, e.book_group, []))[2].append(e)
    return shelves


@st.cache_resource(show_spinner=False, max_entries=256)
def get_material(path: str, name: str) -> ridb.Material:
    return ridb.Material.from_file(ridb.DEFAULT_DB_ROOT / "data" / path, name=name)


@st.cache_data(show_spinner=False, max_entries=10000)
def page_kind(path: str) -> str:
    return ridb.data_kind(ridb.CatalogEntry("", "", "", "", "", "", "", "", path))


def database_panel():
    with st.sidebar.expander("Refractive-index database", expanded=not ridb.database_available()):
        if ridb.database_available():
            st.caption(
                f"refractiveindex.info database (CC0) — downloaded "
                f"{ridb.database_timestamp() or 'unknown date'}."
            )
            label = "Update database"
        else:
            st.warning("The refractiveindex.info database has not been downloaded yet.")
            label = "Download database (~60 MB on disk)"
        if st.button(label, width="stretch"):
            bar = st.progress(0.0, text="Downloading…")

            def progress(done, total):
                frac = min(done / total, 1.0) if total else 0.0
                bar.progress(frac, text=f"Downloading… {done / 1e6:.1f} MB")

            try:
                ridb.download_database(progress=progress)
            except Exception as exc:  # network etc.
                st.error(f"Download failed: {exc}")
            else:
                catalog_index.clear()
                get_material.clear()
                page_kind.clear()
                st.rerun()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def fmt_time(fs: float) -> str:
    if not np.isfinite(fs):
        return "n/a"
    if abs(fs) >= 1e4:
        return f"{fs / 1e3:,.2f} ps"
    if abs(fs) >= 100:
        return f"{fs:,.1f} fs"
    return f"{fs:.2f} fs"


def parse_spectrum_file(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Two numeric columns (wavelength, intensity); header/comment lines are skipped."""
    rows = []
    for line in data.decode("utf-8", errors="ignore").splitlines():
        parts = line.replace(",", " ").replace(";", " ").replace("\t", " ").split()
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except (ValueError, IndexError):
            continue
    if len(rows) < 3:
        raise ValueError("Could not find two numeric columns (wavelength, intensity).")
    arr = np.array(rows)
    return arr[:, 0], arr[:, 1]


def crop_and_decimate(x, y_list, rel=1e-4, margin=0.25, max_pts=4000):
    """Crop to where any y > rel·max, then max-in-bin decimate for plotting."""
    ref = np.max(np.vstack(y_list), axis=0)
    idx = np.nonzero(ref > rel * ref.max())[0]
    i0, i1 = idx[0], idx[-1]
    pad = int((i1 - i0) * margin) + 2
    i0, i1 = max(i0 - pad, 0), min(i1 + pad, len(x) - 1)
    x = x[i0 : i1 + 1]
    ys = [y[i0 : i1 + 1] for y in y_list]
    step = int(np.ceil(len(x) / max_pts))
    if step > 1:
        n = len(x) // step * step
        x = x[:n].reshape(-1, step).mean(axis=1)
        ys = [y[:n].reshape(-1, step).max(axis=1) for y in ys]
    return x, ys


def time_unit(span_fs: float):
    return (1e3, "ps") if span_fs > 2e4 else (1.0, "fs")


# --------------------------------------------------------------------------
# Sidebar: pulse definition
# --------------------------------------------------------------------------
def pulse_inputs() -> tuple[Spectrum, dict[int, float]]:
    sb = st.sidebar
    sb.header("Input pulse")
    shape = sb.selectbox(
        "Spectral shape",
        ["gaussian", "sech2", "supergaussian", "spm", "custom"],
        format_func={
            "gaussian": "Gaussian",
            "sech2": "sech²",
            "supergaussian": "Super-Gaussian",
            "spm": "SPM-broadened (e.g. multipass cell)",
            "custom": "Measured spectrum (file)",
        }.get,
        key="shape", on_change=_shape_changed,
    )
    custom_wl = custom_I = None
    per_wl = True
    if shape == "custom":
        up = sb.file_uploader("Spectrum file (λ, intensity)", type=["csv", "txt", "dat", "tsv"])
        unit = sb.radio("Wavelength column unit", ["nm", "µm"], horizontal=True)
        per_wl = sb.checkbox(
            "Intensity is per unit wavelength (spectrometer)", value=True,
            help="Converts S(λ) to S(ω) with the λ² Jacobian.",
        )
        if up is None:
            sb.info("Upload a two-column text/CSV file to continue.")
            st.stop()
        try:
            custom_wl, custom_I = parse_spectrum_file(up.getvalue())
        except ValueError as exc:
            sb.error(str(exc))
            st.stop()
        if unit == "µm":
            custom_wl = custom_wl * 1e3
        w = np.clip(custom_I, 0, None)
        centroid = float((custom_wl * w).sum() / w.sum())
        sb.caption(f"Spectrum centroid: {centroid:.1f} nm")
        if sb.button("Use centroid as centre wavelength"):
            st.session_state.center_nm = round(centroid, 2)

    center = sb.number_input(
        "Centre wavelength λ₀ (nm)", min_value=100.0, max_value=20000.0, step=10.0,
        key="center_nm", help="Reference wavelength for the spectral-phase Taylor series.",
    )
    fwhm_nm, sg = 30.0, 2
    spm = {}
    if shape == "spm":
        spm = spm_inputs(center)
    elif shape != "custom":
        _keep("fwhm_nm")
        fwhm_nm = sb.number_input(
            "Bandwidth, intensity FWHM (nm)", min_value=0.01, max_value=1.9 * center,
            step=1.0, key="fwhm_nm",
        )
        _remember("fwhm_nm")
        if shape == "supergaussian":
            sg = sb.slider("Super-Gaussian order", 2, 10, 3)

    sb.subheader("Input spectral phase")
    sb.caption("φ(ω) = Σ φₙ/n! · (ω−ω₀)ⁿ. Positive GDD = up-chirp (like glass)."
               + (" Added on top of the SPM phase chosen above." if shape == "spm" else ""))
    c1, c2 = sb.columns(2)
    coeffs = {
        2: c1.number_input("GDD (fs²)", step=10.0, key="gdd", format="%.1f"),
        3: c2.number_input("TOD (fs³)", step=100.0, key="tod", format="%.1f"),
        4: c1.number_input("FOD (fs⁴)", step=1000.0, key="fod", format="%.1f"),
        5: c2.number_input("5th (fs⁵)", step=1e4, key="fifth", format="%.1f"),
    }
    sb.button("Reset phase to zero", on_click=_reset_phase, width="stretch")
    spec = Spectrum(center, shape, fwhm_nm, sg, custom_wl, custom_I, per_wl, **spm)
    return spec, coeffs


SPM_PHASE_LABELS = {
    "gdd": "Compressed with best GDD only (chirped mirrors)",
    "flat": "Ideally compressed (flat phase)",
    "full": "Uncompressed (full SPM chirp)",
}


WIDGET_DEFAULTS = {
    "fwhm_nm": 30.0,
    "spm_seed_fs": 300.0,
    "spm_seed_shape": "sech2",
    "spm_by": "Target compressed FWHM",
    "spm_b": 15.0,
    "spm_target_fs": 25.0,
    "spm_phase": "gdd",
}


def _keep(*keys):
    """Give conditionally shown widgets their default, or their last value, before they render.

    Streamlit drops a widget's state while it is not rendered, and only honours a
    preset value when it is set in the same run as the widget first appears.
    """
    ss = st.session_state
    for key in keys:
        if key not in ss:
            ss[key] = ss.get(f"_last_{key}", WIDGET_DEFAULTS[key])


def _remember(*keys):
    for key in keys:
        if key in st.session_state:
            st.session_state[f"_last_{key}"] = st.session_state[key]


def spm_inputs(center_nm: float) -> dict:
    """Seed pulse + nonlinear phase → SPM-broadened spectrum (e.g. a multipass cell)."""
    sb = st.sidebar
    spm_keys = [k for k in WIDGET_DEFAULTS if k.startswith("spm_")]
    _keep(*spm_keys)
    c1, c2 = sb.columns(2)
    seed_fs = c1.number_input("Seed FWHM (fs)", min_value=5.0, max_value=1e5, step=10.0, key="spm_seed_fs")
    seed_shape = c2.selectbox("Seed shape", ["sech2", "gaussian"], key="spm_seed_shape",
                              format_func={"sech2": "sech²", "gaussian": "Gaussian"}.get)
    by = sb.radio("Set broadening by", ["Target compressed FWHM", "B-integral"], horizontal=True,
                  key="spm_by")
    try:
        if by == "B-integral":
            b = sb.number_input("B-integral, peak nonlinear phase (rad)", min_value=0.0,
                                max_value=500.0, step=1.0, key="spm_b")
        else:
            target = sb.number_input("Target transform-limited FWHM (fs)", min_value=1.0,
                                     step=1.0, key="spm_target_fs")
            b = pulse.spm_b_for_target(float(seed_fs), seed_shape, float(target))
        mode = sb.selectbox("Spectral phase of the pulse", list(SPM_PHASE_LABELS),
                            format_func=SPM_PHASE_LABELS.get, key="spm_phase")
        tl = pulse.spm_tl_fwhm(float(seed_fs), seed_shape, float(b))
        gdd = pulse.spm_compressor_gdd(float(seed_fs), seed_shape, float(b))
    except ValueError as exc:
        sb.error(str(exc))
        st.stop()
    dw, S, _ = pulse.spm_spectrum(float(seed_fs), seed_shape, float(b))
    w0 = pulse.wl_nm_to_omega(center_nm)
    if np.abs(dw[S > 1e-6]).max() > 0.8 * w0:
        sb.error("This much broadening would span close to an octave or beyond (B = "
                 f"{b:.0f} rad), where a pure-SPM envelope model is not valid. "
                 "Use a longer target duration or a smaller B-integral.")
        st.stop()
    wl = pulse.omega_to_wl_nm(w0 + dw[S > 1e-2])
    sb.caption(
        f"B = {b:.2f} rad · TL FWHM {fmt_time(tl)} · spectrum (1 % level) "
        f"{wl.min():.0f}–{wl.max():.0f} nm · best GDD-only compressor {gdd:,.0f} fs². "
        "Pure SPM of a transform-limited seed: no dispersion or spatial effects inside the cell."
    )
    _remember(*spm_keys)
    return dict(spm_seed_fs=float(seed_fs), spm_seed_shape=seed_shape, spm_b=float(b), spm_phase=mode)


def _shape_changed():
    if st.session_state.shape == "spm" and st.session_state.center_nm == 800.0:
        st.session_state.center_nm = 1030.0  # Yb lasers (e.g. CARBIDE) + multipass cells


def _reset_phase():
    for k in ("gdd", "tod", "fod", "fifth"):
        st.session_state[k] = 0.0


def _precompensate():
    tot = st.session_state.get("_material_disp")
    if tot:
        st.session_state.gdd = -round(tot[1], 1)
        st.session_state.tod = -round(tot[2], 1)
        st.session_state.fod = -round(tot[3], 1)
        st.session_state.fifth = 0.0


# --------------------------------------------------------------------------
# Material stack
# --------------------------------------------------------------------------
def _new_layer(shelf, book, page, thickness=10.0):
    st.session_state.layer_seq += 1
    lid = st.session_state.layer_seq
    st.session_state.layers.append(lid)
    st.session_state[f"shelf_{lid}"] = shelf
    st.session_state[f"book_{lid}"] = book
    st.session_state[f"page_{lid}"] = page
    st.session_state[f"thick_{lid}"] = thickness


def _remove_layer(lid):
    st.session_state.layers.remove(lid)


def _add_from_search(index):
    key = st.session_state.get("quick_search")
    if key:
        shelf, book, page = key.split("/", 2)
        _new_layer(shelf, book, page)
        st.session_state.quick_search = None


def material_stack(index) -> list[Layer]:
    st.subheader("Dispersive materials")
    all_keys = [e.key for _, (_, books) in index.items() for _, (_, _, es) in books.items() for e in es]
    labels = {
        e.key: e.label for _, (_, books) in index.items() for _, (_, _, es) in books.items() for e in es
    }
    top = st.columns([3, 1])
    top[0].selectbox(
        f"Search all {len(all_keys):,} entries and add as a layer",
        all_keys, index=None, key="quick_search", format_func=labels.get,
        placeholder="Type e.g. 'sapphire', 'BK7', 'water', 'air', 'ZnSe'…",
        on_change=_add_from_search, args=(index,),
    )
    top[1].write("")
    top[1].button("Add blank layer", width="stretch", on_click=_new_layer, args=DEFAULT_LAYER)

    layers: list[Layer] = []
    if not st.session_state.layers:
        st.info("No material: output = input. Add a layer above.")
    for lid in list(st.session_state.layers):
        cols = st.columns([2, 3, 3, 1.4, 0.5], vertical_alignment="bottom")
        shelves = list(index)
        sk = f"shelf_{lid}"
        if st.session_state.get(sk) not in index:
            st.session_state[sk] = shelves[0]
        shelf = cols[0].selectbox("Shelf", shelves, key=sk, format_func=lambda s, index=index: index[s][0])
        books = index[shelf][1]
        bk = f"book_{lid}"
        if st.session_state.get(bk) not in books:
            st.session_state[bk] = next(iter(books))
        book = cols[1].selectbox(
            "Material", list(books), key=bk,
            format_func=lambda b, books=books: books[b][0] + (f"  ·  {books[b][1]}" if books[b][1] else ""),
        )
        pages = {e.page: e for e in books[book][2]}
        pk = f"page_{lid}"
        if st.session_state.get(pk) not in pages:
            st.session_state[pk] = next(iter(pages))
        kinds = {p: page_kind(e.path) for p, e in pages.items()}
        page = cols[2].selectbox(
            "Data set", list(pages), key=pk,
            format_func=lambda p, pages=pages, kinds=kinds: pages[p].page_name
            + (f"  ·  {kinds[p]}" if kinds[p] else ""),
            help="'formula' entries (Sellmeier etc.) give the most reliable dispersion.",
        )
        thick = cols[3].number_input("Thickness (mm)", min_value=0.0, step=1.0, key=f"thick_{lid}")
        cols[4].button("✕", key=f"rm_{lid}", on_click=_remove_layer, args=(lid,), help="Remove layer")

        entry = pages[page]
        try:
            mat = get_material(entry.path, f"{entry.book_name} ({entry.page})")
        except Exception as exc:
            st.error(f"{entry.label}: {exc}")
            continue
        if thick > 0:
            layers.append(Layer(mat, thick))
    return layers


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_time(r: pulse.Result, normalize_each: bool, show_tl: bool, log_y: bool):
    I_in, I_out, I_tl = r.I_in, r.I_out, r.I_tl
    if normalize_each:
        I_in, I_out = I_in / I_in.max(), I_out / I_out.max()
    span = max(np.ptp(r.t_out[I_out > 1e-3 * I_out.max()]), np.ptp(r.t_in[I_in > 1e-3 * I_in.max()]))
    scale, unit = time_unit(span)
    fig = go.Figure()
    traces = [("Input", r.t_in, I_in, "in", r.fwhm_in), ("Output", r.t_out, I_out, "out", r.fwhm_out)]
    if show_tl:
        traces.append(("Transform limit", r.t, I_tl, "tl", r.fwhm_tl))
    for name, t, I, c, fw in traces:
        x, (y,) = crop_and_decimate(t, [I], rel=1e-6 if log_y else 1e-4)
        fig.add_trace(
            go.Scatter(
                x=x / scale, y=y, name=f"{name} — FWHM {fmt_time(fw)}", mode="lines",
                line=dict(color=COLORS[c], width=2, dash="dash" if c == "tl" else "solid"),
            )
        )
    fig.update_layout(
        xaxis_title=f"Time ({unit})  — t = 0 at the group delay of λ₀",
        yaxis_title="Intensity (norm.)" if normalize_each else "Intensity (rel. to transform limit)",
        height=480, margin=dict(l=10, r=10, t=30, b=10), legend=dict(x=0.01, y=0.99),
        hovermode="x unified",
    )
    if log_y:
        fig.update_yaxes(type="log", range=[-5, 0.1 if normalize_each else None])
    st.plotly_chart(fig, width="stretch")


def plot_spectrum(r: pulse.Result, what: str):
    wl = r.wavelength_nm
    ok = np.isfinite(wl) & (wl > 0)
    # per-wavelength density, as a spectrometer would show
    S_in = np.where(ok, r.spectrum_in / np.where(ok, wl, 1) ** 2, 0)
    S_out = np.where(ok, r.spectrum_out / np.where(ok, wl, 1) ** 2, 0)
    norm = S_in.max()
    sig = r.spectrum_in > 1e-4 * r.spectrum_in.max()
    idx = np.nonzero(sig)[0]
    sl = slice(idx[0], idx[-1] + 1)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=wl[sl], y=S_in[sl] / norm, name="Input spectrum",
                             line=dict(color="#bbbbbb"), fill="tozeroy"), secondary_y=False)
    if r.transmission < 0.9999:
        fig.add_trace(go.Scatter(x=wl[sl], y=S_out[sl] / norm, name="Transmitted spectrum",
                                 line=dict(color="#888888", dash="dot")), secondary_y=False)
    dw = r.dw[sl]
    for name, ph, c in (("input", r.phase_in[sl], "in"), ("output", r.phase_out[sl], "out")):
        if what == "Group delay":
            y = np.gradient(ph, dw)
            y = y - np.interp(0.0, dw, y)  # relative to GD at λ₀
            label = f"Group delay, {name} (fs)"
        else:
            y = ph
            label = f"Spectral phase, {name} (rad)"
        fig.add_trace(go.Scatter(x=wl[sl], y=y, name=label, line=dict(color=COLORS[c])), secondary_y=True)
    fig.update_xaxes(title="Wavelength (nm)")
    fig.update_yaxes(title="Spectral intensity (norm.)", secondary_y=False, rangemode="tozero")
    fig.update_yaxes(
        title="Group delay rel. to λ₀ (fs)" if what == "Group delay" else "Phase (rad, linear part removed)",
        secondary_y=True, showgrid=False,
    )
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")


def material_tab(r: pulse.Result, layers: list[Layer]):
    if not layers:
        st.info("No materials in the stack.")
        return
    rows = []
    for L, (gd, gdd, tod, fod) in zip(layers, r.layer_dispersion):
        rows.append({
            "Material": L.material.name, "Thickness (mm)": L.thickness_mm,
            "Group delay (ps)": gd / 1e3, "GDD (fs²)": gdd, "TOD (fs³)": tod, "FOD (fs⁴)": fod,
            "GDD/mm": gdd / L.thickness_mm, "TOD/mm": tod / L.thickness_mm,
            "n(λ₀)": float(L.material.n(pulse.omega_to_wl_nm(r.omega0) * 1e-3)),
        })
    gd, gdd, tod, fod = np.sum(r.layer_dispersion, axis=0)
    tot = r.total_dispersion
    rows.append({"Material": "Σ materials", "Thickness (mm)": sum(L.thickness_mm for L in layers),
                 "Group delay (ps)": gd / 1e3, "GDD (fs²)": gdd, "TOD (fs³)": tod, "FOD (fs⁴)": fod})
    rows.append({"Material": "Input phase (Taylor terms)", "GDD (fs²)": r.input_dispersion[1],
                 "TOD (fs³)": r.input_dispersion[2], "FOD (fs⁴)": r.input_dispersion[3]})
    rows.append({"Material": "Net (input + materials)", "GDD (fs²)": tot[1], "TOD (fs³)": tot[2],
                 "FOD (fs⁴)": tot[3]})
    st.dataframe(rows, width="stretch", hide_index=True, column_config={
        c: st.column_config.NumberColumn(format="%.4g")
        for c in ["Group delay (ps)", "GDD (fs²)", "TOD (fs³)", "FOD (fs⁴)", "GDD/mm", "TOD/mm", "n(λ₀)"]
    })
    st.caption("Taylor coefficients at λ₀, for reference only: the simulation applies the exact "
               "phase ω·n(ω)·L/c, which includes every order.")

    # n(λ) and GVD over the pulse spectrum
    wl = r.wavelength_nm
    sig = np.isfinite(wl) & (r.spectrum_in > 1e-4 * r.spectrum_in.max())
    lo, hi = wl[sig].min(), wl[sig].max()
    pad = 0.15 * (hi - lo)
    wl_nm = np.linspace(max(lo - pad, 1.0), hi + pad, 600)
    om = pulse.wl_nm_to_omega(wl_nm)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Refractive index n", "GVD (fs²/mm)"))
    for i, L in enumerate(layers):
        n = L.material.n(wl_nm * 1e-3)
        beta = om * n / pulse.C_UM_PER_FS * 1e3  # rad/mm
        gvd = np.gradient(np.gradient(beta, om), om)
        name = L.material.name
        fig.add_trace(go.Scatter(x=wl_nm, y=n, name=name, legendgroup=name,
                                 line=dict(color=f"hsl({(i * 67) % 360},60%,45%)")), 1, 1)
        fig.add_trace(go.Scatter(x=wl_nm, y=gvd, name=name, legendgroup=name, showlegend=False,
                                 line=dict(color=f"hsl({(i * 67) % 360},60%,45%)")), 1, 2)
    for c in (1, 2):
        fig.add_vrect(x0=lo, x1=hi, fillcolor="#999", opacity=0.08, line_width=0, row=1, col=c)
        fig.update_xaxes(title="Wavelength (nm)", row=1, col=c)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    st.caption("Shaded: the pulse spectrum (> 10⁻⁴ of peak).")

    for L in layers:
        m = L.material
        with st.expander(f"About: {m.name}"):
            a, b = m.n_range
            st.markdown(
                f"**Data:** {m.n_kind}, n valid {a * 1e3:,.0f}–{b * 1e3:,.0f} nm"
                + (" · includes absorption (k)" if m.has_k else " · no k data (lossless)")
            )
            if m.comments:
                st.markdown(f"**Comments:** {m.comments}")
            if m.conditions:
                st.markdown("**Conditions:** " + ", ".join(f"{k}: {v}" for k, v in m.conditions.items()))
            if m.references:
                st.markdown(f"**References:** {m.references}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def init_state():
    ss = st.session_state
    if "layers" not in ss:
        ss.layers, ss.layer_seq = [], 0
        ss.center_nm = 800.0
        ss.gdd = ss.tod = ss.fod = ss.fifth = 0.0
        _new_layer(*DEFAULT_LAYER, thickness=10.0)


@st.cache_resource(show_spinner="Downloading the refractiveindex.info database (first start only)…")
def ensure_database() -> str:
    """Fetch the database once per server process if it is missing (e.g. on a fresh cloud container)."""
    if not ridb.database_available():
        try:
            ridb.download_database()
        except Exception as exc:  # offline etc.: the sidebar button can retry
            return f"Automatic database download failed: {exc}"
    return ""


def main():
    init_state()
    st.title("Ultrashort pulse dispersion")
    if problem := ensure_database():
        st.warning(problem)
        ensure_database.clear()
    database_panel()
    spec, coeffs = pulse_inputs()

    if not ridb.database_available():
        st.info("Download the refractive-index database from the sidebar to add materials.")
        layers = []
    else:
        layers = material_stack(catalog_index(ridb.database_timestamp()))
    absorb = st.checkbox("Include absorption (k) where the data has it", value=True)

    try:
        r = pulse.simulate(spec, coeffs, layers, include_absorption=absorb, max_points=MAX_POINTS)
    except Exception as exc:
        st.error(f"Simulation failed: {exc}")
        st.stop()
    st.session_state._material_disp = [sum(d[i] for d in r.layer_dispersion) for i in range(4)]
    for w in r.warnings:
        st.warning(w)

    st.divider()
    m = st.columns(5)
    m[0].metric("Transform-limited FWHM", fmt_time(r.fwhm_tl))
    m[1].metric("Input FWHM", fmt_time(r.fwhm_in),
                f"×{r.fwhm_in / r.fwhm_tl:.2f} TL" if np.isfinite(r.fwhm_in) else None, delta_color="off")
    m[2].metric("Output FWHM", fmt_time(r.fwhm_out),
                f"×{r.fwhm_out / r.fwhm_in:.2f} input" if np.isfinite(r.fwhm_out) else None,
                delta_color="inverse")
    m[3].metric("Output peak power", f"{r.peak_out_rel_tl / r.peak_in_rel_tl:.3f} × input",
                f"{r.peak_out_rel_tl:.3f} × TL", delta_color="off")
    m[4].metric("Transmission", f"{r.transmission:.2%}")

    tabs = st.tabs(["Temporal intensity", "Spectrum & phase", "Material dispersion"])
    with tabs[0]:
        c = st.columns([2, 1, 1, 2])
        norm = c[0].radio("Scaling", ["Relative to transform limit", "Each normalised to 1"],
                          horizontal=True, label_visibility="collapsed")
        show_tl = c[1].checkbox("Show TL", value=False)
        log_y = c[2].checkbox("Log scale", value=False)
        c[3].button("Pre-compensate material GDD/TOD/FOD", on_click=_precompensate,
                    disabled=not layers, width="stretch",
                    help="Set the input phase to the negative of the materials' Taylor coefficients.")
        plot_time(r, norm.startswith("Each"), show_tl, log_y)
        st.caption(f"RMS width: input {fmt_time(r.rms_in)}, output {fmt_time(r.rms_out)} · "
                   f"grid {r.n_points:,} points, dt = {r.dt:.3g} fs")
    with tabs[1]:
        what = st.radio("Show", ["Spectral phase", "Group delay"], horizontal=True)
        plot_spectrum(r, what)
    with tabs[2]:
        material_tab(r, layers)


main()
