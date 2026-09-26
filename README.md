# Pulse Dispersion

Simulates the linear propagation of an ultrashort laser pulse through a stack of
dispersive materials, using real refractive-index data from the full
[refractiveindex.info database](https://github.com/polyanskiy/refractiveindex.info-database)
(3,500+ entries, CC0 public domain).

## Run

```
python -m venv .venv
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\streamlit run app.py
```

or double-click `run.bat`. On first start the app downloads the refractiveindex.info
database automatically (about 10 s, 60 MB on disk in `data/database`). Use
**Refractive-index database → Update** in the sidebar to refresh it later.

Downloads are rate-limited so a shared deployment can't hammer GitHub:
* an existing database can be refreshed at most **once per 24 hours**;
* while no database exists, attempts are spaced **10 minutes** apart;
* only one download runs at a time.

The limit is tracked in `data/`. A fresh cloud container starts without a database, so it
downloads once on its first visit.

Run the tests with `.venv\Scripts\python -m pytest`.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (`data/` and `.venv/` are git-ignored).
2. At [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, choose
   **Create app → Deploy a public app from GitHub**, then pick the repository,
   branch `main` and main file `app.py`. Under **Advanced settings**, choose Python 3.12 or newer.
3. The first visit after each container start downloads the database (about 10 s).
   Every push to `main` redeploys the app.

For a private app, keep the repository private and invite viewers by email in the app's
**Share** settings.

## What it does

* **Input pulse:** Gaussian, sech², super-Gaussian, or a measured spectrum file
  (columns: wavelength in nm, µm or m, then intensity). Spectral phase is a Taylor series
  φ(ω) = Σ φₙ/n! (ω−ω₀)ⁿ with GDD, TOD, FOD and 5th order.
* **Measured spectral phase** (e.g. from FROG, SPIDER or d-scan). It can come from a 3rd
  column of the spectrum file or from a separate (wavelength, phase) file:
  * units are rad or deg, and a wrapped phase can be unwrapped;
  * it is added to the Taylor phase;
  * a sidebar caption shows its fitted GDD/TOD, as a check of the sign convention. Use
    *Flip phase sign* if your software uses the opposite one.

  Outside the measured range the phase is held constant, with a warning.
* **SPM-broadened spectrum** (e.g. a Yb laser such as a 300 fs CARBIDE after a multipass
  cell). A transform-limited seed (sech² or Gaussian) gets E(t) = √I(t)·exp(iB·I(t)/I₀).
  Set either the B-integral or a target compressed FWHM, and the app solves for B.
  The pulse's spectral phase can be:
  * **ideally compressed:** flat phase;
  * **best GDD only:** as with chirped mirrors, leaving the residual non-quadratic SPM
    phase; the app reports the optimal compressor GDD;
  * **uncompressed:** the full SPM chirp, straight out of the cell.

  This is a pure-SPM model: dispersion inside the cell, self-steepening and spatial
  (mode-averaging) effects are neglected. Real multipass-cell spectra are usually somewhat
  smoother; use *Measured spectrum* when you have one.
* **Materials:** any number of layers, each chosen from the database, with a thickness.
  The exact phase ω·n(ω)·L/c is applied, so every dispersion order is included.
  Absorption from k is applied when the data has it.
* **Output:** input, output and transform-limited intensity vs time with FWHM
  (outermost half-maximum crossings), RMS width, peak power, transmission,
  spectrum with spectral phase or group delay, and a table of GD/GDD/TOD/FOD per layer.
  **Pre-compensate** sets the input phase to cancel the materials' GDD/TOD/FOD.

## Conventions

* E(t) = ∫ Ẽ(ω) e^{−iωt} dω, with phase applied as Ẽ·e^{+iφ}. Positive GDD (as in glass)
  delays the blue side, giving an up-chirp.
* Bandwidth is the intensity FWHM in wavelength; the spectral shape is defined in frequency.
* Each pulse is plotted in its own frame, with t = 0 at the group delay of λ₀.
  The absolute group delay through the stack is in the *Material dispersion* table.
* The time/frequency grid sizes itself from the group-delay spread, up to 2²² points.

## Data caveats

* **Formula entries** (Sellmeier and similar) give the most reliable dispersion.
  The picker labels each data set as *formula* or *table*.
* **Tabulated n** is fitted with a quintic smoothing spline in log λ, matched to the
  table's rounding precision. Second and higher derivatives from tables can still be
  inaccurate, for example with coarse sampling or few digits. The app warns for tabulated
  layers, and warns more strongly when the precision alone implies more than 0.3 rad of
  phase error.
* The app warns when the spectrum extends past a material's valid wavelength range
  (formulas are extrapolated, tables are clamped) and when n is undefined (a formula pole).
* Propagation is linear only: no SPM or other nonlinear effects.

## Layout

```
app.py                  Streamlit UI
dispersion/pulse.py     physics: spectrum, phase, FFT propagation, FWHM
dispersion/ridb.py      database download, catalog, formula / table evaluation
tests/                  pytest (physics vs analytic results; database vs published values)
```
