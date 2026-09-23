import numpy as np
import pytest

from dispersion import pulse, ridb

pytestmark = pytest.mark.skipif(not ridb.database_available(), reason="database not downloaded")


@pytest.fixture(scope="module")
def catalog():
    return {e.key: e for e in ridb.load_catalog()}


@pytest.mark.parametrize(
    "key, n800, gdd800",
    [
        ("main/SiO2/Malitson", 1.45332, 36.16),
        ("popular_glass/BK7/SCHOTT", 1.51078, 44.65),
        ("main/Al2O3/Malitson-o", 1.76013, 58.04),
        ("main/CaF2/Malitson", 1.43053, 27.87),
        ("main/H2O/Daimon-20.0C", 1.32861, 24.88),
    ],
)
def test_known_materials(catalog, key, n800, gdd800):
    m = ridb.load_material(catalog[key])
    assert float(m.n(0.8)) == pytest.approx(n800, abs=2e-4)
    _, gdd, _, _ = pulse.dispersion_coefficients(pulse.Layer(m, 1.0).phase, pulse.wl_nm_to_omega(800))
    assert gdd == pytest.approx(gdd800, rel=5e-3)


def test_tabulated_absorbing_material(catalog):
    m = ridb.load_material(catalog["main/H2O/Hale"])
    assert m.has_k and m.n_kind == "tabulated nk"
    r = pulse.simulate(pulse.Spectrum(1450, "gaussian", 60), {}, [pulse.Layer(m, 1.0)])
    assert 0.2 < r.transmission < 0.5  # strong OH overtone absorption near 1.45 µm
    assert any("tabulated" in w for w in r.warnings)


def test_whole_catalog_parses(catalog):
    failures = []
    for e in catalog.values():
        try:
            m = ridb.load_material(e)
        except ValueError:
            continue  # k-only entries
        a, b = m.n_range
        n = m.n(np.linspace(a, b, 20))
        if not np.isfinite(n).any():
            failures.append(e.key)
    assert not failures
