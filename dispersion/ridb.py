"""Access to the refractiveindex.info database (https://refractiveindex.info).

The database (CC0 public domain) is downloaded from GitHub as a zip and the
``database/`` folder is extracted locally. ``catalog-nk.yml`` lists every
material page; each page points to a YAML data file holding either a
dispersion formula (types 1-9) or tabulated n / k / nk data.

Wavelengths in the database are in micrometres.
"""

from __future__ import annotations

import datetime as _dt
import re
import shutil
import tempfile
import warnings
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import yaml
from scipy.interpolate import CubicSpline, UnivariateSpline

try:  # the C loader is ~10x faster on the 0.5 MB catalog
    _Loader = yaml.CSafeLoader
except AttributeError:  # pragma: no cover
    _Loader = yaml.SafeLoader

REPO_ZIP_URL = (
    "https://codeload.github.com/polyanskiy/refractiveindex.info-database/zip/refs/heads/main"
)
DEFAULT_DB_ROOT = Path(__file__).resolve().parent.parent / "data" / "database"
_STAMP_FILE = "_downloaded.txt"


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def database_available(root: Path = DEFAULT_DB_ROOT) -> bool:
    return (Path(root) / "catalog-nk.yml").is_file()


def database_timestamp(root: Path = DEFAULT_DB_ROOT) -> Optional[str]:
    stamp = Path(root) / _STAMP_FILE
    return stamp.read_text(encoding="utf-8").strip() if stamp.is_file() else None


def download_database(
    root: Path = DEFAULT_DB_ROOT,
    progress: Optional[Callable[[int, Optional[int]], None]] = None,
    url: str = REPO_ZIP_URL,
) -> Path:
    """Download the database zip from GitHub and extract ``database/`` to *root*.

    *progress(bytes_done, bytes_total_or_None)* is called while downloading.
    The previous copy is only replaced once the new one extracted successfully.
    """
    root = Path(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root.parent) as tmp:
        tmp = Path(tmp)
        zpath = tmp / "db.zip"
        with urllib.request.urlopen(url, timeout=60) as resp, open(zpath, "wb") as out:
            total = resp.headers.get("Content-Length")
            total = int(total) if total else None
            done = 0
            while chunk := resp.read(1 << 16):
                out.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)

        staging = tmp / "database"
        staging_resolved = staging.resolve()
        with zipfile.ZipFile(zpath) as zf:
            for info in zf.infolist():
                parts = info.filename.split("/")
                # <repo>-main/database/...
                if len(parts) < 3 or parts[1] != "database" or info.is_dir():
                    continue
                target = (staging / Path(*parts[2:])).resolve()
                if staging_resolved not in target.parents:  # zip-slip guard
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)

        if not (staging / "catalog-nk.yml").is_file():
            raise RuntimeError("Downloaded archive does not contain database/catalog-nk.yml")
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        (staging / _STAMP_FILE).write_text(stamp, encoding="utf-8")

        if root.exists():
            shutil.rmtree(root)
        shutil.move(str(staging), str(root))
    return root


# --------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------
_TAG_RE = re.compile(r"<[^>]+>")


def _clean(text) -> str:
    return _TAG_RE.sub("", str(text or "")).strip()


@dataclass(frozen=True)
class CatalogEntry:
    shelf: str
    shelf_name: str
    book: str
    book_name: str
    book_group: str
    page: str
    page_name: str
    page_group: str
    path: str  # relative to <root>/data

    @property
    def key(self) -> str:
        return f"{self.shelf}/{self.book}/{self.page}"

    @property
    def label(self) -> str:
        return f"{self.book_name} — {self.page_name}  [{self.shelf}]"


def load_catalog(root: Path = DEFAULT_DB_ROOT) -> list[CatalogEntry]:
    with open(Path(root) / "catalog-nk.yml", encoding="utf-8") as f:
        shelves = yaml.load(f, Loader=_Loader)

    entries: list[CatalogEntry] = []
    for shelf in shelves or []:
        if "SHELF" not in shelf:
            continue
        shelf_id, shelf_name = str(shelf["SHELF"]), _clean(shelf.get("name"))
        book_group = ""
        for book in shelf.get("content") or []:
            if "DIVIDER" in book:
                book_group = _clean(book["DIVIDER"])
                continue
            if "BOOK" not in book:
                continue
            book_id, book_name = str(book["BOOK"]), _clean(book.get("name"))
            page_group = ""
            for page in book.get("content") or []:
                if "DIVIDER" in page:
                    page_group = _clean(page["DIVIDER"])
                    continue
                if "PAGE" not in page or "data" not in page:
                    continue
                entries.append(
                    CatalogEntry(
                        shelf=shelf_id,
                        shelf_name=shelf_name,
                        book=book_id,
                        book_name=book_name or book_id,
                        book_group=book_group,
                        page=str(page["PAGE"]),
                        page_name=_clean(page.get("name")) or str(page["PAGE"]),
                        page_group=page_group,
                        path=str(page["data"]),
                    )
                )
    return entries


# --------------------------------------------------------------------------
# Material data
# --------------------------------------------------------------------------
def _formula(kind: int, c: np.ndarray, wl: np.ndarray) -> np.ndarray:
    """Evaluate refractiveindex.info dispersion formula *kind* at *wl* (µm).

    Follows database/doc/Dispersion formulas.pdf (and tools/nkexplorer.py).
    *c* is zero-padded to 17 coefficients, c[0] == C1.
    """
    C = np.zeros(17)
    C[: min(len(c), 17)] = c[:17]
    w2 = wl**2
    with np.errstate(all="ignore"):
        if kind == 1:  # Sellmeier
            s = 1 + C[0] + sum(C[i] / (1 - (C[i + 1] / wl) ** 2) for i in range(1, 17, 2))
            return np.sqrt(s)
        if kind == 2:  # Sellmeier-2
            s = 1 + C[0] + sum(C[i] / (1 - C[i + 1] / w2) for i in range(1, 17, 2))
            return np.sqrt(s)
        if kind == 3:  # Polynomial
            s = C[0] + sum(C[i] * wl ** C[i + 1] for i in range(1, 17, 2))
            return np.sqrt(s)
        if kind == 4:  # RefractiveIndex.INFO
            s = (
                C[0]
                + C[1] * wl ** C[2] / (w2 - C[3] ** C[4])
                + C[5] * wl ** C[6] / (w2 - C[7] ** C[8])
                + sum(C[i] * wl ** C[i + 1] for i in range(9, 17, 2))
            )
            return np.sqrt(s)
        if kind == 5:  # Cauchy
            return C[0] + sum(C[i] * wl ** C[i + 1] for i in range(1, 11, 2))
        if kind == 6:  # Gases
            return 1 + C[0] + sum(C[i] / (C[i + 1] - wl**-2.0) for i in range(1, 11, 2))
        if kind == 7:  # Herzberger
            L = 1 / (w2 - 0.028)
            return C[0] + C[1] * L + C[2] * L**2 + C[3] * w2 + C[4] * w2**2 + C[5] * w2**3
        if kind == 8:  # Retro
            t = C[0] + C[1] * w2 / (w2 - C[2]) + C[3] * w2
            return np.sqrt((2 * t + 1) / (1 - t))
        if kind == 9:  # Exotic
            return np.sqrt(C[0] + C[1] / (w2 - C[2]) + C[3] * (wl - C[4]) / ((wl - C[4]) ** 2 + C[5]))
    raise ValueError(f"Unsupported formula type {kind}")


def _decimals(token: str) -> int:
    """Number of decimal places a number was written with ("1.332" -> 3, "1.39E-8" -> 10)."""
    mant, _, exp = token.lower().partition("e")
    frac = mant.partition(".")[2]
    return len(frac) - (int(exp) if exp else 0)


def _parse_table(text: str, ncols: int) -> tuple[np.ndarray, list[float]]:
    """Rows sorted by wavelength (duplicates dropped) and the quantisation step of each column."""
    rows, decs = [], []
    for line in str(text).splitlines():
        parts = line.split()
        if len(parts) >= ncols:
            rows.append([float(p) for p in parts[:ncols]])
            decs.append([_decimals(p) for p in parts[:ncols]])
    arr = np.asarray(rows, dtype=float).reshape(-1, ncols)
    steps = [10.0 ** -float(np.median(col)) for col in np.asarray(decs).reshape(-1, ncols).T]
    arr = arr[np.argsort(arr[:, 0], kind="stable")]
    _, idx = np.unique(arr[:, 0], return_index=True)
    return arr[idx], steps


_MAX_SMOOTH_POINTS = 1500


def _interpolator(wl: np.ndarray, y: np.ndarray, smooth: bool, sigma: float = 0.0):
    """Interpolate tabulated data; values outside the table are clamped to the edges.

    With *smooth*, a quintic smoothing spline in log(λ) is used whose rms residual
    equals *sigma* (the table's rounding noise), so that the 2nd-4th derivatives
    (GDD, TOD, FOD) are not dominated by the rounding staircase of the data.
    """
    lo, hi = wl[0], wl[-1]
    if len(wl) == 1:
        return lambda x: np.full_like(np.asarray(x, dtype=float), y[0])
    if smooth and len(wl) >= 4:
        x = np.log(wl)
        if sigma < 1e-8:  # effectively exact data: interpolate (smoothing would be slow and pointless)
            spline = CubicSpline(x, y, bc_type="natural")
            return lambda v: spline(np.log(np.clip(v, lo, hi)))
        xs, ys, counts = x, y, np.ones_like(x)
        if len(x) > _MAX_SMOOTH_POINTS:
            # FITPACK cost grows steeply with N: average into narrow log-λ bins first
            edges = np.linspace(x[0], x[-1], _MAX_SMOOTH_POINTS + 1)
            b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, _MAX_SMOOTH_POINTS - 1)
            counts = np.bincount(b, minlength=_MAX_SMOOTH_POINTS)
            keep = counts > 0
            xs = (np.bincount(b, x, _MAX_SMOOTH_POINTS) / np.maximum(counts, 1))[keep]
            ys = (np.bincount(b, y, _MAX_SMOOTH_POINTS) / np.maximum(counts, 1))[keep]
            counts = counts[keep]
        try:
            with warnings.catch_warnings():  # "s too small" just means ~interpolating
                warnings.simplefilter("ignore")
                spline = UnivariateSpline(
                    xs, ys, w=np.sqrt(counts) / sigma, k=min(5, len(xs) - 1), s=len(xs)
                )
        except Exception:  # pragma: no cover - fall back to exact interpolation
            spline = CubicSpline(x, y, bc_type="natural")
        return lambda v: spline(np.log(np.clip(v, lo, hi)))
    return lambda v: np.interp(v, wl, y)


@dataclass
class Material:
    name: str
    n_kind: str  # e.g. "formula 1" or "tabulated nk"
    n_range: tuple[float, float]  # µm
    k_range: Optional[tuple[float, float]]
    n_sigma: float = 0.0  # rms rounding noise of tabulated n (0 for formulas)
    references: str = ""
    comments: str = ""
    conditions: dict = field(default_factory=dict)
    _n: Callable = field(default=None, repr=False)
    _k: Optional[Callable] = field(default=None, repr=False)

    def n(self, wl_um) -> np.ndarray:
        return self._n(np.asarray(wl_um, dtype=float))

    def k(self, wl_um) -> np.ndarray:
        wl_um = np.asarray(wl_um, dtype=float)
        return self._k(wl_um) if self._k else np.zeros_like(wl_um)

    @property
    def has_k(self) -> bool:
        return self._k is not None

    @classmethod
    def from_yaml_text(cls, text: str, name: str = "") -> "Material":
        doc = yaml.load(text, Loader=_Loader) or {}
        n_func = k_func = None
        n_kind = ""
        n_range = k_range = None
        n_sigma = 0.0
        for block in doc.get("DATA") or []:
            kind = str(block.get("type", "")).split()
            if not kind:
                continue
            if kind[0] == "formula":
                num = int(kind[1])
                coeffs = np.array(str(block["coefficients"]).split(), dtype=float)
                rng = tuple(float(v) for v in str(block["wavelength_range"]).split()[:2])
                n_func = lambda wl, num=num, coeffs=coeffs: _formula(num, coeffs, wl)
                n_kind, n_range = f"formula {num}", rng
            elif kind[0] == "tabulated":
                sub = kind[1] if len(kind) > 1 else ""
                if sub in ("nk", "n"):
                    t, steps = _parse_table(block["data"], 3 if sub == "nk" else 2)
                    n_sigma = steps[1] / np.sqrt(12)  # uniform rounding error
                    n_func = _interpolator(t[:, 0], t[:, 1], smooth=True, sigma=n_sigma)
                    n_kind, n_range = f"tabulated {sub}", (t[0, 0], t[-1, 0])
                    if sub == "nk":
                        k_func = _interpolator(t[:, 0], t[:, 2], smooth=False)
                        k_range = n_range
                elif sub == "k":
                    t, _ = _parse_table(block["data"], 2)
                    k_func = _interpolator(t[:, 0], t[:, 1], smooth=False)
                    k_range = (t[0, 0], t[-1, 0])
        if n_func is None:
            raise ValueError("This entry has no refractive-index (n) data, only k.")
        return cls(
            name=name,
            n_kind=n_kind,
            n_range=n_range,
            k_range=k_range,
            n_sigma=n_sigma,
            references=_clean(doc.get("REFERENCES", "")),
            comments=_clean(doc.get("COMMENTS", "")),
            conditions=doc.get("CONDITIONS") or {},
            _n=n_func,
            _k=k_func,
        )

    @classmethod
    def from_file(cls, path: Path, name: str = "") -> "Material":
        return cls.from_yaml_text(Path(path).read_text(encoding="utf-8"), name=name)


def load_material(entry: CatalogEntry, root: Path = DEFAULT_DB_ROOT) -> Material:
    return Material.from_file(Path(root) / "data" / entry.path, name=f"{entry.book_name} ({entry.page})")


_KIND_RE = re.compile(r"^\s*-?\s*type:\s*(formula|tabulated)", re.MULTILINE)


def data_kind(entry: CatalogEntry, root: Path = DEFAULT_DB_ROOT) -> str:
    """'formula', 'table' or '' — a cheap peek used to label entries in the UI."""
    try:
        text = (Path(root) / "data" / entry.path).read_text(encoding="utf-8")
    except OSError:
        return ""
    kinds = set(_KIND_RE.findall(text))
    return "formula" if "formula" in kinds else ("table" if kinds else "")
