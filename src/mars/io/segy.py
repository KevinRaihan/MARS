"""SEG-Y inversion cubes -> decimated, analysis-ready Zarr.

The stochastic-inversion cubes are inline-sorted, regular SEG-Y files (IBM float, 0.1 ms) in
which every trace holds one contiguous *live* segment of non-zero samples; the rest is exact-zero
padding (see vault 03-Data/seismic-volumes.md). :func:`decimate_cube` reads such a file in one
sequential pass (hashing the raw bytes on the way), extends each live segment to the trace ends
before filtering so that the zero padding does not ring into the data, low-passes and resamples
with :func:`scipy.signal.resample_poly` (zero-phase polyphase FIR) and writes float32 Zarr with
dims ``(il, xl, twt_ms)``; samples outside the live segment become NaN.

Geometry: CDP X/Y are fitted with an affine ``[x, y] = A @ [il, xl, 1]`` (:func:`fit_affine`);
:func:`ilxl_to_xy` / :func:`xy_to_ilxl` apply it and its inverse.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import segyio
from scipy.signal import resample_poly

TRACE_HEADER_BYTES = 240
# 0-based offsets of the trace-header fields we use (SEG-Y rev1 byte position - 1)
HDR_SCALCO = 70  # coordinate scalar, int16
HDR_CDPX = 180
HDR_CDPY = 184
HDR_ILINE = 188
HDR_XLINE = 192


@dataclass(frozen=True)
class SegyGeometry:
    """Layout of a regular, inline-sorted post-stack SEG-Y file."""

    path: str
    ilines: tuple[int, ...]
    xlines: tuple[int, ...]
    n_samples: int
    dt_ms: float
    t0_ms: float
    sample_format: int
    data_offset: int  # byte offset of the first trace header

    @property
    def trace_bytes(self) -> int:
        return TRACE_HEADER_BYTES + 4 * self.n_samples

    @property
    def times_ms(self) -> np.ndarray:
        return self.t0_ms + self.dt_ms * np.arange(self.n_samples)


def read_geometry(path: str | Path) -> SegyGeometry:
    """Read and check the geometry of an inline-sorted, regular SEG-Y file with segyio."""
    with segyio.open(str(path), ignore_geometry=False) as f:
        if f.sorting != segyio.TraceSortingFormat.INLINE_SORTING:
            raise ValueError(f"{path}: expected inline sorting, got {f.sorting}")
        fmt = int(f.bin[segyio.BinField.Format])
        if fmt not in (1, 5):
            raise ValueError(f"{path}: unsupported sample format {fmt}")
        geom = SegyGeometry(
            path=str(path),
            ilines=tuple(int(i) for i in f.ilines),
            xlines=tuple(int(x) for x in f.xlines),
            n_samples=len(f.samples),
            dt_ms=segyio.tools.dt(f) / 1000.0,
            t0_ms=float(f.samples[0]),
            sample_format=fmt,
            data_offset=3600 + 3200 * f.ext_headers,
        )
    size = Path(path).stat().st_size
    expected = geom.data_offset + len(geom.ilines) * len(geom.xlines) * geom.trace_bytes
    if size != expected:
        raise ValueError(f"{path}: size {size} != {expected} expected for a full regular grid")
    return geom


# ---------------------------------------------------------------------------- geometry


def fit_affine(il, xl, x, y) -> dict:
    """Least-squares affine ``[x, y] = A @ [il, xl, 1]`` from trace headers.

    Returns ``{"A": 2x3 list, "A_inv": 2x3 list, "rms_m": float, "max_abs_m": float}`` where
    ``A_inv`` maps ``[x, y, 1]`` back to ``[il, xl]``.
    """
    il, xl, x, y = (np.asarray(v, dtype=float).ravel() for v in (il, xl, x, y))
    g = np.column_stack([il, xl, np.ones_like(il)])
    coef, *_ = np.linalg.lstsq(g, np.column_stack([x, y]), rcond=None)
    a = coef.T  # 2x3
    resid = np.column_stack([x, y]) - g @ coef
    full = np.vstack([a, [0.0, 0.0, 1.0]])
    a_inv = np.linalg.inv(full)[:2]
    return {
        "A": a.tolist(),
        "A_inv": a_inv.tolist(),
        "rms_m": float(np.sqrt(np.mean(np.sum(resid**2, axis=1)))),
        "max_abs_m": float(np.abs(resid).max()),
    }


def ilxl_to_xy(affine: dict, il, xl) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(affine["A"])
    il, xl = np.asarray(il, dtype=float), np.asarray(xl, dtype=float)
    return a[0, 0] * il + a[0, 1] * xl + a[0, 2], a[1, 0] * il + a[1, 1] * xl + a[1, 2]


def xy_to_ilxl(affine: dict, x, y) -> tuple[np.ndarray, np.ndarray]:
    """Fractional inline/crossline of map coordinates (round to snap to the grid)."""
    b = np.asarray(affine["A_inv"])
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return b[0, 0] * x + b[0, 1] * y + b[0, 2], b[1, 0] * x + b[1, 1] * y + b[1, 2]


def apply_coordinate_scalar(values, scalar) -> np.ndarray:
    """SEG-Y coordinate scalar: positive multiplies, negative divides, 0 means 1."""
    v = np.asarray(values, dtype=float)
    s = np.asarray(scalar, dtype=float)
    factor = np.where(s > 0, s, np.where(s < 0, -1.0 / np.where(s == 0, 1, s), 1.0))
    return v * factor


# ---------------------------------------------------------------------------- resampling


def live_extent(traces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and last non-zero (and finite) sample index per trace; -1 for dead traces."""
    live = (traces != 0) & np.isfinite(traces)
    has = live.any(axis=1)
    n = traces.shape[1]
    first = np.where(has, live.argmax(axis=1), -1)
    last = np.where(has, n - 1 - live[:, ::-1].argmax(axis=1), -1)
    return first, last


def decimate_traces(
    traces: np.ndarray, factor: int, first: np.ndarray | None = None, last: np.ndarray | None = None
) -> np.ndarray:
    """Anti-alias and downsample traces ``(n_traces, n_samples)`` by an integer ``factor``.

    Each trace's live segment ``[first, last]`` is extended with its edge values before
    filtering (no ringing from the zero padding); output sample ``k`` (input sample
    ``k * factor``) is NaN unless ``first <= k * factor <= last``. Output is float32 with
    ``ceil(n_samples / factor)`` samples.
    """
    traces = np.atleast_2d(np.asarray(traces, dtype=np.float32))
    if first is None or last is None:
        first, last = live_extent(traces)
    n = traces.shape[1]
    n_out = math.ceil(n / factor)
    out = np.full((traces.shape[0], n_out), np.nan, dtype=np.float32)
    alive = first >= 0
    if not alive.any():
        return out
    lo, hi = first[alive, None], last[alive, None]
    idx = np.clip(np.arange(n)[None, :], lo, hi)
    filled = np.take_along_axis(traces[alive], idx, axis=1)
    res = resample_poly(filled, 1, factor, axis=1, padtype="line").astype(np.float32)
    k = np.arange(n_out)[None, :] * factor
    out[alive] = np.where((k >= lo) & (k <= hi), res[:, :n_out], np.nan)
    return out


# ---------------------------------------------------------------------------- reading


def _block_headers(raw: np.ndarray) -> dict[str, np.ndarray]:
    h = raw[:, :TRACE_HEADER_BYTES]

    def i32(off):
        return np.ascontiguousarray(h[:, off : off + 4]).view(">i4").ravel().astype(np.int64)

    scal = np.ascontiguousarray(h[:, HDR_SCALCO : HDR_SCALCO + 2]).view(">i2").ravel()
    return {
        "il": i32(HDR_ILINE),
        "xl": i32(HDR_XLINE),
        "x": apply_coordinate_scalar(i32(HDR_CDPX), scal),
        "y": apply_coordinate_scalar(i32(HDR_CDPY), scal),
    }


def _decode(raw: np.ndarray, fmt: int) -> np.ndarray:
    data = np.ascontiguousarray(raw[:, TRACE_HEADER_BYTES:])
    if fmt == 5:
        return data.view(">f4").astype(np.float32)
    return segyio.tools.native(data, format=fmt, copy=False)


def iter_inlines(
    geom: SegyGeometry, hasher=None, ilines: list[int] | None = None
) -> Iterator[tuple[int, dict[str, np.ndarray], np.ndarray]]:
    """Yield ``(inline, headers, traces)`` per inline by raw sequential reads.

    With ``hasher`` (a hashlib object) every byte of the file is fed to it in order, which
    requires reading all inlines (``ilines=None``). Header il/xl are checked against the grid.
    """
    nx = len(geom.xlines)
    block = nx * geom.trace_bytes
    want = set(geom.ilines if ilines is None else ilines)
    if hasher is not None and ilines is not None:
        raise ValueError("hashing needs a full pass (ilines=None)")
    xl_expected = np.asarray(geom.xlines)
    with open(geom.path, "rb", buffering=0) as fh:
        head = fh.read(geom.data_offset)
        if hasher is not None:
            hasher.update(head)
        for i, il in enumerate(geom.ilines):
            if il not in want:
                continue
            fh.seek(geom.data_offset + i * block)
            buf = fh.read(block)
            if len(buf) != block:
                raise OSError(f"short read at inline {il}")
            if hasher is not None:
                hasher.update(buf)
            raw = np.frombuffer(buf, dtype=np.uint8).reshape(nx, geom.trace_bytes)
            hdr = _block_headers(raw)
            if (hdr["il"] != il).any() or (hdr["xl"] != xl_expected).any():
                raise ValueError(f"inline {il}: trace headers do not match the regular grid")
            yield il, hdr, _decode(raw, geom.sample_format)


def scan_live_window(geom: SegyGeometry, stride: int = 8) -> tuple[int, int]:
    """Global first/last live sample index over every ``stride``-th inline (plus the last)."""
    ils = list(geom.ilines[::stride])
    if ils[-1] != geom.ilines[-1]:
        ils.append(geom.ilines[-1])
    lo, hi = geom.n_samples, -1
    for _, _, tr in iter_inlines(geom, ilines=ils):
        first, last = live_extent(tr)
        alive = first >= 0
        if alive.any():
            lo, hi = min(lo, int(first[alive].min())), max(hi, int(last[alive].max()))
    if hi < 0:
        raise ValueError(f"{geom.path}: no live samples in scanned inlines")
    return lo, hi


# ---------------------------------------------------------------------------- writing


def decimate_cube(
    src: str | Path,
    dst: str | Path,
    var: str,
    attrs: dict | None = None,
    factor: int = 10,
    pad_ms: float = 10.0,
    scan_stride: int = 8,
    chunks: tuple[int, int, int] = (16, 256, 128),
    sha256: bool = True,
    progress=None,
) -> dict:
    """Decimate one SEG-Y cube into a Zarr store holding ``var(il, xl, twt_ms)``.

    The output time axis is the scanned global live window (padded by ``pad_ms``, snapped to
    the output grid). Live samples found outside it in the full pass are counted in the
    ``samples_outside_window`` attribute (should be 0). Also stored: ``cdp_x``/``cdp_y`` and
    the per-trace live top/base (``live_top_ms``/``live_base_ms``, NaN for dead traces) as
    ``(il, xl)`` coordinates, the affine fit and the SHA-256 of the source file.
    Returns the attributes written.
    """
    import zarr
    from zarr.codecs import BloscCodec

    geom = read_geometry(src)
    nil, nxl = len(geom.ilines), len(geom.xlines)
    dt_out = geom.dt_ms * factor
    n_out = math.ceil(geom.n_samples / factor)

    lo, hi = scan_live_window(geom, scan_stride)
    k0 = max(0, int(math.floor((lo * geom.dt_ms - pad_ms) / dt_out)))
    k1 = min(n_out - 1, int(math.ceil((hi * geom.dt_ms + pad_ms) / dt_out)))
    twt = geom.t0_ms + dt_out * np.arange(k0, k1 + 1)
    nt = len(twt)

    root = zarr.open_group(str(dst), mode="w", zarr_format=3)
    comp = (BloscCodec(cname="zstd", clevel=5, shuffle="shuffle"),)

    def coord(name, values, dims, dtype):
        a = root.create_array(
            name, shape=values.shape, dtype=dtype, dimension_names=dims, compressors=comp
        )
        a[...] = values
        return a

    coord("il", np.asarray(geom.ilines, dtype=np.int32), ["il"], "int32")
    coord("xl", np.asarray(geom.xlines, dtype=np.int32), ["xl"], "int32")
    coord("twt_ms", twt.astype(np.float64), ["twt_ms"], "float64").attrs.update(
        {"units": "ms", "long_name": "two-way time"}
    )
    cube = root.create_array(
        var,
        shape=(nil, nxl, nt),
        chunks=chunks,
        dtype="float32",
        fill_value=np.nan,
        dimension_names=["il", "xl", "twt_ms"],
        compressors=comp,
    )
    plane_names = ("cdp_x", "cdp_y", "live_top_ms", "live_base_ms")
    plane = {k: np.full((nil, nxl), np.nan) for k in plane_names}
    il_all = np.empty((nil, nxl), dtype=np.int64)
    xl_all = np.empty((nil, nxl), dtype=np.int64)

    hasher = hashlib.sha256() if sha256 else None
    buf = np.full((chunks[0], nxl, nt), np.nan, dtype=np.float32)
    outside = 0
    vmin, vmax = np.inf, -np.inf
    for i, (_il, hdr, tr) in enumerate(iter_inlines(geom, hasher=hasher)):
        first, last = live_extent(tr)
        alive = first >= 0
        if alive.any():
            outside += int(np.clip(k0 * factor - first[alive], 0, None).sum())
            outside += int(np.clip(last[alive] - k1 * factor, 0, None).sum())
            live_vals = tr[alive][(tr[alive] != 0)]
            vmin, vmax = min(vmin, float(live_vals.min())), max(vmax, float(live_vals.max()))
        dec = decimate_traces(tr, factor, first, last)
        buf[i % chunks[0]] = dec[:, k0 : k1 + 1]
        il_all[i], xl_all[i] = hdr["il"], hdr["xl"]
        plane["cdp_x"][i], plane["cdp_y"][i] = hdr["x"], hdr["y"]
        plane["live_top_ms"][i] = np.where(alive, geom.t0_ms + first * geom.dt_ms, np.nan)
        plane["live_base_ms"][i] = np.where(alive, geom.t0_ms + last * geom.dt_ms, np.nan)
        if (i + 1) % chunks[0] == 0 or i == nil - 1:
            j0 = i - i % chunks[0]
            cube[j0 : i + 1] = buf[: i + 1 - j0]
            buf[:] = np.nan
        if progress is not None:
            progress(i + 1, nil)

    for name, values in plane.items():
        coord(name, values, ["il", "xl"], "float64")
    root["cdp_x"].attrs["units"] = root["cdp_y"].attrs["units"] = "m"
    root["live_top_ms"].attrs["units"] = root["live_base_ms"].attrs["units"] = "ms"

    affine = fit_affine(il_all, xl_all, plane["cdp_x"], plane["cdp_y"])
    meta = {
        **(attrs or {}),
        "source_file": Path(src).name,
        "source_sha256": hasher.hexdigest() if hasher is not None else None,
        "source_geometry": {k: v for k, v in asdict(geom).items() if k not in ("ilines", "xlines")},
        "source_dt_ms": geom.dt_ms,
        "dt_ms": dt_out,
        "decimation": (
            f"live segment edge-extended, scipy.signal.resample_poly(up=1, down={factor}, "
            "window=('kaiser', 5.0), padtype='line'); zero-phase; NaN outside live segment"
        ),
        "scan_live_window_ms": [geom.t0_ms + lo * geom.dt_ms, geom.t0_ms + hi * geom.dt_ms],
        "pad_ms": pad_ms,
        "samples_outside_window": outside,
        "source_live_min": vmin,
        "source_live_max": vmax,
        "affine_ilxl_to_xy": affine,
    }
    cube.attrs.update({k: v for k, v in meta.items() if k in ("units", "long_name")})
    cube.attrs["coordinates"] = " ".join(plane)
    root.attrs.update(json.loads(json.dumps(meta)))
    zarr.consolidate_metadata(str(dst))
    return meta
