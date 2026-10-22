"""QC of the decimated cubes against the source SEG-Y.

Usage:
    python scripts/qc_decimation.py [--src DIR] [--cubes-dir data/interim/cubes] [--out results/qc_decimation]
For each cube: overlays of a few traces (0.1 ms source vs 1 ms Zarr), their amplitude spectra,
one inline section, and p1/p50/p99 of the source non-zero samples vs the Zarr non-NaN samples
on the same random traces. Writes PNGs and qc_summary.json (no raw values beyond percentiles).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

from mars import paths  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from mars.io.segy import iter_inlines, live_extent, read_geometry  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from decimate_cubes import CUBES, DEFAULT_SRC  # noqa: E402

SRC_COLOR, DEC_COLOR = "#2a78d6", "#eb6834"  # categorical slots 1-2 (dataviz default palette)
EXPECTED = {  # sampled non-zero p1/p50/p99 from the 2026-10-20 probe (vault seismic-volumes.md)
    "zp": (10170, 23950, 48660),
    "zs": (3541, 12410, 31180),
    "rho": (1.91, 2.32, 2.91),
}
QC_INLINES = (400, 500, 600)


def spectrum(x: np.ndarray, dt_ms: float) -> tuple[np.ndarray, np.ndarray]:
    x = x - x.mean()
    x = x * np.hanning(len(x))
    amp = np.abs(np.fft.rfft(x)) * 2 / len(x)
    return np.fft.rfftfreq(len(x), dt_ms / 1000.0), amp


def qc_cube(name: str, src: Path, store: Path, out: Path, rng: np.random.Generator) -> dict:
    geom = read_geometry(src)
    ds = xr.open_zarr(store)
    cube = ds[name]
    t_src = geom.times_ms
    t_dec = ds["twt_ms"].values
    units = cube.attrs.get("units", "")

    src_vals, dec_vals, traces = [], [], []
    for il, _, tr in iter_inlines(geom, ilines=list(QC_INLINES)):
        dec = cube.sel(il=il).values  # (xl, twt)
        first, last = live_extent(tr)
        alive = np.flatnonzero(first >= 0)
        pick = rng.choice(alive, size=min(150, len(alive)), replace=False)
        src_vals.append(tr[pick][tr[pick] != 0])
        dec_vals.append(dec[pick][np.isfinite(dec[pick])])
        longest = alive[np.argmax((last - first)[alive])]
        traces.append((il, int(geom.xlines[longest]), tr[longest], dec[longest], first[longest], last[longest]))
        if il == QC_INLINES[1]:
            section = dec

    q = [1, 50, 99]
    p_src = np.percentile(np.concatenate(src_vals), q)
    p_dec = np.percentile(np.concatenate(dec_vals), q)
    rel = np.abs(p_dec - p_src) / np.abs(p_src)

    # traces + spectra
    fig, axes = plt.subplots(2, len(traces), figsize=(5 * len(traces), 8.5), constrained_layout=True)
    for j, (il, xl, tr, dec, a, b) in enumerate(traces):
        ax = axes[0, j]
        ax.plot(tr[a : b + 1], t_src[a : b + 1], color=SRC_COLOR, lw=1.0, label="source 0.1 ms")
        ax.plot(dec, t_dec, color=DEC_COLOR, lw=1.5, ls="--", label="Zarr 1 ms")
        ax.invert_yaxis()
        ax.set(title=f"IL {il} / XL {xl}", xlabel=f"{name} [{units}]", ylabel="TWT [ms]")
        ax.grid(alpha=0.25, lw=0.5)
        f_s, a_s = spectrum(tr[a : b + 1].astype(float), geom.dt_ms)
        live = np.isfinite(dec)
        f_d, a_d = spectrum(dec[live].astype(float), t_dec[1] - t_dec[0])
        ax = axes[1, j]
        ax.semilogy(f_s, a_s, color=SRC_COLOR, lw=1.0, label="source 0.1 ms")
        ax.semilogy(f_d, a_d, color=DEC_COLOR, lw=1.5, ls="--", label="Zarr 1 ms")
        ax.axvline(500, color="0.4", lw=0.8, ls=":")
        ax.text(505, a_s.max(), "1 ms Nyquist", fontsize=8, color="0.3", va="top")
        ax.set(xlim=(0, 700), xlabel="frequency [Hz]", ylabel="amplitude")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0, 0].legend(loc="lower right", fontsize=8)
    axes[1, 0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"{name}: source vs decimated traces (top) and amplitude spectra (bottom)")
    fig.savefig(out / f"{name}_traces_spectra.png", dpi=110)
    plt.close(fig)

    # one inline section of the decimated cube
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    xl = ds["xl"].values
    lo, hi = np.nanpercentile(section, [1, 99])
    im = ax.imshow(section.T, aspect="auto", cmap="viridis", vmin=lo, vmax=hi,
                   extent=(xl[0], xl[-1], t_dec[-1], t_dec[0]), interpolation="nearest")
    fig.colorbar(im, ax=ax, label=f"{name} [{units}]")
    ax.set(title=f"{name}: inline {QC_INLINES[1]} (1 ms Zarr; white = NaN padding)",
           xlabel="crossline", ylabel="TWT [ms]")
    fig.savefig(out / f"{name}_inline{QC_INLINES[1]}.png", dpi=110)
    plt.close(fig)

    # high-frequency energy removed by the decimation (fraction of spectral power above 500 Hz)
    hf = []
    for _, _, tr, _, a, b in traces:
        f_s, a_s = spectrum(tr[a : b + 1].astype(float), geom.dt_ms)
        hf.append(float((a_s[f_s > 500] ** 2).sum() / (a_s**2).sum()))

    return {
        "units": units,
        "p1_p50_p99_source": p_src.round(4).tolist(),
        "p1_p50_p99_zarr": p_dec.round(4).tolist(),
        "p1_p50_p99_expected_probe": list(EXPECTED[name]),
        "max_rel_diff_source_vs_zarr": float(rel.max().round(5)),
        "power_fraction_above_500Hz": [round(h, 6) for h in hf],
        "attrs": {k: ds.attrs[k] for k in ("samples_outside_window", "scan_live_window_ms", "dt_ms")},
        "affine_rms_m": ds.attrs["affine_ilxl_to_xy"]["rms_m"],
        "nan_fraction_inline500": float(np.isnan(section).mean().round(4)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--cubes-dir", type=Path, default=paths.CUBES)
    ap.add_argument("--out", type=Path, default=paths.RESULTS / "qc_decimation")
    ap.add_argument("--cubes", nargs="+", choices=list(CUBES), default=list(CUBES))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    summary = {}
    for name in args.cubes:
        summary[name] = qc_cube(name, args.src / CUBES[name][0], args.cubes_dir / f"{name}.zarr", args.out, rng)
        print(name, json.dumps(summary[name]))
    (args.out / "qc_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
