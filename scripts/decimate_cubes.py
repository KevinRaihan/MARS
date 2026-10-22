"""Decimate the three stochastic-inversion SEG-Y cubes (0.1 ms) to float32 Zarr (1 ms).

Usage:
    python scripts/decimate_cubes.py [--src DIR] [--out DIR] [--factor 10] [--cubes zp zs rho]
Reads `<src>/{Zp,Zs,RHOB}_Real1_renumbering.sgy` ($MARS_SEISMIC_DIR, or --src) and writes `<out>/{zp,zs,rho}.zarr`
(default data/interim/cubes).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mars import paths  # noqa: E402
from mars.io.segy import decimate_cube  # noqa: E402

DEFAULT_SRC = Path(
    os.environ.get("MARS_SEISMIC_DIR", str(paths.RAW.parent / "seismic"))
)
CRS = {
    "crs_name": "MAXUS Djakarta / UTM zone 48S (Petrel PowerPlan:IND-48S5)",
    "crs_note": "Batavia-datum UTM 48S family; CDP X/Y from trace bytes 181/185, scalar byte 71",
}
CUBES = {
    "zp": ("Zp_Real1_renumbering.sgy", "ft/s*g/cc", "P-impedance (stochastic inversion, Real1)"),
    "zs": ("Zs_Real1_renumbering.sgy", "ft/s*g/cc", "S-impedance (stochastic inversion, Real1)"),
    "rho": ("RHOB_Real1_renumbering.sgy", "g/cc", "density (stochastic inversion, Real1)"),
}


def du(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=paths.CUBES)
    ap.add_argument("--factor", type=int, default=10, help="0.1 ms -> 1 ms is 10")
    ap.add_argument("--pad-ms", type=float, default=10.0)
    ap.add_argument("--cubes", nargs="+", choices=list(CUBES), default=list(CUBES))
    ap.add_argument("--no-sha", action="store_true", help="skip SHA-256 of the source")
    args = ap.parse_args()
    if args.src.resolve().drive.upper().startswith("E:"):
        sys.exit("refusing to read from drive E: (failing disk)")
    args.out.mkdir(parents=True, exist_ok=True)

    summary = {}
    for name in args.cubes:
        fname, units, long_name = CUBES[name]
        src, dst = args.src / fname, args.out / f"{name}.zarr"
        t0 = time.time()

        def progress(i, n, name=name, t0=t0):
            if i % 20 == 0 or i == n:
                print(f"  {name}: inline {i}/{n}  {time.time() - t0:6.0f} s", flush=True)

        print(f"{name}: {src} -> {dst}", flush=True)
        meta = decimate_cube(
            src,
            dst,
            name,
            attrs={"units": units, "long_name": long_name, **CRS},
            factor=args.factor,
            pad_ms=args.pad_ms,
            sha256=not args.no_sha,
            progress=progress,
        )
        size = du(dst)
        summary[name] = {
            "path": str(dst.relative_to(ROOT)) if dst.is_relative_to(ROOT) else str(dst),
            "bytes": size,
            "seconds": round(time.time() - t0, 1),
            **{k: meta[k] for k in ("source_sha256", "dt_ms", "scan_live_window_ms",
                                    "samples_outside_window", "affine_ilxl_to_xy")},
        }
        print(f"{name}: {size / 1e9:.2f} GB in {summary[name]['seconds']} s, "
              f"outside window {meta['samples_outside_window']}", flush=True)

    total = sum(du(p) for p in args.out.glob("*.zarr"))
    print(f"total {total / 1e9:.2f} GB in {args.out}")
    report = args.out / "decimation_report.json"
    old = json.loads(report.read_text()) if report.exists() else {}
    report.write_text(json.dumps({**old, **summary, "total_bytes": total}, indent=2))
    if total > 3e9:
        print("WARNING: over the 3 GB budget")


if __name__ == "__main__":
    main()
