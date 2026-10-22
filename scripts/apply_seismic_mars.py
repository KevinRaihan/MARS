"""Apply the final Vs-free MARS transforms to the inversion cubes, zone by zone.

Usage:
    python scripts/apply_seismic_mars.py [--backend auto|numpy|cupy] [--chunk 32]
        [--lateral-sigma-m 50] [--benchmark-only]
Inputs: data/interim/cubes/{zp,rho}.zarr, data/interim/props/surfaces.zarr
(scripts/build_surfaces.py), results/transforms/*.json. Output: data/interim/props/props.zarr
with phie/vsh/swt and the *_zponly controls (float32, NaN outside UBRn/LBRc) and an int8 zone
label (1 UBRn, 2 BRF, 3 LBRc, 0 outside), cropped to Top UBR - 20 ms ... Talang Akar + 20 ms.
LBRc PHIE is two-step: NaN where the predicted LBRc VSH > 0.49 (non-reservoir).
A GPU-vs-CPU benchmark on one chunk is written to results/seismic_qc/benchmark.json; the faster
backend processes the volume (CPU if CuPy or the GPU fails).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mars import dataset, paths  # noqa: E402

warnings.filterwarnings("ignore", message="CUDA path could not be detected")

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

from mars.conditioning import load_well_zone  # noqa: E402
from mars.horizons import zone_labels  # noqa: E402
from mars.seismic_apply import (  # noqa: E402
    ZP_VALID,
    SmoothingPlan,
    ZoneRecipe,
    count_nonphysical,
    cpu_backend,
    get_backend,
    lateral_sigma_traces,
    process_chunk,
    vertical_sigma_ms,
)
from mars.transform import MarsTransform  # noqa: E402

TARGETS = ("PHIE", "VSH", "SWT")
ZONE_LABELS = {"UBRn": 1, "BRF": 2, "LBRc": 3}
BOUNDARIES = ("top_UBR", "top_BRF", "top_LBRc", "top_TAF")
PAD_MS = 20.0


def load_recipes(tdir: Path) -> tuple[list[ZoneRecipe], dict]:
    recipes, upscale = [], {}
    for zone in ("UBRn", "LBRc"):
        tr, rule = {}, {}
        for t in TARGETS:
            for suffix in ("", "_zponly"):
                m = MarsTransform.from_json(tdir / f"{zone}_{t}{suffix}.json")
                name = t.lower() + suffix
                tr[name] = m
                upscale[zone] = m.meta["upscale_ft"]
                cut = m.meta.get("apply_where_vsh_le")
                if cut is not None:
                    rule[name] = ("vsh" + suffix, float(cut))
        recipes.append(ZoneRecipe(zone, ZONE_LABELS[zone], tr, rule))
    return recipes, upscale


def zone_velocity(data: Path, zone: str) -> float:
    """Median log Vp (ft/s) of a zone over all wells that have it."""
    vals = []
    for f in sorted((data / "wells").glob("*.las")):
        w = dataset.well_id(f.stem)
        try:
            vals.append(load_well_zone(data, w, zone, require_vs=False)["VP"].to_numpy())
        except (ValueError, KeyError):
            continue
    return float(np.median(np.concatenate(vals)))


def create_store(path: Path, il, xl, twt, surfaces, names, attrs):
    import zarr
    from zarr.codecs import BloscCodec

    root = zarr.open_group(str(path), mode="w", zarr_format=3)
    comp = (BloscCodec(cname="zstd", clevel=5, shuffle="shuffle"),)

    def put(name, values, dims, dtype, **kw):
        a = root.create_array(name, shape=values.shape, dtype=dtype, dimension_names=dims,
                              compressors=comp, **kw)
        a[...] = values
        return a

    put("il", np.asarray(il, np.int32), ["il"], "int32")
    put("xl", np.asarray(xl, np.int32), ["xl"], "int32")
    put("twt_ms", np.asarray(twt, np.float64), ["twt_ms"], "float64").attrs["units"] = "ms"
    for b in BOUNDARIES:
        put(b, surfaces[b].astype(np.float32), ["il", "xl"], "float32").attrs["units"] = "ms"
    shape, chunks = (len(il), len(xl), len(twt)), (32, 256, 128)
    for n in names:
        root.create_array(n, shape=shape, chunks=chunks, dtype="float32", fill_value=np.nan,
                          dimension_names=["il", "xl", "twt_ms"], compressors=comp)
        root[n].attrs.update({"units": "v/v", "coordinates": " ".join(BOUNDARIES)})
    root.create_array("zone", shape=shape, chunks=chunks, dtype="int8", fill_value=0,
                      dimension_names=["il", "xl", "twt_ms"], compressors=comp)
    root["zone"].attrs.update({"flag_values": [0, 1, 2, 3],
                               "flag_meanings": "outside UBRn BRF LBRc"})
    root.attrs.update(attrs)
    return root


class ChunkReader:
    """Reads (zp, rho) inline blocks with halo and a time window from the decimated cubes."""

    def __init__(self, cubes: Path, t_lo: float, t_hi: float, halo_il: int, halo_ms: float):
        self.zp = xr.open_zarr(cubes / "zp.zarr")["zp"]
        self.rho = xr.open_zarr(cubes / "rho.zarr")["rho"]
        tw = self.zp["twt_ms"].values
        self.dt = float(tw[1] - tw[0])
        self.k0 = int(np.searchsorted(tw, t_lo))
        self.k1 = int(np.searchsorted(tw, t_hi, side="right"))
        h = int(math.ceil(halo_ms / self.dt))
        self.r0, self.r1 = max(0, self.k0 - h), min(len(tw), self.k1 + h)
        self.twt_out = tw[self.k0:self.k1]
        self.t_core = slice(self.k0 - self.r0, self.k1 - self.r0)
        self.halo_il, self.n_il = halo_il, self.zp.sizes["il"]

    def read(self, i0: int, i1: int):
        a, b = max(0, i0 - self.halo_il), min(self.n_il, i1 + self.halo_il)
        sl = {"il": slice(a, b), "twt_ms": slice(self.r0, self.r1)}
        zp = self.zp.isel(sl).values.astype(np.float32)
        rho = self.rho.isel(sl).values.astype(np.float32)
        return zp, rho, slice(i0 - a, i0 - a + (i1 - i0))


def run_chunk(be, reader, surfaces, recipes, plan, names, i0, i1):
    t = time.perf_counter()
    zp, rho, core = reader.read(i0, i1)
    t_read = time.perf_counter() - t
    labels = zone_labels(reader.twt_out, [surfaces[b][i0:i1] for b in BOUNDARIES])
    t = time.perf_counter()
    out = process_chunk(be, zp, rho, labels, recipes, plan, names, core, reader.t_core)
    be.sync()
    t_proc = time.perf_counter() - t
    counts = count_nonphysical(zp[core, :, reader.t_core], rho[core, :, reader.t_core], labels)
    out["_counts"] = counts
    return out, labels, t_read, t_proc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path,
                    default=paths.RAW)
    ap.add_argument("--cubes", type=Path, default=paths.CUBES)
    ap.add_argument("--surfaces", type=Path,
                    default=paths.PROPS / "surfaces.zarr")
    ap.add_argument("--transforms", type=Path, default=paths.TRANSFORMS)
    ap.add_argument("--out", type=Path, default=paths.PROPS / "props.zarr")
    ap.add_argument("--report", type=Path, default=paths.RESULTS / "seismic_qc")
    ap.add_argument("--backend", choices=("auto", "numpy", "cupy"), default="auto")
    ap.add_argument("--chunk", type=int, default=24, help="core inlines per chunk")
    ap.add_argument("--lateral-sigma-m", type=float, default=50.0)
    ap.add_argument("--benchmark-only", action="store_true")
    args = ap.parse_args()
    args.report.mkdir(parents=True, exist_ok=True)

    recipes, upscale = load_recipes(args.transforms)
    vel = {z: zone_velocity(args.data, z) for z in upscale}
    vsig = {z: vertical_sigma_ms(upscale[z], vel[z]) for z in upscale}
    plan = SmoothingPlan(args.lateral_sigma_m, vsig)
    print("zone Vp", {z: round(v) for z, v in vel.items()}, "vertical sigma ms",
          {z: round(v, 2) for z, v in vsig.items()}, "lateral sigma traces",
          lateral_sigma_traces(args.lateral_sigma_m), "halo inlines", plan.halo_inlines(),
          flush=True)

    sf = xr.open_zarr(args.surfaces)
    surfaces = {b: sf[b].values for b in BOUNDARIES}
    t_lo = math.floor(np.nanmin(surfaces["top_UBR"]) - PAD_MS)
    t_hi = math.ceil(np.nanmax(surfaces["top_TAF"]) + PAD_MS)
    reader = ChunkReader(args.cubes, t_lo, t_hi, plan.halo_inlines(), plan.halo_ms())
    names = [n for rc in recipes for n in rc.transforms]
    names = list(dict.fromkeys(names))
    n_il = reader.n_il

    # ---- benchmark on one mid-volume chunk
    i0 = (n_il // 2 // args.chunk) * args.chunk
    i1 = min(n_il, i0 + args.chunk)
    bench = {"chunk_inlines": [int(i0), int(i1)], "halo_inlines": plan.halo_inlines(),
             "samples_per_chunk_read": None}
    cpu = cpu_backend()
    out_cpu, _, t_read, t_cpu = run_chunk(cpu, reader, surfaces, recipes, plan, names, i0, i1)
    bench.update({"read_s": round(t_read, 2), "cpu_s": round(t_cpu, 2)})
    gpu = None
    if args.backend != "numpy":
        try:
            gpu = get_backend("cupy")
        except Exception as e:  # noqa: BLE001 - no CuPy / no GPU: CPU only
            if args.backend == "cupy":
                raise
            bench["gpu_error"] = f"{type(e).__name__}: {e}"
    if gpu is not None:
        try:
            _, _, _, t_cold = run_chunk(gpu, reader, surfaces, recipes, plan, names, i0, i1)
            out_gpu, _, _, t_gpu = run_chunk(gpu, reader, surfaces, recipes, plan, names, i0, i1)
            diff = max(float(np.nanmax(np.abs(out_gpu[n] - out_cpu[n]))) for n in names
                       if np.isfinite(out_cpu[n]).any())
            nan_diff = sum(int((np.isnan(out_gpu[n]) != np.isnan(out_cpu[n])).sum()) for n in names)
            n_valid = sum(int(np.isfinite(out_cpu[n]).sum()) for n in names)
            bench.update({"gpu_cold_s": round(t_cold, 2), "gpu_s": round(t_gpu, 2),
                          "gpu_vs_cpu_max_abs_diff": diff, "gpu_vs_cpu_nan_pattern_diff": nan_diff,
                          "cpu_valid_samples": n_valid})
        except Exception as e:  # noqa: BLE001 - e.g. out of GPU memory: fall back to CPU
            gpu, bench["gpu_error"] = None, f"{type(e).__name__}: {e}"
    use_gpu = gpu is not None and (args.backend == "cupy" or bench.get("gpu_s", 1e9) < t_cpu)
    be = gpu if use_gpu else cpu
    bench["chosen_backend"] = be.name
    zr = reader.zp.isel(il=slice(0, 1), xl=slice(0, 1), twt_ms=slice(reader.r0, reader.r1))
    bench["samples_per_chunk_read"] = int((i1 - i0 + 2 * plan.halo_inlines())
                                          * reader.zp.sizes["xl"] * zr.sizes["twt_ms"])
    print("benchmark", json.dumps(bench), flush=True)
    (args.report / "benchmark.json").write_text(json.dumps(bench, indent=2))
    if args.benchmark_only:
        return

    attrs = {
        "zones": json.dumps(ZONE_LABELS), "boundaries": json.dumps(BOUNDARIES),
        "window_ms": [float(reader.twt_out[0]), float(reader.twt_out[-1])],
        "rho_clip": [1.8, 3.0], "zp_valid_masked_outside": list(ZP_VALID),
        "lateral_sigma_m": args.lateral_sigma_m,
        "lateral_sigma_traces_il_xl": list(lateral_sigma_traces(args.lateral_sigma_m)),
        "vertical_sigma_ms": vsig, "zone_vp_ft_s": vel, "upscale_ft": upscale,
        "vertical_sigma_rule": "sigma = (2 L / Vp) / sqrt(12): Gaussian with the variance of the "
                               "L-ft log-upscaling boxcar at the zone's median log Vp",
        "smoothing": "NaN-aware normalised convolution, separable Gaussian (truncate 4 sigma)",
        "clip_predictions": [0.0, 1.0],
        "lbrc_phie_rule": "two-step: PHIE is NaN where predicted LBRc VSH > 0.49 (non-reservoir); "
                          "the *_zponly chain uses the Zp-only VSH",
        "brf": "zone label only (no property transforms)",
        "transforms": {rc.zone: {n: f"{rc.zone}_{n.split('_')[0].upper()}"
                                    f"{'_zponly' if n.endswith('_zponly') else ''}.json"
                                 for n in rc.transforms} for rc in recipes},
        "backend": be.name, "source_cubes": ["zp.zarr", "rho.zarr"],
    }
    root = create_store(args.out, reader.zp["il"].values, reader.zp["xl"].values,
                        reader.twt_out, surfaces, names, attrs)
    t_all, t_rd, t_pr = time.perf_counter(), 0.0, 0.0
    tally = {}
    for i0 in range(0, n_il, args.chunk):
        i1 = min(n_il, i0 + args.chunk)
        try:
            out, labels, tr, tp = run_chunk(be, reader, surfaces, recipes, plan, names, i0, i1)
        except Exception as e:  # noqa: BLE001 - GPU failure mid-run: redo this chunk on CPU
            if be.name == "numpy":
                raise
            print(f"GPU failed at inline chunk {i0} ({e}); continuing on CPU", flush=True)
            be = cpu
            root.attrs["backend"] = "numpy (GPU failed mid-run)"
            out, labels, tr, tp = run_chunk(be, reader, surfaces, recipes, plan, names, i0, i1)
        for k, c in out.pop("_counts").items():
            zname = {v: z for z, v in ZONE_LABELS.items()}.get(k, "outside")
            agg = tally.setdefault(zname, {"live": 0, "zp_masked": 0, "rho_clipped": 0})
            for key in agg:
                agg[key] += c[key]
        for n in names:
            root[n][i0:i1] = out[n]
        root["zone"][i0:i1] = labels
        t_rd, t_pr = t_rd + tr, t_pr + tp
        print(f"  inlines {i0}-{i1}: read {tr:.1f} s, process {tp:.1f} s", flush=True)
    import zarr

    root.attrs["nonphysical_samples_per_zone"] = tally
    zarr.consolidate_metadata(str(args.out))
    (args.report / "nonphysical_per_zone.json").write_text(json.dumps(tally, indent=2))
    print("non-physical per zone", json.dumps(tally))
    size = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    timing = {"total_s": round(time.perf_counter() - t_all, 1), "read_s": round(t_rd, 1),
              "process_s": round(t_pr, 1), "backend": be.name, "bytes": size}
    bench["full_run"] = timing
    (args.report / "benchmark.json").write_text(json.dumps(bench, indent=2))
    print("done", json.dumps(timing))


if __name__ == "__main__":
    main()
