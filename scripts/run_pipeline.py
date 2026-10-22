"""Run the whole revamp in order: well-log analysis, then the seismic chain, then figures.

Usage:
    python scripts/run_pipeline.py --list                 # stages, inputs and whether done
    python scripts/run_pipeline.py                        # run every stage whose output is missing
    python scripts/run_pipeline.py --only transforms spatial_blind --force
    python scripts/run_pipeline.py --from props           # from a stage onwards
    python scripts/run_pipeline.py --group wells          # well-log stages only (Colab friendly)

Locations come from mars.paths (MARS_DATA_DIR, MARS_INTERIM_DIR, MARS_RESULTS_DIR); the seismic
SEG-Y source for `decimate` from MARS_SEISMIC_DIR. A stage is skipped when its output exists,
unless --force. Stages run as subprocesses with this interpreter, stopping at the first failure.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mars import paths  # noqa: E402


@dataclass
class Stage:
    name: str
    group: str
    script: str
    output: Path
    args: list[str] = field(default_factory=list)
    note: str = ""


STAGES = [
    # ---- well logs (needs MARS_DATA_DIR only; minutes each, a few take ~30 min) ----
    Stage("parity", "wells", "run_legacy_parity.py", paths.RESULTS / "legacy_parity_thesis.json",
          note="thesis Tables 4.1-4.3, legacy mode"),
    Stage("ablation", "wells", "run_ablation.py", paths.RESULTS / "ablation" / "ablation_2well.md",
          ["--multiwell"], "Phase 2: what each legacy fix changes"),
    Stage("offsets", "wells", "diagnose_offsets.py",
          paths.RESULTS / "phase3" / "offsets" / "summary.md", note="Phase 3 step 1"),
    Stage("phase3", "wells", "run_phase3.py", paths.RESULTS / "phase3" / "brf_zone.csv",
          note="Vs-free, upscaling, BRF zone (12 wells)"),
    Stage("phase4", "wells", "run_phase4.py", paths.RESULTS / "phase4" / "summary.md",
          note="12 vs 22 wells"),
    Stage("reservoir_rule", "wells", "run_reservoir_rule.py",
          paths.RESULTS / "phase4" / "reservoir_cutoff.txt", note="two-step LBR rule"),
    Stage("transforms", "wells", "fit_final_transforms.py", paths.TRANSFORMS / "summary.md",
          note="final 22-well transforms (+ Zp-only)"),
    Stage("spatial_blind", "wells", "run_spatial_blind.py",
          paths.RESULTS / "spatial_blind" / "summary.md", note="frozen EAST / SOUTH holdouts"),
    Stage("baselines", "wells", "run_baselines.py", paths.RESULTS / "baselines" / "summary.md",
          note="MARS 1-3D vs OLS / LASSO / GBM, 22 wells"),
    Stage("closed_form", "wells", "run_closed_form_check.py",
          paths.RESULTS / "closed_form" / "summary.json",
          note="closed form vs 1-degree grid search (~1 h, timing benchmark)"),
    # ---- seismic (needs the cubes; GPU used when CuPy works) ----
    Stage("decimate", "seismic", "decimate_cubes.py", paths.CUBES / "zp.zarr",
          note="SEG-Y (MARS_SEISMIC_DIR) -> 1 ms Zarr, ~1.3 GB"),
    Stage("qc_decimation", "seismic", "qc_decimation.py",
          paths.RESULTS / "qc_decimation" / "qc_summary.json"),
    Stage("tie", "seismic", "tie_wells.py", paths.RESULTS / "tie22" / "tie_table.csv",
          ["--out", str(paths.RESULTS / "tie22")], "elastic well tie, 22 wells"),
    Stage("surfaces", "seismic", "build_surfaces.py", paths.PROPS / "surfaces.zarr"),
    Stage("props", "seismic", "apply_seismic_mars.py", paths.PROPS / "props_s25.zarr",
          ["--lateral-sigma-m", "25", "--out", str(paths.PROPS / "props_s25.zarr"),
           "--report", str(paths.RESULTS / "seismic_qc" / "sigma25")], "map version, ~1 GB"),
    Stage("props_s0", "seismic", "apply_seismic_mars.py", paths.PROPS / "props_s0.zarr",
          ["--lateral-sigma-m", "0", "--out", str(paths.PROPS / "props_s0.zarr"),
           "--report", str(paths.RESULTS / "seismic_qc" / "sigma0")], "sensitivity"),
    Stage("props_s50", "seismic", "apply_seismic_mars.py", paths.PROPS / "props.zarr",
          ["--lateral-sigma-m", "50"], "sensitivity (QC reference)"),
    Stage("qc_props", "seismic", "qc_seismic_props.py",
          paths.RESULTS / "seismic_qc" / "qc_table.csv", note="properties at the wells (not blind)"),
    Stage("figures", "seismic", "make_phase5_figures.py",
          paths.RESULTS / "figures" / "phase5" / "reservoir_property_maps.png",
          note="maps, sections, 3D (codenames)"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    names = [s.name for s in STAGES]
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", nargs="+", choices=names)
    ap.add_argument("--from", dest="start", choices=names)
    ap.add_argument("--group", choices=("wells", "seismic"))
    ap.add_argument("--force", action="store_true", help="re-run even when the output exists")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    todo = STAGES
    if a.start:
        todo = todo[names.index(a.start):]
    if a.group:
        todo = [s for s in todo if s.group == a.group]
    if a.only:
        todo = [s for s in todo if s.name in a.only]
    if a.list:
        for s in STAGES:
            done = "done" if s.output.exists() else "    "
            print(f"[{done}] {s.group:7s} {s.name:15s} {s.script:26s} {s.note}")
        return
    for s in todo:
        if s.output.exists() and not a.force:
            print(f"skip  {s.name} (output exists: {s.output.relative_to(ROOT) if s.output.is_relative_to(ROOT) else s.output})")
            continue
        cmd = [sys.executable, str(ROOT / "scripts" / s.script), *s.args]
        print(f"run   {s.name}: {' '.join(cmd[1:])}", flush=True)
        if a.dry_run:
            continue
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=ROOT, env=os.environ.copy())
        if rc:
            raise SystemExit(f"stage {s.name} failed (exit {rc})")
        print(f"done  {s.name} in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
