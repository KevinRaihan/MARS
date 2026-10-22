"""Run the legacy 2D MARS search for the six thesis cases and compare with Tabel 4.1-4.3.

Usage:
    python scripts/run_legacy_parity.py [--data data/raw] [--transforms thesis|final] [--cases VSH:LBR ...]
Writes results/legacy_parity_<transforms>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from mars import paths  # noqa: E402
from mars.attributes import (  # noqa: E402
    LEGACY_EXCLUDE_THESIS,
    TRANSFORMS_FINAL_NOTEBOOK,
    TRANSFORMS_THESIS,
    valid_columns,
)
from mars.legacy import load_legacy_zone  # noqa: E402
from mars.search import legacy_search_2d, multiple_correlation  # noqa: E402
from mars.thesis_reference import THESIS_TABLES  # noqa: E402


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=paths.RAW)
    ap.add_argument("--transforms", choices=["thesis", "final"], default="thesis")
    ap.add_argument("--cases", nargs="*", default=[f"{p}:{z}" for p, z in THESIS_TABLES])
    args = ap.parse_args()
    tset = TRANSFORMS_THESIS if args.transforms == "thesis" else TRANSFORMS_FINAL_NOTEBOOK
    excl = LEGACY_EXCLUDE_THESIS if args.transforms == "thesis" else ("IP^2", "sqrt(MU_RHO)")

    out = {}
    for case in args.cases:
        prop, zone = case.split(":")
        t0 = time.time()
        d = load_legacy_zone(args.data, zone, transforms=tset, exclude=excl)
        y = d.targets[prop]
        res = legacy_search_2d(d.X, y)
        R, _ = multiple_correlation(d.X[[res.attr1, res.attr2]].to_numpy(float), y.to_numpy(float))
        ref = THESIS_TABLES[(prop, zone)]
        checks = {
            "r": rel(res.r, ref["r"]), "theta": abs(res.theta_deg - ref["theta"]),
            "sf1": rel(res.sf1, ref["sf1"]), "sf2": rel(res.sf2, ref["sf2"]),
            "m": rel(res.m, ref["m"]), "c": rel(res.c, ref["c"]),
        }
        match = checks["theta"] == 0 and all(checks[k] < 1e-6 for k in ("r", "m", "c")) \
            and all(checks[k] < 1e-6 for k in ("sf1", "sf2"))
        out[case] = dict(
            n=len(y), n_per_well=d.n_per_well, n_attrs=d.X.shape[1],
            n_valid_attrs=len(valid_columns(d.X)), attr1=res.attr1, attr2=res.attr2,
            sf1=res.sf1, sf2=res.sf2, theta=res.theta_deg, r=res.r, m=res.m, c=res.c,
            closed_form_R=R, thesis=ref, rel_err=checks, match=bool(match),
            seconds=round(time.time() - t0, 1),
        )
        print(f"{case:9s} {'MATCH' if match else 'DIFF ':5s} r={res.r:+.6f} (thesis {ref['r']:.6f}) "
              f"theta={res.theta_deg:+.0f} ({ref['theta']:+d})  {res.attr1} | {res.attr2}  "
              f"closed-form R={R:.6f}  [{out[case]['seconds']}s]", flush=True)
    dst = paths.RESULTS / f"legacy_parity_{args.transforms}.json"
    dst.parent.mkdir(exist_ok=True)
    dst.write_text(json.dumps(out, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
    print(f"\n{sum(v['match'] for v in out.values())}/{len(out)} cases match -> {dst}")


if __name__ == "__main__":
    main()
