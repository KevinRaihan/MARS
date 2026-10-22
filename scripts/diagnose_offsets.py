"""Phase 3, step 1: why do wells disagree (UBR pooled blind r ~ 0, per-well r 0.3-0.8)?

For each zone and target this script quantifies:
  * between-well level offsets: per-well mean residual of a pooled linear model (target ~ best
    single attribute), and the share of residual variance they explain;
  * slope consistency: per-well slopes vs the pooled slope;
  * elastic-log calibration: density / Vp / Vs in reference lithologies (tight limestone,
    shale) per well. Real calibration problems show as offsets in the *same* rock;
  * Vs provenance: offsets for wells with measured vs predicted shear.

Usage:  python scripts/diagnose_offsets.py
Outputs: results/phase3/offsets/{summary.md, per_well_*.csv, *.png}
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

from mars import dataset, paths  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mars.attributes import BASE_ATTRIBUTES, AttributeLibrary  # noqa: E402
from mars.conditioning import load_well_zone  # noqa: E402
from mars.io.las import read_las  # noqa: E402

DATA = paths.RAW
OUT = paths.RESULTS / "phase3" / "offsets"
WELLS = dataset.wells("first12")
MEASURED_VS = set(dataset.wells("measured_vs"))  # availability sheet: only these have DTS


def load(zone: str) -> pd.DataFrame:
    frames = []
    for w in WELLS:
        try:
            z = load_well_zone(DATA, w, zone)
        except Exception as e:  # noqa: BLE001
            print(f"skip {w}: {e}")
            continue
        # fraction of the zone's samples where shear is measured (curve S-Velocity present)
        raw = read_las(dataset.las_path(DATA, w))
        meas = raw.set_index("DEPT")["S-VELOCITY"] if "S-VELOCITY" in raw else None
        z["VS_MEASURED"] = meas.reindex(z["DEPT"]).notna().to_numpy() if meas is not None else False
        if len(z) >= 30:
            frames.append(z)
    return pd.concat(frames, ignore_index=True)


def best_single(df: pd.DataFrame, target: str) -> str:
    d = df.dropna(subset=[target])
    lib = AttributeLibrary().fit(d[list(BASE_ATTRIBUTES)])
    X = lib.transform(d[list(BASE_ATTRIBUTES)])
    X = X.loc[:, np.isfinite(X.to_numpy()).all(0)]
    r = X.apply(lambda c: np.corrcoef(c, d[target])[0, 1])
    return r.abs().idxmax(), lib


def offset_analysis(df: pd.DataFrame, target: str, zone: str) -> tuple[pd.DataFrame, dict, str]:
    d = df.dropna(subset=[target]).copy()
    attr, lib = best_single(d, target)
    d["x"] = lib.transform(d[list(BASE_ATTRIBUTES)], columns=[attr])[attr]
    xs = (d["x"] - d["x"].mean()) / d["x"].std()
    b, a = np.polyfit(xs, d[target], 1)
    d["resid"] = d[target] - (a + b * xs)
    rows = []
    for w, g in d.groupby("WELL"):
        gx = (g["x"] - d["x"].mean()) / d["x"].std()
        bw, aw = np.polyfit(gx, g[target], 1) if len(g) > 10 and gx.std() > 0 else (np.nan, np.nan)
        rows.append({"well": w, "n": len(g), "vs_measured_frac": g["VS_MEASURED"].mean(),
                     f"mean_{target}": g[target].mean(), "mean_resid_offset": g["resid"].mean(),
                     "within_r": np.corrcoef(g["x"], g[target])[0, 1] if g["x"].std() > 0 else np.nan,
                     "slope_within": bw, "slope_pooled": b, "intercept_within": aw})
    pw = pd.DataFrame(rows).sort_values("mean_resid_offset")
    off = d.groupby("WELL")["resid"].transform("mean")
    share = float(off.var() / d["resid"].var())
    stats = {"zone": zone, "target": target, "attr": attr,
             "pooled_r": float(np.corrcoef(d["x"], d[target])[0, 1]),
             "offset_share_of_resid_var": share,
             "r_after_removing_offsets": float(np.corrcoef(d["x"], d[target] - off)[0, 1]),
             "slope_cv": float(pw["slope_within"].std() / abs(pw["slope_within"].mean()))}
    # plot: target vs attribute coloured by well, per-well lines
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for w, g in d.groupby("WELL"):
        ax[0].scatter(g["x"], g[target], s=3, alpha=0.4, label=w)
        if len(g) > 10:
            p = np.polyfit(g["x"], g[target], 1)
            xx = np.linspace(g["x"].min(), g["x"].max(), 10)
            ax[0].plot(xx, np.polyval(p, xx), lw=1.5)
    ax[0].set_xlabel(attr)
    ax[0].set_ylabel(target)
    ax[0].set_title(f"{zone} {target} vs best single attribute (per-well fits)")
    ax[0].legend(markerscale=4, fontsize=7, ncol=2)
    pw_sorted = pw.set_index("well")
    colors = ["tab:red" if v > 0.5 else "tab:blue" for v in pw_sorted["vs_measured_frac"]]
    ax[1].barh(pw_sorted.index, pw_sorted["mean_resid_offset"], color=colors)
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].set_title(f"per-well offset (mean residual); red = measured Vs\n"
                    f"offsets explain {share:.0%} of residual variance")
    fig.tight_layout()
    fig.savefig(OUT / f"offsets_{zone}_{target}.png", dpi=110)
    plt.close(fig)
    return pw, stats, attr


def reference_lithology(df: pd.DataFrame, zone: str) -> pd.DataFrame:
    """Median elastic logs per well in reference rocks; calibration issues show here."""
    out = []
    for name, sel in {"tight_limestone (PHIE<0.04, VSH<0.15)": (df.PHIE < 0.04) & (df.VSH < 0.15),
                      "shale (VSH>0.6)": df.VSH > 0.6}.items():
        for w, g in df[sel].groupby("WELL"):
            if len(g) >= 10:
                out.append({"zone": zone, "rock": name, "well": w, "n": len(g),
                            "RHOB": g.RHOB.median(), "VP": g.VP.median(), "VS": g.VS.median(),
                            "VP/VS": (g.VP / g.VS).median()})
    return pd.DataFrame(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stats_all, refs = [], []
    md = ["# Between-well offset diagnosis (Phase 3 step 1)", ""]
    for zone in ("UBR", "LBR"):
        df = load(zone)
        refs.append(reference_lithology(df, zone))
        for target in ("PHIE", "SWT", "VSH"):
            pw, st, attr = offset_analysis(df, target, zone)
            pw.to_csv(OUT / f"per_well_{zone}_{target}.csv", index=False)
            stats_all.append(st)
            print(f"{zone} {target}: attr={attr} pooled r={st['pooled_r']:.2f}  offsets explain "
                  f"{st['offset_share_of_resid_var']:.0%} of resid var  r after removing offsets="
                  f"{st['r_after_removing_offsets']:.2f}  slope CV={st['slope_cv']:.2f}", flush=True)
    st = pd.DataFrame(stats_all)
    st.to_csv(OUT / "summary_stats.csv", index=False)
    ref = pd.concat(refs, ignore_index=True)
    ref.to_csv(OUT / "reference_lithology.csv", index=False)
    md += ["## Offsets vs best single attribute", "", st.round(3).to_markdown(index=False), "",
           "## Elastic logs in reference rocks (calibration check)", "",
           ref.round(3).to_markdown(index=False), ""]
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", OUT / "summary.md")


if __name__ == "__main__":
    main()
