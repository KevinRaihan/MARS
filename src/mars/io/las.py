"""LAS reading and integrity checks.

Curve names are normalised to upper case so that both Petrel export schemas map onto one
vocabulary (see vault 03-Data/wells.md).
"""

from __future__ import annotations

import re
from pathlib import Path

import lasio
import numpy as np
import pandas as pd

# canonical name -> accepted aliases (upper case)
CURVE_ALIASES: dict[str, tuple[str, ...]] = {
    "DEPT": ("DEPT", "DEPTH", "MD"),
    "VP": ("P-VELOCITY",),
    "VS": ("S-VELOCITY_FINAL",),
    "VS_MEAS": ("S-VELOCITY", "S-VELOCITYORIGINAL"),
    "RHOB": ("RHOB", "RHOB_ED"),
    "PHIE": ("PHIE",),
    "PHIT": ("PHIT",),
    "VSH": ("VSH",),
    "SWT": ("SWT",),
    "GR": ("GR_ED", "GR"),
}


def read_las(path: str | Path, canonical: bool = False) -> pd.DataFrame:
    """Read a LAS file into a DataFrame with a ``DEPT`` column (MD, ft) and upper-case curve names.

    Null values (-999.25) become NaN. With ``canonical=True`` the known curves are renamed to
    the keys of :data:`CURVE_ALIASES` (``VP``, ``VS``, ``RHOB`` ...); other curves keep their names.
    """
    las = lasio.read(str(path), ignore_header_errors=True)
    df = las.df().reset_index()
    df.columns = [str(c).strip().upper() for c in df.columns]
    df = df.rename(columns={df.columns[0]: "DEPT"})
    if canonical:
        rename = {}
        for canon, aliases in CURVE_ALIASES.items():
            for a in aliases:
                if a in df.columns and canon not in rename.values():
                    rename[a] = canon
                    break
        df = df.rename(columns=rename)
    return df


def scan_las_integrity(path: str | Path) -> dict:
    """Scan the ~A section of a LAS file for signs of corruption.

    Reports non-printable bytes, rows with non-numeric tokens or the wrong token count, and
    depth steps that differ from the header STEP. Does not depend on lasio, so it works even on
    files lasio cannot parse.
    """
    raw = Path(path).read_bytes()
    nonprint = int(np.count_nonzero(np.frombuffer(raw, dtype=np.uint8) > 126)) + sum(
        1 for b in raw if b < 9 or 13 < b < 32
    )
    sec, ncurves, step, prev, rows = None, 0, None, None, 0
    issues: list[tuple[int, str]] = []
    for ln, line in enumerate(raw.decode("latin-1").splitlines(), 1):
        if line.startswith("~"):
            sec = line[1:2].upper()
            continue
        if sec == "W" and line.strip().upper().startswith("STEP"):
            m = re.search(r"(-?\d+\.?\d*)", line.split(":")[0].split(".", 1)[-1])
            step = float(m.group(1)) if m else None
        elif sec == "C" and line.strip() and not line.startswith("#"):
            ncurves += 1
        elif sec == "A" and line.strip():
            rows += 1
            try:
                vals = [float(t) for t in line.split()]
            except ValueError:
                issues.append((ln, "non-numeric token"))
                prev = None
                continue
            if len(vals) != ncurves:
                issues.append((ln, f"{len(vals)} tokens, expected {ncurves}"))
            if prev is not None and step and abs(vals[0] - prev - step) > 1e-3:
                issues.append((ln, f"depth jump {prev} -> {vals[0]}"))
            prev = vals[0]
    return {"rows": rows, "nonprint_bytes": nonprint, "n_issues": len(issues), "issues": issues}
