# mars — closed-form Multi-Attribute Rotation Scheme

Python code for predicting reservoir properties (porosity PHIE, shale volume VSH, water saturation SWT) from elastic attributes with the Multi-Attribute Rotation Scheme (MARS; Alvarez et al., 2015, *Interpretation* 3(4), SAE9–SAE18), validating the transforms blind, and applying them to seismic inversion cubes. It accompanies the paper *Closed-form multi-attribute rotation and spatially validated prediction of carbonate reservoir properties: Baturaja Formation, Sunda Basin* (in preparation; source in [`paper/`](paper/)).

What it does:
- **Closed-form MARS.** The best rotation of $N$ standardised attributes is $\mathbf{u}^\ast \propto \mathbf{C}^{-1}\mathbf{r}$ with $R^2 = \mathbf{r}^\top\mathbf{C}^{-1}\mathbf{r}$, so no angle grid is needed; an exhaustive best-subset search takes seconds (`mars.search.closed_form_search`). Results convert back to the published MARS angles (`MarsTransform.paper_angles`). `mars.legacy` keeps the original grid-search algorithm, bit-for-bit, for comparison.
- **Validation.** Leave-one-well-out with the attribute selection nested inside each fold, sidetracks grouped with their parent well, spatial holdouts, and baselines (OLS, LASSO, gradient boosting).
- **Conditioning.** Per-curve missing values, padded-constant removal, Backus upscaling.
- **Seismic application.** Chunked GPU (CuPy) or CPU application to Zp / density cubes with NaN-aware smoothing, well ties, horizon gridding, interval and reservoir maps, sections along deviated well paths, and a 3D scene.

## Setup
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[seismic,viz3d,dev]"   # add ,gpu for CuPy (CUDA 12)
```
or `pip install -e ".[seismic,dev]"`.

## Data
The well and seismic data used in the paper are confidential and are not distributed. The code expects this layout (locations overridable with environment variables, see `mars.paths`):

| variable | default | contents |
|---|---|---|
| `MARS_DATA_DIR` | `data/raw` | `wells/*.las`, `tops/<well>_Marker.txt`, `tops/welltops.csv`, `horizons/*.xyt` |
| `MARS_INTERIM_DIR` | `data/interim` | derived large data: `cubes/*.zarr`, `props/*.zarr` |
| `MARS_RESULTS_DIR` | `results` | tables and figures |
| `MARS_SEISMIC_DIR` | `data/seismic` | source SEG-Y inversion cubes (`decimate` stage only) |

Two small JSON files next to `raw/` describe a data set and are never committed:
- `dataset.json`: LAS file naming and the named well sets the scripts use (all wells, holdout regions, section lines, ...). Template: [`configs/dataset.example.json`](configs/dataset.example.json).
- `well_codenames.json`: `{"<well id>": "<CODENAME>"}`. Every figure and table meant for sharing labels wells by codename (`mars.naming.codename`) and uses coordinates relative to the survey corner.

## Running
```bash
python scripts/run_pipeline.py --list            # stages and whether their output exists
python scripts/run_pipeline.py --group wells     # well-log analysis only (no cubes needed)
python scripts/run_pipeline.py                   # everything that is missing
python scripts/run_pipeline.py --only transforms --force
```

| stage | script | output |
|---|---|---|
| parity | `run_legacy_parity.py` | the original (grid-search) results reproduced in legacy mode |
| ablation | `run_ablation.py --multiwell` | what each fix to the original workflow changes |
| offsets, phase3 | `diagnose_offsets.py`, `run_phase3.py` | predicted-Vs leakage, Vs-free library, upscaling length, BRF zone |
| phase4, reservoir_rule | `run_phase4.py`, `run_reservoir_rule.py` | 12 vs 22 wells; two-step lower-reservoir rule |
| transforms | `fit_final_transforms.py` | `results/transforms/*.json` (+ Zp-only controls) |
| spatial_blind | `run_spatial_blind.py` | holdout regions fixed in advance, predicted once |
| baselines | `run_baselines.py` | MARS 1–3D against OLS, LASSO, GBM (in-sample and blind) |
| closed_form | `run_closed_form_check.py` | closed form against a 1° grid search: optima, chosen subsets, run time |
| decimate … surfaces | `decimate_cubes.py`, `qc_decimation.py`, `tie_wells.py`, `build_surfaces.py` | 1 ms Zarr cubes, well tie, zone surfaces |
| props, props_s0, props_s50 | `apply_seismic_mars.py` | property cubes at lateral σ = 25 / 0 / 50 m |
| qc_props, figures | `qc_seismic_props.py`, `make_phase5_figures.py` | QC at the wells; maps, sections, 3D scene |

The paper's tables and figures are generated from `results/` by `paper/make_tables.py` and `paper/make_figures.py`; `bash paper/build.sh [--figures] [en] [id]` compiles it with [tectonic](https://tectonic-typesetting.github.io/).

## Package layout (`src/mars`)
- `io/` — LAS, markers/zones, SEG-Y geometry and decimation.
- `attributes`, `search`, `transform` — attribute library, closed-form best-subset MARS, JSON transforms; `legacy` and `thesis_reference` for the original algorithm.
- `conditioning`, `validation` — zone loading, Backus upscaling, leave-one-well-out CV with nested selection, baselines.
- `tie`, `horizons`, `seismic_apply` — well paths in time, horizon gridding, chunked GPU/CPU cube application.
- `viz/` — sections along well paths and interval / net / reservoir maps.
- `dataset`, `naming`, `paths` — data-set configuration, codenames, locations.

## Tests
```bash
python -m pytest -q
```
Tests that need the real data skip when it is absent; the rest use synthetic data.

## Citation
If you use this code, please cite the paper (reference to follow) and Alvarez et al. (2015).

## Licence
MIT (see [`LICENSE`](LICENSE)). Copyright (c) 2026 Kevin Raihan Yassin.
