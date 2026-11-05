from pathlib import Path
import argparse, json, math
import numpy as np
import pandas as pd

from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from skimage.morphology import disk
import tifffile

RADII_UM = [0.5, 1.0, 1.5, 2.0]
MPP_EXPECTED = 0.16
COVERAGES = [0.5, 0.6, 0.7, 0.8, 0.9]
EPS = 1e-8

# ---------------------------------------------------------------------
# Frozen Histo10b implementation, reproduced from the original package.
# ---------------------------------------------------------------------
def overlap_pairs(gt, pred):
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    gi = np.unique(gt); gi = gi[gi > 0]
    pi = np.unique(pred); pi = pi[pi > 0]
    if not len(gi) or not len(pi):
        return [], len(gi), len(pi)

    mask = (gt > 0) & (pred > 0)
    gg = gt[mask]; pp = pred[mask]
    gidx = np.searchsorted(gi, gg)
    pidx = np.searchsorted(pi, pp)
    inter = np.bincount(
        gidx * len(pi) + pidx,
        minlength=len(gi) * len(pi)
    ).reshape(len(gi), len(pi))

    ga = np.bincount(
        gt.ravel().astype(np.int64),
        minlength=int(gi.max()) + 1
    )[gi]
    pa = np.bincount(
        pred.ravel().astype(np.int64),
        minlength=int(pi.max()) + 1
    )[pi]

    union = ga[:, None] + pa[None, :] - inter
    iou = np.divide(
        inter, union,
        out=np.zeros_like(inter, dtype=float),
        where=union > 0
    )
    rr, cc = linear_sum_assignment(-iou)
    return [
        (int(gi[r]), int(pi[c]), float(iou[r, c]))
        for r, c in zip(rr, cc)
        if inter[r, c] > 0
    ], len(gi), len(pi)

def sape(a, b, eps=EPS):
    a = float(a); b = float(b)
    return 2 * abs(a - b) / (abs(a) + abs(b) + eps)

def span(a, b, eps=EPS):
    a = float(a); b = float(b)
    return 2 * abs(a - b) / (abs(a) + abs(b) + eps)

def expand_slice(sl, shape, pad):
    return (
        slice(max(0, sl[0].start - pad), min(shape[0], sl[0].stop + pad)),
        slice(max(0, sl[1].start - pad), min(shape[1], sl[1].stop + pad)),
    )

def read_tif(p):
    return np.asarray(tifffile.imread(p))

def load_label(p):
    last = None
    for d in (None, ",", "\t"):
        try:
            x = np.loadtxt(p, delimiter=d)
            if x.size:
                return np.rint(x).astype(np.int32)
        except Exception as e:
            last = e
    raise RuntimeError(f"Cannot parse {p}: {last}")

def organ_from_case(c):
    x = c.lower()
    if "smallintestine" in x or "small_intestine" in x or "small intestine" in x:
        return "small_intestine"
    for o in ["appendix", "colon", "esophagus", "gallbladder", "liver", "pancreas", "stomach"]:
        if o in x:
            return o
    return "unknown"

def membrane(img):
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[-1] >= 3:
        return np.asarray(a[..., 0], float)
    if a.ndim == 3 and a.shape[0] >= 3 and a.shape[0] <= 4:
        return np.asarray(a[0], float)
    raise ValueError(f"Unexpected TIFF shape {a.shape}")

def cell_rows(pred, gt, img, mpp, case, organ, radius_um):
    pred = np.asarray(pred, np.int32)
    gt = np.asarray(gt, np.int32)
    img = np.asarray(img, float)

    pairs, _, _ = overlap_pairs(gt, pred)
    match = {p: (g, iou) for g, p, iou in pairs if iou >= 0.5}
    if not match:
        return []

    maxg = int(gt.max()); maxp = int(pred.max())
    ga = np.bincount(gt.ravel().astype(np.int64), minlength=maxg + 1)
    pa = np.bincount(pred.ravel().astype(np.int64), minlength=maxp + 1)
    gs = np.bincount(
        gt.ravel().astype(np.int64),
        weights=img.ravel(),
        minlength=maxg + 1
    )
    ps = np.bincount(
        pred.ravel().astype(np.int64),
        weights=img.ravel(),
        minlength=maxp + 1
    )

    slices = ndi.find_objects(pred)

    # IMPORTANT: this is exactly the frozen Histo10b discretization rule.
    rpx = max(1, int(round(float(radius_um) / float(mpp))))
    foot = disk(rpx).astype(bool)

    out = []
    for p, (g, iou) in match.items():
        if p - 1 >= len(slices) or slices[p - 1] is None:
            continue

        sl = expand_slice(slices[p - 1], pred.shape, rpx + 1)
        loc = pred[sl] == p
        inner = ndi.binary_erosion(loc, structure=foot, border_value=0)
        outer = ndi.binary_dilation(loc, structure=foot, border_value=0)
        vals = img[sl]

        ai = float(inner.sum())
        ao = float(outer.sum())
        a0 = float(pa[p])
        ag = float(ga[g])
        if min(ai, ao, a0, ag) <= 0:
            continue

        mi = float(vals[inner].mean()) if inner.any() else np.nan
        mo = float(vals[outer].mean()) if outer.any() else np.nan
        m0 = float(ps[p] / a0)
        mg = float(gs[g] / ag)

        out.append(dict(
            case=case,
            organ=organ,
            gt_label=int(g),
            pred_label=int(p),
            matched_iou=float(iou),
            sms_mean=span(mi, mo),
            geometry_span=span(ai, ao),
            inverse_sqrt_area=1.0 / math.sqrt(max(a0, 1.0)),
            actual_mean_error=sape(m0, mg),
            raw_mean=m0,
            gt_mean=mg,
            inner_mean=mi,
            outer_mean=mo,
            raw_area=a0,
            gt_area=ag,
            radius_um=float(radius_um),
            radius_px=int(rpx),
            effective_radius_um=float(rpx * mpp),
            mpp=float(mpp),
        ))
    return out

def evaluate_group(rr, score):
    x = np.asarray([r[score] for r in rr], float)
    y = np.asarray([r["actual_mean_error"] for r in rr], float)
    k = np.isfinite(x) & np.isfinite(y)
    x = x[k]; y = y[k]
    if len(y) < 10 or np.std(x) == 0:
        return None

    order = np.argsort(x, kind="mergesort")
    base = float(y.mean())
    d = {
        "n_cells": int(len(y)),
        "spearman": float(spearmanr(x, y).statistic),
        "overall_risk": base,
    }
    for c in COVERAGES:
        n = max(1, int(np.floor(c * len(y))))
        risk = float(y[order[:n]].mean())
        d[f"risk_{int(c*100)}"] = risk
        d[f"risk_ratio_{int(c*100)}"] = risk / base if base > 0 else np.nan
    return d

def case_metrics(cells, score="sms_mean"):
    groups = {}
    for r in cells:
        groups.setdefault(r["case"], []).append(r)

    rows = []
    for case, rr in groups.items():
        e = evaluate_group(rr, score)
        if e is not None:
            rows.append({
                "case": case,
                "organ": rr[0]["organ"],
                "radius_um": float(rr[0]["radius_um"]),
                "radius_px": int(rr[0]["radius_px"]),
                "effective_radius_um": float(rr[0]["effective_radius_um"]),
                **e,
            })
    return rows

def aggregate_case_metrics(rows):
    def mf(k):
        a = np.asarray([r.get(k, np.nan) for r in rows], float)
        return float(np.nanmean(a)) if np.isfinite(a).any() else np.nan

    return {
        "n_evaluable_cases": int(len(rows)),
        "n_cells_in_evaluable_cases": int(sum(r["n_cells"] for r in rows)),
        "mean_spearman": mf("spearman"),
        "risk_ratio_50": mf("risk_ratio_50"),
        "risk_ratio_60": mf("risk_ratio_60"),
        "risk_ratio_70": mf("risk_ratio_70"),
        "risk_ratio_80": mf("risk_ratio_80"),
        "risk_ratio_90": mf("risk_ratio_90"),
    }

def bootstrap_case_metrics(rows, B=5000, seed=20261104):
    if not rows:
        return {}
    rng = np.random.default_rng(seed)
    n = len(rows)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        b = [rows[i] for i in idx]
        vals.append((
            np.nanmean([r["risk_ratio_70"] for r in b]),
            np.nanmean([r["spearman"] for r in b]),
        ))
    a = np.asarray(vals, float)
    return {
        "risk_ratio_70_95ci": [
            float(np.nanpercentile(a[:, 0], 2.5)),
            float(np.nanpercentile(a[:, 0], 97.5)),
        ],
        "spearman_95ci": [
            float(np.nanpercentile(a[:, 1], 2.5)),
            float(np.nanpercentile(a[:, 1], 97.5)),
        ],
    }

def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)

def normalized_case(x):
    return str(x).strip().lower()

def validate_one_um_reconstruction(new_rows, frozen_csv):
    frozen = pd.read_csv(frozen_csv)
    new = pd.DataFrame(new_rows)

    key = ["case", "gt_label", "pred_label"]
    for df in (frozen, new):
        df["case"] = df["case"].astype(str)
        df["gt_label"] = pd.to_numeric(df["gt_label"], errors="raise").astype(int)
        df["pred_label"] = pd.to_numeric(df["pred_label"], errors="raise").astype(int)

    frozen = frozen.sort_values(key, kind="mergesort").reset_index(drop=True)
    new = new.sort_values(key, kind="mergesort").reset_index(drop=True)

    checks = {
        "frozen_rows": int(len(frozen)),
        "reconstructed_rows": int(len(new)),
        "row_count_equal": bool(len(frozen) == len(new)),
    }

    old_keys = list(map(tuple, frozen[key].to_numpy()))
    new_keys = list(map(tuple, new[key].to_numpy()))
    checks["identity_keys_equal"] = bool(old_keys == new_keys)

    numeric_cols = [
        "matched_iou",
        "sms_mean",
        "geometry_span",
        "inverse_sqrt_area",
        "actual_mean_error",
        "raw_mean",
        "gt_mean",
        "inner_mean",
        "outer_mean",
        "raw_area",
        "gt_area",
        "radius_um",
        "radius_px",
        "mpp",
    ]
    diffs = {}
    all_numeric_ok = True

    if checks["row_count_equal"] and checks["identity_keys_equal"]:
        for c in numeric_cols:
            if c not in frozen.columns or c not in new.columns:
                diffs[c] = {"status": "missing_column"}
                all_numeric_ok = False
                continue
            a = pd.to_numeric(frozen[c], errors="coerce").to_numpy(float)
            b = pd.to_numeric(new[c], errors="coerce").to_numpy(float)
            d = np.abs(a - b)
            finite = np.isfinite(d)
            max_abs = float(np.max(d[finite])) if finite.any() else 0.0
            same_nan = bool(np.array_equal(np.isnan(a), np.isnan(b)))
            ok = bool(same_nan and np.allclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True))
            diffs[c] = {
                "max_abs_diff": max_abs,
                "allclose": ok,
            }
            all_numeric_ok = all_numeric_ok and ok
    else:
        all_numeric_ok = False

    checks["numeric_comparison"] = diffs
    checks["all_numeric_columns_match"] = bool(all_numeric_ok)
    checks["PASS"] = bool(
        checks["row_count_equal"]
        and checks["identity_keys_equal"]
        and checks["all_numeric_columns_match"]
        and len(new) == 6658
    )
    return checks

def common_cell_rows(all_rows):
    df = pd.DataFrame(all_rows)
    key = ["case", "gt_label", "pred_label"]
    radius_sets = {}
    for r in RADII_UM:
        s = df[np.isclose(df["radius_um"], r)]
        radius_sets[r] = set(map(tuple, s[key].to_numpy()))
    common = set.intersection(*radius_sets.values())
    mask = df[key].apply(tuple, axis=1).isin(common)
    return df.loc[mask].copy(), common

def summarize_dataframe_by_radius(df):
    summaries = []
    per_case_all = []
    for r in RADII_UM:
        sub = df[np.isclose(df["radius_um"], r)].copy()
        rows = sub.to_dict("records")
        cms = case_metrics(rows, "sms_mean")
        agg = aggregate_case_metrics(cms)
        boot = bootstrap_case_metrics(cms)
        radius_px_vals = sorted(sub["radius_px"].dropna().astype(int).unique().tolist())
        eff_vals = sorted(sub["effective_radius_um"].dropna().astype(float).unique().tolist())
        summaries.append({
            "requested_radius_um": float(r),
            "radius_px": radius_px_vals[0] if len(radius_px_vals) == 1 else str(radius_px_vals),
            "effective_radius_um": eff_vals[0] if len(eff_vals) == 1 else str(eff_vals),
            "raw_cell_rows": int(len(sub)),
            **agg,
            "rr70_ci_low": boot.get("risk_ratio_70_95ci", [np.nan, np.nan])[0],
            "rr70_ci_high": boot.get("risk_ratio_70_95ci", [np.nan, np.nan])[1],
            "spearman_ci_low": boot.get("spearman_95ci", [np.nan, np.nan])[0],
            "spearman_ci_high": boot.get("spearman_95ci", [np.nan, np.nan])[1],
        })
        for x in cms:
            per_case_all.append({"requested_radius_um": float(r), **x})
    return summaries, per_case_all

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Experiment 2: Mesmer physical-scale robustness on frozen UNSEG. "
            "Hard-gates on exact 1.0-um reproduction before evaluating 0.5/1.5/2.0 um."
        )
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(r"E:\UNSEG_GIT"),
    )
    ap.add_argument(
        "--mesmer",
        type=Path,
        default=Path(r"E:\UNSEG_GIT\histo09_run\mesmer_predictions.npz"),
    )
    ap.add_argument(
        "--frozen-rows",
        type=Path,
        default=Path(r"E:\Histo_Pilot\histo10b_mean_sms\run_20261102_151526\unseg_cell_rows.csv"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(r"E:\Histo_Pilot\sms_backbone_transfer\physical_scale_mesmer"),
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.mesmer.exists():
        raise FileNotFoundError(
            f"Frozen UNSEG Mesmer source not found:\n{args.mesmer}\n"
            "This is the exact source used by the original Histo10b run."
        )
    if not args.frozen_rows.exists():
        raise FileNotFoundError(args.frozen_rows)

    tifs = sorted(args.root.glob("*.tif"))
    if len(tifs) != 75:
        raise RuntimeError(f"Expected 75 UNSEG TIFFs, found {len(tifs)}")

    print("Loading frozen Mesmer source:", args.mesmer)
    with np.load(args.mesmer, allow_pickle=False) as z:
        required = {"case_names", "whole_cell", "pixel_size"}
        missing = required - set(z.files)
        if missing:
            raise RuntimeError(f"Mesmer NPZ missing keys: {sorted(missing)}; keys={z.files}")
        names = z["case_names"].astype(str).tolist()
        pred = np.asarray(z["whole_cell"], np.int32)
        mpp = np.asarray(z["pixel_size"], float)

    expected_names = [t.stem for t in tifs]
    if names != expected_names:
        raise RuntimeError("Mesmer case order mismatch with the 75 sorted UNSEG TIFFs.")
    if pred.shape[0] != 75:
        raise RuntimeError(f"Expected 75 Mesmer masks, got shape {pred.shape}")
    if not np.allclose(mpp, MPP_EXPECTED):
        raise RuntimeError(f"Mesmer pixel_size is not uniformly {MPP_EXPECTED}: {np.unique(mpp)}")

    print("Mesmer whole_cell shape:", pred.shape)
    print("Mesmer pixel_size:", sorted(np.unique(mpp).tolist()))
    print()

    # -------------------------------------------------------------
    # GATE 1: reconstruct exact frozen 1.0-um analysis.
    # -------------------------------------------------------------
    print("GATE 1: reconstructing frozen 1.0-um SMS...")
    one_rows = []
    for i, t in enumerate(tifs):
        gt = load_label(args.root / (t.stem + "_annotation_cell.txt"))
        mem = membrane(read_tif(t))
        rr = cell_rows(
            pred[i], gt, mem, float(mpp[i]),
            t.stem, organ_from_case(t.stem), 1.0
        )
        one_rows.extend(rr)
        print(f"  1.0 um {i+1:02d}/75  cells={len(rr)}", flush=True)

    one_csv = args.out / "reconstructed_1um_cell_rows.csv"
    write_csv(one_csv, one_rows)

    gate = validate_one_um_reconstruction(one_rows, args.frozen_rows)
    gate_json = args.out / "reproduction_gate_1um.json"
    gate_json.write_text(json.dumps(gate, indent=2), encoding="utf-8")

    print()
    print("1.0-um reproduction gate")
    print("-" * 55)
    print("Frozen rows:       ", gate["frozen_rows"])
    print("Reconstructed rows:", gate["reconstructed_rows"])
    print("Identity keys equal:", gate["identity_keys_equal"])
    print("Numeric match:      ", gate["all_numeric_columns_match"])
    print("GATE PASS:          ", gate["PASS"])

    if not gate["PASS"]:
        print("\nSTOP: 1.0-um reconstruction did not reproduce the frozen baseline.")
        print("Do NOT interpret 0.5/1.5/2.0 um until this mismatch is resolved.")
        print("Saved:", gate_json)
        raise SystemExit(40)

    # -------------------------------------------------------------
    # GATE PASSED: compute the remaining radii.
    # 1.0 rows are reused exactly; no duplicate recomputation.
    # -------------------------------------------------------------
    print("\nGATE PASSED. Computing physical-scale sensitivity...")
    all_rows = list(one_rows)

    for r in [0.5, 1.5, 2.0]:
        print(f"\nRequested radius = {r:.1f} um")
        r_rows = []
        for i, t in enumerate(tifs):
            gt = load_label(args.root / (t.stem + "_annotation_cell.txt"))
            mem = membrane(read_tif(t))
            rr = cell_rows(
                pred[i], gt, mem, float(mpp[i]),
                t.stem, organ_from_case(t.stem), r
            )
            r_rows.extend(rr)
            print(f"  {r:.1f} um {i+1:02d}/75  cells={len(rr)}", flush=True)
        all_rows.extend(r_rows)

    all_df = pd.DataFrame(all_rows)
    all_df = all_df.sort_values(
        ["radius_um", "case", "gt_label", "pred_label"],
        kind="mergesort"
    ).reset_index(drop=True)
    all_csv = args.out / "mesmer_physical_scale_all_cell_rows.csv"
    all_df.to_csv(all_csv, index=False)

    # Native per-radius summaries: reproduces the Histo10b estimand separately at each radius.
    native_summary, native_per_case = summarize_dataframe_by_radius(all_df)
    native_csv = args.out / "mesmer_physical_scale_native_summary.csv"
    pd.DataFrame(native_summary).to_csv(native_csv, index=False)
    pd.DataFrame(native_per_case).to_csv(
        args.out / "mesmer_physical_scale_native_per_case.csv",
        index=False
    )

    # Common-cell sensitivity: same cell identities across all four radii.
    common_df, common_ids = common_cell_rows(all_df)
    common_summary, common_per_case = summarize_dataframe_by_radius(common_df)
    common_csv = args.out / "mesmer_physical_scale_commoncell_summary.csv"
    pd.DataFrame(common_summary).to_csv(common_csv, index=False)
    pd.DataFrame(common_per_case).to_csv(
        args.out / "mesmer_physical_scale_commoncell_per_case.csv",
        index=False
    )
    common_df.to_csv(
        args.out / "mesmer_physical_scale_commoncell_rows.csv",
        index=False
    )

    discretization = []
    for r in RADII_UM:
        px = max(1, int(round(r / MPP_EXPECTED)))
        discretization.append({
            "requested_radius_um": r,
            "pixel_size_um": MPP_EXPECTED,
            "radius_px_frozen_round_rule": px,
            "effective_radius_um": px * MPP_EXPECTED,
            "absolute_discretization_error_um": abs(px * MPP_EXPECTED - r),
        })

    report = {
        "protocol": (
            "Experiment 2: post-hoc physical-scale robustness of frozen mean-intensity SMS "
            "on UNSEG using the exact frozen Mesmer masks."
        ),
        "claim_status": (
            "Sensitivity/robustness analysis only. The frozen primary operating point remains "
            "1.0 um and 70% coverage; no radius is reselected from UNSEG."
        ),
        "source": {
            "unseg_root": str(args.root),
            "frozen_mesmer_npz": str(args.mesmer),
            "frozen_reference_rows": str(args.frozen_rows),
            "n_images": 75,
            "pixel_size_um": MPP_EXPECTED,
        },
        "one_um_reproduction_gate": gate,
        "radii_discretization": discretization,
        "native_per_radius_summary": native_summary,
        "common_cell_definition": {
            "identity_key": ["case", "gt_label", "pred_label"],
            "n_common_cell_identities": int(len(common_ids)),
            "purpose": (
                "Separate true radius sensitivity from changes caused only by radius-dependent "
                "loss of cells whose eroded mask becomes empty."
            ),
        },
        "common_cell_per_radius_summary": common_summary,
        "interpretation_rule": (
            "Robustness is supported if the direction of selective risk reduction remains "
            "consistent over 0.5, 1.0, 1.5, and 2.0 um, especially on the common-cell analysis. "
            "This experiment must not be used to replace the frozen 1.0-um primary setting."
        ),
    }
    report_json = args.out / "mesmer_physical_scale_report.json"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 82)
    print("PHYSICAL-SCALE RESULTS — NATIVE PER-RADIUS")
    print("=" * 82)
    print(pd.DataFrame(native_summary)[[
        "requested_radius_um",
        "radius_px",
        "effective_radius_um",
        "raw_cell_rows",
        "n_evaluable_cases",
        "mean_spearman",
        "risk_ratio_70",
        "rr70_ci_low",
        "rr70_ci_high",
    ]].to_string(index=False))

    print("\n" + "=" * 82)
    print("COMMON-CELL SENSITIVITY")
    print("Common cell identities:", len(common_ids))
    print("=" * 82)
    print(pd.DataFrame(common_summary)[[
        "requested_radius_um",
        "radius_px",
        "effective_radius_um",
        "raw_cell_rows",
        "n_evaluable_cases",
        "mean_spearman",
        "risk_ratio_70",
        "rr70_ci_low",
        "rr70_ci_high",
    ]].to_string(index=False))

    print("\nDiscretization:")
    print(pd.DataFrame(discretization).to_string(index=False))

    print("\nSaved to:", args.out)
    print("  ", gate_json)
    print("  ", native_csv)
    print("  ", common_csv)
    print("  ", report_json)
    print("\nExperiment 2 completed. Do not choose a new radius from these UNSEG results.")

if __name__ == "__main__":
    main()