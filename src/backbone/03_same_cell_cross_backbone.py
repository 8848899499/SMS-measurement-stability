from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SEED = 20261104
N_BOOT = 5000
COVERAGE = 0.70
MIN_COMMON_PER_IMAGE = 10

def norm_case(x):
    s = str(x).replace("\\", "/").split("/")[-1]
    if s.lower().endswith(".tif"):
        s = s[:-4]
    elif s.lower().endswith(".tiff"):
        s = s[:-5]
    return s.strip().lower()

def require_cols(df, name):
    needed = ["case", "gt_label", "sms_mean", "actual_mean_error"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{name}: missing required columns {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

def load_rows(path, prefix):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    require_cols(df, prefix)
    df = df.copy()
    df["case_norm"] = df["case"].map(norm_case)
    df["gt_label"] = pd.to_numeric(df["gt_label"], errors="raise").astype(int)

    dup = int(df.duplicated(["case_norm", "gt_label"]).sum())
    if dup:
        raise RuntimeError(
            f"{prefix}: {dup} duplicate (case, gt_label) rows. "
            "Same-cell pairing would be ambiguous."
        )

    keep = ["case", "case_norm", "gt_label", "sms_mean", "actual_mean_error"]
    for c in ["matched_iou", "raw_area", "raw_mean", "pred_label"]:
        if c in df.columns:
            keep.append(c)
    df = df[keep].rename(columns={
        "case": f"{prefix}_case",
        "sms_mean": f"{prefix}_sms",
        "actual_mean_error": f"{prefix}_error",
        "matched_iou": f"{prefix}_iou",
        "raw_area": f"{prefix}_area",
        "raw_mean": f"{prefix}_raw_mean",
        "pred_label": f"{prefix}_pred_label",
    })
    return df

def rho(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    if np.nanstd(x[ok]) == 0 or np.nanstd(y[ok]) == 0:
        return np.nan
    return float(spearmanr(x[ok], y[ok]).statistic)

def rr70(score, err, coverage=COVERAGE):
    score = np.asarray(score, float)
    err = np.asarray(err, float)
    ok = np.isfinite(score) & np.isfinite(err)
    score, err = score[ok], err[ok]
    n = len(err)
    if n < 2:
        return np.nan
    base = float(np.mean(err))
    if not np.isfinite(base) or base <= 0:
        return np.nan
    k = max(1, int(np.ceil(coverage * n)))
    idx = np.argsort(score, kind="mergesort")[:k]
    return float(np.mean(err[idx]) / base)

def case_metrics(g):
    return {
        "n_common": int(len(g)),
        "rho_mesmer_sms__mesmer_error": rho(g["mesmer_sms"], g["mesmer_error"]),
        "rho_cellpose_sms__cellpose_error": rho(g["cellpose_sms"], g["cellpose_error"]),
        "rho_mesmer_sms__cellpose_error": rho(g["mesmer_sms"], g["cellpose_error"]),
        "rho_cellpose_sms__mesmer_error": rho(g["cellpose_sms"], g["mesmer_error"]),
        "rr70_mesmer_sms__mesmer_error": rr70(g["mesmer_sms"], g["mesmer_error"]),
        "rr70_cellpose_sms__cellpose_error": rr70(g["cellpose_sms"], g["cellpose_error"]),
        "rr70_mesmer_sms__cellpose_error": rr70(g["mesmer_sms"], g["cellpose_error"]),
        "rr70_cellpose_sms__mesmer_error": rr70(g["cellpose_sms"], g["mesmer_error"]),
        "rho_sms_between_backbones": rho(g["mesmer_sms"], g["cellpose_sms"]),
        "rho_error_between_backbones": rho(g["mesmer_error"], g["cellpose_error"]),
    }

def mean_finite(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if len(a) else np.nan

def ci(v, lo=2.5, hi=97.5):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return [np.nan, np.nan]
    return [float(np.percentile(v, lo)), float(np.percentile(v, hi))]

def main():
    ap = argparse.ArgumentParser(
        description="Same-cell cross-backbone specificity audit for SMS on UNSEG."
    )
    ap.add_argument(
        "--mesmer",
        type=Path,
        default=Path(r"E:\Histo_Pilot\histo10b_mean_sms\run_20261102_151526\unseg_cell_rows.csv"),
    )
    ap.add_argument(
        "--cellpose",
        type=Path,
        default=Path(r"E:\Histo_Pilot\sms_backbone_transfer\cellpose_evaluation\cell_rows.csv"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(r"E:\Histo_Pilot\sms_backbone_transfer\same_cell_cross_backbone"),
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    mes = load_rows(args.mesmer, "mesmer")
    cp = load_rows(args.cellpose, "cellpose")

    common = mes.merge(
        cp,
        on=["case_norm", "gt_label"],
        how="inner",
        validate="one_to_one",
    )
    if len(common) == 0:
        raise RuntimeError("No common manual cells found between Mesmer and Cellpose files.")

    # Preserve a readable case label.
    common["case"] = common["cellpose_case"].astype(str)

    # This >=10 criterion is inherited from the prior UNSEG per-image stability analysis.
    counts = common.groupby("case_norm", sort=True).size()
    eligible_cases = counts[counts >= MIN_COMMON_PER_IMAGE].index
    eval_df = common[common["case_norm"].isin(eligible_cases)].copy()

    rows = []
    for case_norm, g in eval_df.groupby("case_norm", sort=True):
        m = case_metrics(g)
        m["case_norm"] = case_norm
        m["case"] = str(g["case"].iloc[0])
        rows.append(m)
    case_df = pd.DataFrame(rows)

    metric_cols = [c for c in case_df.columns if c.startswith("rho_") or c.startswith("rr70_")]
    means = {c: mean_finite(case_df[c]) for c in metric_cols}

    # Specificity contrasts. Positive values support "own-boundary" specificity.
    case_df["delta_rho_for_mesmer_error"] = (
        case_df["rho_mesmer_sms__mesmer_error"]
        - case_df["rho_cellpose_sms__mesmer_error"]
    )
    case_df["delta_rho_for_cellpose_error"] = (
        case_df["rho_cellpose_sms__cellpose_error"]
        - case_df["rho_mesmer_sms__cellpose_error"]
    )
    case_df["delta_rr_advantage_for_mesmer_error"] = (
        case_df["rr70_cellpose_sms__mesmer_error"]
        - case_df["rr70_mesmer_sms__mesmer_error"]
    )
    case_df["delta_rr_advantage_for_cellpose_error"] = (
        case_df["rr70_mesmer_sms__cellpose_error"]
        - case_df["rr70_cellpose_sms__cellpose_error"]
    )

    contrast_cols = [
        "delta_rho_for_mesmer_error",
        "delta_rho_for_cellpose_error",
        "delta_rr_advantage_for_mesmer_error",
        "delta_rr_advantage_for_cellpose_error",
    ]
    contrasts = {c: mean_finite(case_df[c]) for c in contrast_cols}

    # Paired image bootstrap: resample eligible images, preserving all four pairings.
    rng = np.random.default_rng(SEED)
    n_cases = len(case_df)
    boot = {c: [] for c in metric_cols + contrast_cols}
    if n_cases:
        for _ in range(N_BOOT):
            idx = rng.integers(0, n_cases, n_cases)
            b = case_df.iloc[idx]
            for c in boot:
                boot[c].append(mean_finite(b[c]))

    bootstrap_ci = {c: ci(v) for c, v in boot.items()}

    # Pooled metrics are secondary diagnostics only.
    pooled = {
        "rho_mesmer_sms__mesmer_error": rho(eval_df["mesmer_sms"], eval_df["mesmer_error"]),
        "rho_cellpose_sms__cellpose_error": rho(eval_df["cellpose_sms"], eval_df["cellpose_error"]),
        "rho_mesmer_sms__cellpose_error": rho(eval_df["mesmer_sms"], eval_df["cellpose_error"]),
        "rho_cellpose_sms__mesmer_error": rho(eval_df["cellpose_sms"], eval_df["mesmer_error"]),
        "rho_sms_between_backbones": rho(eval_df["mesmer_sms"], eval_df["cellpose_sms"]),
        "rho_error_between_backbones": rho(eval_df["mesmer_error"], eval_df["cellpose_error"]),
    }

    # Descriptive support: how often diagonal associations/RR beat their crossed counterparts.
    desc = {
        "images_mesmer_own_rho_gt_cross": int(
            np.sum(case_df["delta_rho_for_mesmer_error"] > 0)
        ),
        "images_cellpose_own_rho_gt_cross": int(
            np.sum(case_df["delta_rho_for_cellpose_error"] > 0)
        ),
        "images_mesmer_own_rr_better_than_cross": int(
            np.sum(case_df["delta_rr_advantage_for_mesmer_error"] > 0)
        ),
        "images_cellpose_own_rr_better_than_cross": int(
            np.sum(case_df["delta_rr_advantage_for_cellpose_error"] > 0)
        ),
        "n_evaluable_images": int(n_cases),
    }

    summary = {
        "protocol": "Exploratory same-cell cross-backbone specificity audit of SMS on UNSEG",
        "status_note": (
            "This mechanism analysis was specified after the Cellpose-SAM backbone-transfer "
            "results were known; it is explanatory/exploratory, not a prespecified confirmatory test."
        ),
        "identity_key": ["case", "manual gt_label"],
        "min_common_cells_per_image": MIN_COMMON_PER_IMAGE,
        "coverage": COVERAGE,
        "bootstrap_resamples": N_BOOT,
        "bootstrap_unit": "image",
        "seed": SEED,
        "source_files": {
            "mesmer": str(args.mesmer),
            "cellpose": str(args.cellpose),
        },
        "counts": {
            "mesmer_rows": int(len(mes)),
            "cellpose_rows": int(len(cp)),
            "common_manual_cells_all": int(len(common)),
            "common_images_all": int(common["case_norm"].nunique()),
            "common_manual_cells_evaluable": int(len(eval_df)),
            "evaluable_images_ge10_common_cells": int(n_cases),
        },
        "image_balanced_means": means,
        "specificity_contrasts_positive_supports_own_boundary": contrasts,
        "bootstrap_95ci": bootstrap_ci,
        "pooled_secondary_diagnostics": pooled,
        "descriptive_image_counts": desc,
        "interpretation_rule": {
            "association": (
                "For each error source, own-boundary specificity is supported when the "
                "diagonal SMS-error Spearman exceeds the crossed SMS-error Spearman."
            ),
            "selective_risk": (
                "For each error source, own-boundary specificity is supported when the "
                "diagonal SMS ranking has lower RR70 than the crossed SMS ranking; the "
                "reported RR advantage is cross RR70 minus own RR70, so positive is favorable."
            ),
            "claim_limit": (
                "Because this audit was designed after observing the Cellpose transfer result, "
                "it can explain attenuation but cannot upgrade the failed prespecified "
                "backbone-transfer decision."
            ),
        },
    }

    common.to_csv(args.out / "common_cell_rows.csv", index=False)
    case_df.to_csv(args.out / "per_image_cross_backbone.csv", index=False)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def fmt(x):
        return "NA" if not np.isfinite(x) else f"{x:.4f}"

    lines = []
    lines.append("SMS SAME-CELL CROSS-BACKBONE SPECIFICITY AUDIT")
    lines.append("=" * 62)
    lines.append(summary["status_note"])
    lines.append("")
    lines.append(f"Mesmer rows: {len(mes)}")
    lines.append(f"Cellpose rows: {len(cp)}")
    lines.append(f"Common manual cells (all): {len(common)}")
    lines.append(f"Evaluable images (>=10 common cells): {n_cases}")
    lines.append(f"Common cells in evaluable images: {len(eval_df)}")
    lines.append("")
    lines.append("IMAGE-BALANCED 2x2 SPEARMAN MATRIX")
    lines.append("                         Error: Mesmer    Error: Cellpose")
    lines.append(
        f"SMS: Mesmer              {fmt(means['rho_mesmer_sms__mesmer_error']):>12}    "
        f"{fmt(means['rho_mesmer_sms__cellpose_error']):>15}"
    )
    lines.append(
        f"SMS: Cellpose            {fmt(means['rho_cellpose_sms__mesmer_error']):>12}    "
        f"{fmt(means['rho_cellpose_sms__cellpose_error']):>15}"
    )
    lines.append("")
    lines.append("IMAGE-BALANCED 2x2 RR70 MATRIX (LOWER IS BETTER)")
    lines.append("                         Error: Mesmer    Error: Cellpose")
    lines.append(
        f"SMS: Mesmer              {fmt(means['rr70_mesmer_sms__mesmer_error']):>12}    "
        f"{fmt(means['rr70_mesmer_sms__cellpose_error']):>15}"
    )
    lines.append(
        f"SMS: Cellpose            {fmt(means['rr70_cellpose_sms__mesmer_error']):>12}    "
        f"{fmt(means['rr70_cellpose_sms__cellpose_error']):>15}"
    )
    lines.append("")
    lines.append("SPECIFICITY CONTRASTS (POSITIVE SUPPORTS OWN-BOUNDARY SPECIFICITY)")
    for c in contrast_cols:
        lines.append(f"{c}: {fmt(contrasts[c])}  95% CI {bootstrap_ci[c]}")
    lines.append("")
    lines.append("DESCRIPTIVE IMAGE COUNTS")
    for k, v in desc.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("IMPORTANT: explanatory/post-hoc mechanism audit; do not relabel the")
    lines.append("prespecified Cellpose backbone-transfer decision as positive.")

    report = "\n".join(lines)
    (args.out / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print("\nSaved to:", args.out)

if __name__ == "__main__":
    main()