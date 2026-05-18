#!/usr/bin/env python3
"""
Generate Blood Advances-style prediction atlas figures from the REDS recalled donor spreadsheet.

Input:
    Lead - REDS Recalled.csv

Outputs:
    Figure 1: workflow + model benchmark + best model by phenotype
    Figure 2: observed-vs-predicted models + stable predictor network + bubble matrix
    Figure 3: stable predictor coefficients + predictor-class composition + performance summary
    Supplementary Figure 1: full-page hive plot
    Source tables: model performance, best models, feature stability, top full-model predictors

Example:
    python generate_BA_prediction_figures_from_REDS_recalled.py \
        --input "Lead - REDS Recalled.csv" \
        --outdir "BloodAdvances_prediction_outputs"

Notes:
    - All preprocessing is performed inside cross-validation folds: imputation, scaling,
      feature screening, model fitting, and prediction.
    - Main model is ridge regression because omics predictors are high-dimensional and correlated.
    - The script exports both PNG and SVG; SVG text is preserved as editable text.
"""

from __future__ import annotations

import argparse
import os
import re
import textwrap
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.family"] = "Arial"
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MplPath

from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Global visual style
# -----------------------------------------------------------------------------
PANEL_LETTER_SIZE = 10
PANEL_TITLE_SIZE = 8
BODY_SIZE = 6
SMALL_SIZE = 5.5
FIG_WIDTH_IN = 16 / 2.54  # 16 cm

DARK = "#111827"
MID = "#475569"
GRID = "#D1D5DB"
BLUE = "#3756c4"
GREEN = "#57b894"
PURPLE = "#7a4ae2"
AMBER = "#e3a33a"
RED = "#d55e5e"
TEAL = "#2b8cbe"
GREY = "#9aa5b1"
PINK = "#b14f98"

BLOCK_COLORS = {
    "Metadata": GREY,
    "CBC/Ferritin": RED,
    "Trace elements": TEAL,
    "Metabolomics": GREEN,
    "Lipidomics": AMBER,
    "Proteomics": PURPLE,
    "All omics": "#536878",
    "Full model": BLUE,
}
GROUP_COLORS = {
    "Metadata": GREY,
    "CBC/Ferritin": RED,
    "Trace elements": TEAL,
    "Metabolomics": GREEN,
    "Lipidomics": AMBER,
    "Proteomics": PURPLE,
}

OUTCOME_ORDER = [
    "Osmotic hemolysis",
    "Oxidative hemolysis",
    "RBC EV CD235+CD108-",
    "Storage hemolysis",
    "Platelet EV CD41+CD62p-",
    "Delta all EVs",
    "RBC EV CD235+CD108+",
    "All EVs",
]

SHORT_OUTCOME = {
    "Osmotic hemolysis": "Osmotic\nhemolysis",
    "Oxidative hemolysis": "Oxidative\nhemolysis",
    "RBC EV CD235+CD108-": "RBC EV\nCD235+CD108−",
    "Storage hemolysis": "Storage\nhemolysis",
    "Platelet EV CD41+CD62p-": "Platelet EV\nCD41+CD62p−",
    "Delta all EVs": "Δ all EVs",
    "RBC EV CD235+CD108+": "RBC EV\nCD235+CD108+",
    "All EVs": "All EVs",
}

BLOCK_ORDER = [
    "Metadata",
    "CBC/Ferritin",
    "Trace elements",
    "Metabolomics",
    "Lipidomics",
    "Proteomics",
    "All omics",
    "Full model",
]


# -----------------------------------------------------------------------------
# Input preprocessing and feature block definition
# -----------------------------------------------------------------------------
def read_recalled_spreadsheet(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read the spreadsheet and return raw dataframe plus day-42 donor-level dataframe."""
    df = pd.read_csv(path, low_memory=False)

    # Coerce anything that can be numeric. Categorical columns are left alone.
    categorical_cols = {
        "Sample",
        "CU ID",
        "Global ID",
        "RECALLED ID",
        "INDEX ID",
        "Recalled ID",
        "AS",
        "Gender",
        "DONDB.ABO_RH",
        "RBCOmics.Race.Ethnicity.Group",
    }
    for col in df.columns:
        if col not in categorical_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # One row per donor at day 42.
    donor = df[df["Day"] == 42].copy().sort_values("Recalled ID").drop_duplicates("Recalled ID")

    # Add Delta all EVs = day 42 - day 10 for All Events Count uL.
    if "All Events Count uL" in df.columns:
        pivot = df[df["Day"].isin([10, 42])].pivot_table(
            index="Recalled ID", columns="Day", values="All Events Count uL", aggfunc="first"
        )
        if 10 in pivot.columns and 42 in pivot.columns:
            donor = donor.merge(
                (pivot[42] - pivot[10]).rename("Delta All Events Count uL"),
                left_on="Recalled ID",
                right_index=True,
                how="left",
            )

    return df, donor


def get_feature_blocks(donor: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Build predictor blocks from the spreadsheet structure."""
    meta_cols = [
        c
        for c in ["AS", "Gender", "DONDB.ABO_RH", "RBCOmics.Race.Ethnicity.Group", "Age", "BMI", "Weight", "Height"]
        if c in donor.columns
    ]
    cbc_cols = [
        c
        for c in ["CBC.WBC", "CBC.RBC", "CBC.HGB", "CBC.HCT", "CBC.MCV", "CBC.RDW", "CBC.PLT", "Ferritin"]
        if c in donor.columns
    ]
    trace_cols = [
        c
        for c in [
            "Sodium",
            "Magnesium",
            "Potassium",
            "Calcium",
            "Chromium",
            "Manganese",
            "Iron",
            "Copper",
            "Zinc",
            "Selenium",
            "Lead",
        ]
        if c in donor.columns
    ]

    # These column windows match the submitted REDS recalled donor spreadsheet layout.
    # They are intentionally explicit because the file contains thousands of omics variables.
    all_cols = donor.columns.tolist()
    metabolomics_cols = all_cols[35:339]
    proteomics_cols = all_cols[364:2500]
    lipidomics_cols = all_cols[2500:]

    # Prevent leakage by excluding outcome and ID-like variables from all feature sets.
    outcome_cols = set(get_outcome_map().values()) | {"Delta All Events Count uL"}
    id_like = {"Recalled ID", "Day", "Sample", "CU ID", "Global ID", "RECALLED ID", "INDEX ID"}

    def clean_feature_list(cols: Iterable[str]) -> List[str]:
        cleaned = []
        seen = set()
        for c in cols:
            if c in donor.columns and c not in outcome_cols and c not in id_like and c not in seen:
                cleaned.append(c)
                seen.add(c)
        return cleaned

    meta_numeric = [c for c in meta_cols if c in ["Age", "BMI", "Weight", "Height"]]
    meta_cat = [c for c in meta_cols if c not in meta_numeric]
    X_meta = []
    if meta_numeric:
        X_meta.append(donor[meta_numeric].apply(pd.to_numeric, errors="coerce"))
    if meta_cat:
        X_meta.append(pd.get_dummies(donor[meta_cat].astype(str), dummy_na=True, drop_first=False))
    metadata = pd.concat(X_meta, axis=1) if X_meta else pd.DataFrame(index=donor.index)

    cbc = donor[clean_feature_list(cbc_cols)].apply(pd.to_numeric, errors="coerce")
    trace = donor[clean_feature_list(trace_cols)].apply(pd.to_numeric, errors="coerce")
    metab = donor[clean_feature_list(metabolomics_cols)].apply(pd.to_numeric, errors="coerce")
    prot = donor[clean_feature_list(proteomics_cols)].apply(pd.to_numeric, errors="coerce")
    lipids = donor[clean_feature_list(lipidomics_cols)].apply(pd.to_numeric, errors="coerce")

    full_model = pd.concat([metadata, cbc, trace, metab, lipids, prot], axis=1)
    all_omics = pd.concat([metab, lipids, prot], axis=1)

    return {
        "Metadata": metadata,
        "CBC/Ferritin": cbc,
        "Trace elements": trace,
        "Metabolomics": metab,
        "Lipidomics": lipids,
        "Proteomics": prot,
        "All omics": all_omics,
        "Full model": full_model,
    }


def get_outcome_map() -> Dict[str, str]:
    return {
        "Storage hemolysis": "Recall.Transfer.Storage.Hemolysis",
        "Osmotic hemolysis": "Recall.Transfer.Osmotic.Hemolysis",
        "Oxidative hemolysis": "Recall.Transfer.Oxidative.Hemolysis",
        "All EVs": "All Events Count uL",
        "RBC EV CD235+CD108+": "CD235a+CD108+ Count uL",
        "RBC EV CD235+CD108-": "CD235a+CD108- Count uL",
        "Platelet EV CD41+CD62p-": "CD41a+CD62p- Count uL",
        "Total EV log10": "log10 Total EV count ul",
        "Delta all EVs": "Delta All Events Count uL",
    }


def feature_group(feature: str, blocks: Dict[str, pd.DataFrame]) -> str:
    for group in ["Metadata", "CBC/Ferritin", "Trace elements", "Metabolomics", "Lipidomics", "Proteomics"]:
        if feature in blocks[group].columns:
            return group
    return "Other"


# -----------------------------------------------------------------------------
# Modeling utilities
# -----------------------------------------------------------------------------
def transform_outcome(y: pd.Series) -> Tuple[np.ndarray, str]:
    """Return transformed outcome and transform name."""
    yy = pd.to_numeric(y, errors="coerce").astype(float).to_numpy()
    finite = yy[np.isfinite(yy)]
    if len(finite) == 0:
        return yy, "raw"
    if np.nanmin(finite) >= 0 and pd.Series(finite).skew() > 1:
        return np.log1p(yy), "log1p"
    return yy, "raw"


def rank_features_fast(X: pd.DataFrame, y: np.ndarray, top_k: int) -> List[str]:
    """Rank features by absolute Pearson correlation after median imputation and z-scoring.

    This is used only inside the training fold. It is intentionally fast because the
    full model can contain thousands of predictors.
    """
    keep = X.notna().sum(axis=0) > 0
    X = X.loc[:, keep]
    if X.shape[1] == 0:
        return []

    X_imp = X.fillna(X.median(axis=0))
    variances = X_imp.var(axis=0)
    X_imp = X_imp.loc[:, variances > 0]
    if X_imp.shape[1] == 0:
        return []

    arr = X_imp.to_numpy(dtype=float)
    arr -= np.nanmean(arr, axis=0)
    denom_x = np.sqrt(np.nansum(arr * arr, axis=0))
    yy = y - np.nanmean(y)
    denom_y = np.sqrt(np.nansum(yy * yy))
    corr = np.abs(np.nansum(arr * yy[:, None], axis=0) / (denom_x * denom_y))
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(corr)[::-1]
    return X_imp.columns[order[: min(top_k, len(order))]].tolist()


def top_k_for_block(block: str) -> int:
    if block in {"All omics", "Full model"}:
        return 250
    if block in {"Metabolomics", "Lipidomics", "Proteomics"}:
        return 100
    return 10_000  # keep all for small clinical/metadata/trace blocks


def fit_cross_validated_ridge(
    X: pd.DataFrame,
    y: pd.Series,
    block: str,
    random_state: int = 42,
) -> Dict[str, object]:
    """Fit leakage-safe 5-fold cross-validated ridge regression and return metrics."""
    y_transformed, transform_name = transform_outcome(y)
    mask = np.isfinite(y_transformed) & (~X.isna().all(axis=1).to_numpy())
    X = X.loc[mask].apply(pd.to_numeric, errors="coerce")
    yy = y_transformed[mask]

    if len(yy) < 80 or X.shape[1] < 1:
        return {
            "n": len(yy),
            "features_tested": X.shape[1],
            "cv_pearson_r": np.nan,
            "cv_R2": np.nan,
            "cv_spearman_rho": np.nan,
            "y_transform": transform_name,
            "y_true": yy,
            "y_pred": np.full_like(yy, np.nan, dtype=float),
            "selected_records": pd.DataFrame(),
        }

    top_k = top_k_for_block(block)
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    preds = np.full(len(yy), np.nan)
    selected_records = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = yy[train_idx]

        selected = rank_features_fast(X_train, y_train, top_k=top_k)
        if not selected:
            continue

        X_train = X_train[selected]
        X_test = X_test[selected]

        pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 30), cv=5)),
            ]
        )
        pipe.fit(X_train, y_train)
        preds[test_idx] = pipe.predict(X_test)

        coefs = pipe.named_steps["ridge"].coef_
        for feat, coef in zip(selected, coefs):
            selected_records.append({"fold": fold, "feature": feat, "coef": float(coef)})

    ok = np.isfinite(preds)
    pearson = float(pearsonr(yy[ok], preds[ok]).statistic) if ok.sum() > 3 else np.nan
    r2 = float(r2_score(yy[ok], preds[ok])) if ok.sum() > 3 else np.nan
    # Spearman without importing scipy.stats.spearmanr: Pearson on ranks.
    sp = float(pd.Series(yy[ok]).rank().corr(pd.Series(preds[ok]).rank())) if ok.sum() > 3 else np.nan

    return {
        "n": int(ok.sum()),
        "features_tested": int(X.shape[1]),
        "cv_pearson_r": pearson,
        "cv_R2": r2,
        "cv_spearman_rho": sp,
        "y_transform": transform_name,
        "y_true": yy,
        "y_pred": preds,
        "selected_records": pd.DataFrame(selected_records),
    }


def run_prediction_atlas(donor: pd.DataFrame, blocks: Dict[str, pd.DataFrame], outdir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[Tuple[str, str], Dict[str, object]]]:
    """Run all outcome x feature-block models and save benchmark tables."""
    outcome_map = {k: v for k, v in get_outcome_map().items() if v in donor.columns}

    performance_rows = []
    prediction_store: Dict[Tuple[str, str], Dict[str, object]] = {}
    importance_rows = []

    for outcome, col in outcome_map.items():
        y = donor[col]
        for block, X in blocks.items():
            print(f"Modeling {outcome} ~ {block} ...")
            res = fit_cross_validated_ridge(X, y, block)
            prediction_store[(outcome, block)] = res
            performance_rows.append(
                {
                    "outcome": outcome,
                    "outcome_col": col,
                    "block": block,
                    "n": res["n"],
                    "features_tested": res["features_tested"],
                    "cv_pearson_r": res["cv_pearson_r"],
                    "cv_spearman_rho": res["cv_spearman_rho"],
                    "cv_R2": res["cv_R2"],
                    "y_transform": res["y_transform"],
                }
            )

            sel = res["selected_records"]
            if isinstance(sel, pd.DataFrame) and not sel.empty:
                summary = (
                    sel.groupby("feature")
                    .agg(
                        selection_frequency=("fold", lambda s: s.nunique() / 5),
                        mean_abs_coef=("coef", lambda x: np.mean(np.abs(x))),
                        mean_coef=("coef", "mean"),
                    )
                    .reset_index()
                )
                summary["outcome"] = outcome
                summary["block"] = block
                summary["feature_group"] = summary["feature"].apply(lambda f: feature_group(f, blocks))
                importance_rows.append(summary)

    performance = pd.DataFrame(performance_rows)
    performance.to_csv(outdir / "prediction_performance_by_outcome_and_feature_block.csv", index=False)

    best = (
        performance.sort_values(["outcome", "cv_pearson_r"], ascending=[True, False])
        .groupby("outcome", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best.to_csv(outdir / "best_prediction_model_per_outcome.csv", index=False)

    importance = pd.concat(importance_rows, ignore_index=True) if importance_rows else pd.DataFrame()
    if not importance.empty:
        importance["rank_score"] = importance["selection_frequency"] * importance["mean_abs_coef"]
    importance.to_csv(outdir / "model_feature_importance_stability.csv", index=False)

    # Top 25 features per outcome from the full model.
    top25 = (
        importance[importance["block"] == "Full model"]
        .sort_values(["outcome", "rank_score"], ascending=[True, False])
        .groupby("outcome")
        .head(25)
        .reset_index(drop=True)
    )
    top25.to_csv(outdir / "top25_full_model_features_per_outcome.csv", index=False)

    return performance, best, importance, prediction_store


# -----------------------------------------------------------------------------
# Figure helpers
# -----------------------------------------------------------------------------
def add_panel_letter(ax, letter: str) -> None:
    ax.text(-0.08, 1.04, letter, transform=ax.transAxes, fontsize=PANEL_LETTER_SIZE, fontweight="bold", va="bottom", ha="left")


def clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def wrap_text(s: str, width: int = 18) -> str:
    return "\n".join(textwrap.wrap(str(s), width=width))


# -----------------------------------------------------------------------------
# Figure 1
# -----------------------------------------------------------------------------
def workflow_panel(ax, donor_n: int) -> None:
    ax.set_axis_off()
    add_panel_letter(ax, "A")
    ax.set_title("Analytic workflow", loc="left", pad=4)

    items = [
        ("1", f"REDS recalled donor cohort\n{donor_n} day-42 samples", "#e9eff9"),
        ("2", "Predictor blocks\nmetadata, CBC/ferritin, trace elements,\nmetabolomics, lipidomics, proteomics", "#edf4ef"),
        ("3", "Storage phenotypes\nhemolysis + EV outcomes", "#f8efe0"),
        ("4", "5-fold cross-validation\nfeature screening + ridge models", "#f6e8f0"),
        ("5", "Outputs\nperformance summary and stable predictors", "#e8f0f8"),
    ]
    x = 0.07
    w = 0.82
    h = 0.105
    gap = 0.038
    ys = [0.82 - i * (h + gap) for i in range(len(items))]

    for (num, txt, fc), y in zip(items, ys):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02", facecolor=fc, edgecolor="#4f5b66", lw=0.8)
        ax.add_patch(box)
        circ = Circle((x + 0.05, y + h / 2), 0.032, facecolor="white", edgecolor="#4f5b66", lw=0.8)
        ax.add_patch(circ)
        ax.text(x + 0.05, y + h / 2, num, ha="center", va="center", fontsize=7, fontweight="bold")
        ax.text(x + 0.10, y + h / 2, txt, ha="left", va="center", fontsize=5.6)

    for i in range(len(ys) - 1):
        arrow = FancyArrowPatch((x + w / 2, ys[i] - 0.01), (x + w / 2, ys[i + 1] + h + 0.01), arrowstyle="-|>", mutation_scale=10, lw=1, color="#4f5b66")
        ax.add_patch(arrow)

    ax.text(
        0.07,
        0.045,
        "Performance metrics shown as cross-validated Pearson r. Models were screened within training folds to avoid leakage.",
        fontsize=5.3,
        color="#505050",
        va="bottom",
        wrap=True,
    )


def performance_heatmap(ax, performance: pd.DataFrame) -> None:
    add_panel_letter(ax, "B")
    ax.set_title("Benchmark across feature blocks", loc="left", pad=4)

    mat = (
        performance.pivot(index="outcome", columns="block", values="cv_pearson_r")
        .reindex(index=OUTCOME_ORDER, columns=BLOCK_ORDER)
    )
    cmap = LinearSegmentedColormap.from_list("ba", ["#f7fbff", "#deebf7", "#9ecae1", "#4292c6", "#08519c"])
    im = ax.imshow(mat.values, cmap=cmap, aspect="auto", vmin=0, vmax=0.65)

    ax.set_xticks(np.arange(len(BLOCK_ORDER)))
    ax.set_xticklabels(["Meta", "CBC/Fer", "Trace", "Metab", "Lipid", "Prot", "All omics", "Full"], rotation=40, ha="right")
    ax.set_yticks(np.arange(len(OUTCOME_ORDER)))
    ax.set_yticklabels([SHORT_OUTCOME[o] for o in OUTCOME_ORDER])

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.values[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.2, color=("white" if val > 0.38 else "#14304a"))

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Pearson r", fontsize=6)
    cb.ax.tick_params(labelsize=5)
    clean_axes(ax)


def best_block_bar(ax, best: pd.DataFrame) -> None:
    add_panel_letter(ax, "C")
    ax.set_title("Best block by phenotype", loc="left", pad=4)

    b = best.copy()
    b["outcome"] = pd.Categorical(b["outcome"], OUTCOME_ORDER, ordered=True)
    b = b.sort_values("outcome")
    ypos = np.arange(len(b))[::-1]
    colors = [BLOCK_COLORS.get(x, "#777777") for x in b["block"]]

    ax.barh(ypos, b["cv_pearson_r"], color=colors, edgecolor="none", height=0.68)
    ax.set_yticks(ypos)
    ax.set_yticklabels([SHORT_OUTCOME[str(o)] for o in b["outcome"]])
    ax.set_xlim(0, 0.68)
    ax.set_xlabel("Pearson r")

    for y, r, blk in zip(ypos, b["cv_pearson_r"], b["block"]):
        ax.text(r + 0.015, y, blk.replace(" model", "").replace(" elements", ""), va="center", fontsize=5.6, color="#333333")

    handles = [Line2D([0], [0], color=BLOCK_COLORS[k], lw=5) for k in ["Metabolomics", "Proteomics", "Full model"]]
    ax.legend(handles, ["Metabolomics", "Proteomics", "Full model"], frameon=False, fontsize=5.6, ncols=1, loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    clean_axes(ax)


def plot_figure1(performance: pd.DataFrame, best: pd.DataFrame, donor_n: int, outdir: Path) -> None:
    fig = plt.figure(figsize=(FIG_WIDTH_IN, 5.25), dpi=300)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[0.28, 0.42, 0.30], left=0.05, right=0.99, top=0.96, bottom=0.09, wspace=0.28)
    workflow_panel(fig.add_subplot(gs[0, 0]), donor_n=donor_n)
    performance_heatmap(fig.add_subplot(gs[0, 1]), performance)
    best_block_bar(fig.add_subplot(gs[0, 2]), best)
    fig.savefig(outdir / "Figure_1_BA_prediction_workflow_performance.png", dpi=300)
    fig.savefig(outdir / "Figure_1_BA_prediction_workflow_performance_text.svg")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 2
# -----------------------------------------------------------------------------
def make_prediction_subsets(best: pd.DataFrame, prediction_store: Dict[Tuple[str, str], Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    subset_outcomes = [
        "Osmotic hemolysis",
        "Oxidative hemolysis",
        "RBC EV CD235+CD108-",
        "Storage hemolysis",
        "Platelet EV CD41+CD62p-",
        "Delta all EVs",
    ]
    out = {}
    best_map = best.set_index("outcome")
    for outcome in subset_outcomes:
        if outcome not in best_map.index:
            continue
        block = best_map.loc[outcome, "block"]
        res = prediction_store[(outcome, block)]
        y = np.asarray(res["y_true"])
        p = np.asarray(res["y_pred"])
        ok = np.isfinite(y) & np.isfinite(p)
        r = float(pearsonr(y[ok], p[ok]).statistic) if ok.sum() > 3 else np.nan
        out[outcome] = {"y": y[ok], "pred": p[ok], "r": r, "r2": r**2 if np.isfinite(r) else np.nan, "block": block, "n": ok.sum()}
    return out


def scatter_grid(fig, subspec, pred_results: Dict[str, Dict[str, object]]) -> None:
    host = fig.add_subplot(subspec)
    host.set_axis_off()
    add_panel_letter(host, "A")

    outcomes = list(pred_results.keys())
    gs = GridSpecFromSubplotSpec(2, 3, subplot_spec=subspec, wspace=0.32, hspace=0.35)
    for idx, outcome in enumerate(outcomes):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        y = pred_results[outcome]["y"]
        p = pred_results[outcome]["pred"]
        ax.scatter(y, p, s=3.5, color="#6d8fb6", alpha=0.35, edgecolor="none")
        mn = min(np.nanmin(y), np.nanmin(p))
        mx = max(np.nanmax(y), np.nanmax(p))
        pad = 0.05 * (mx - mn if mx > mn else 1)
        ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], ls="--", lw=0.8, color="#7f7f7f")
        ax.set_xlim(mn - pad, mx + pad)
        ax.set_ylim(mn - pad, mx + pad)
        ax.set_title(f"{chr(65 + idx)}  {outcome}\nBest block: {pred_results[outcome]['block']}", fontsize=5.9, loc="left", pad=1.5)
        ax.text(
            0.03,
            0.97,
            f"r={pred_results[outcome]['r']:.2f}\nR²={pred_results[outcome]['r2']:.2f}\nn={pred_results[outcome]['n']}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=5.4,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
        )
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.grid(alpha=0.15)
        clean_axes(ax)


def get_shared_predictor_table(importance: pd.DataFrame, outcomes: List[str]) -> pd.DataFrame:
    shared = importance[importance["outcome"].isin(outcomes)].copy()
    shared["score"] = shared["selection_frequency"] * shared["mean_abs_coef"].abs()
    return shared


def network_panel(ax, importance: pd.DataFrame, outcomes: List[str]) -> None:
    add_panel_letter(ax, "B")
    ax.set_title("Stable predictors linked to phenotypes", loc="left", pad=4)
    ax.set_axis_off()

    shared = get_shared_predictor_table(importance, outcomes)
    recurrent = (
        shared.groupby("feature")
        .agg(n_outcomes=("outcome", "nunique"), score=("score", "mean"))
        .reset_index()
        .sort_values(["n_outcomes", "score"], ascending=[False, False])
    )
    center_nodes = recurrent.head(8)["feature"].tolist()

    left_nodes = ["Metadata", "CBC/Ferritin", "Trace elements", "Metabolomics", "Lipidomics", "Proteomics"]
    xl, xc, xr = 0.08, 0.50, 0.90
    yl = np.linspace(0.85, 0.15, len(left_nodes))
    yc = np.linspace(0.88, 0.12, len(center_nodes))
    yr = np.linspace(0.85, 0.15, len(outcomes))
    left_pos = {n: (xl, y) for n, y in zip(left_nodes, yl)}
    center_pos = {n: (xc, y) for n, y in zip(center_nodes, yc)}
    right_pos = {n: (xr, y) for n, y in zip(outcomes, yr)}

    center_df = shared[shared["feature"].isin(center_nodes) & shared["outcome"].isin(outcomes)].copy()
    feature_group = center_df.groupby("feature")["feature_group"].agg(lambda s: s.value_counts().index[0]).to_dict()

    def curve(p0, p1, color, lw=1.0, alpha=0.6):
        verts = [p0, ((p0[0] + p1[0]) * 0.55, p0[1]), ((p0[0] + p1[0]) * 0.75, p1[1]), p1]
        path = MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=color, lw=lw, alpha=alpha))

    for feat in center_nodes:
        group = feature_group.get(feat, "Proteomics")
        curve(left_pos[group], center_pos[feat], GROUP_COLORS.get(group, "#777777"), lw=1.2)

    for _, row in center_df.iterrows():
        if row["selection_frequency"] < 0.6:
            continue
        curve(center_pos[row["feature"]], right_pos[row["outcome"]], GROUP_COLORS.get(row["feature_group"], "#888888"), lw=0.5 + 1.6 * min(1.0, row["selection_frequency"]), alpha=0.55)

    for n, (x, y) in left_pos.items():
        ax.text(x - 0.01, y, n, ha="right", va="center", fontsize=5.8, color=GROUP_COLORS[n], fontweight="bold")
    for n, (x, y) in center_pos.items():
        ax.scatter([x], [y], s=22, facecolor="white", edgecolor="#4f4f4f", zorder=3, lw=0.8)
        ax.text(x, y + 0.028, wrap_text(n, 14), ha="center", va="bottom", fontsize=5.2)
    for n, (x, y) in right_pos.items():
        ax.text(x + 0.01, y, SHORT_OUTCOME[n].replace("\n", " "), ha="left", va="center", fontsize=5.7, color=PINK, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def bubble_panel(ax, importance: pd.DataFrame, outcomes: List[str]) -> None:
    add_panel_letter(ax, "C")
    ax.set_title("Shared stable predictors", loc="left", pad=4)

    shared = get_shared_predictor_table(importance, outcomes)
    recurrent = (
        shared.groupby("feature")
        .agg(n_outcomes=("outcome", "nunique"), score=("score", "mean"))
        .reset_index()
        .sort_values(["n_outcomes", "score"], ascending=[False, False])
    )
    features = recurrent.head(9)["feature"].tolist()

    rows = []
    for feature in features:
        for outcome in outcomes:
            tmp = shared[(shared["feature"] == feature) & (shared["outcome"] == outcome)]
            if len(tmp):
                r = tmp.iloc[0]
                rows.append({"feature": feature, "outcome": outcome, "sel": r["selection_frequency"], "coef": r["mean_coef"]})
            else:
                rows.append({"feature": feature, "outcome": outcome, "sel": 0, "coef": 0})
    d = pd.DataFrame(rows)
    x_map = {o: i for i, o in enumerate(outcomes)}
    y_map = {f: i for i, f in enumerate(features[::-1])}

    vmax = max(1e-9, np.nanpercentile(np.abs(d["coef"]), 95))
    sc = ax.scatter(
        d["outcome"].map(x_map),
        d["feature"].map(y_map),
        s=20 + 140 * d["sel"],
        c=d["coef"],
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        edgecolor="#4f4f4f",
        lw=0.4,
    )
    ax.set_xticks(range(len(outcomes)))
    ax.set_xticklabels([SHORT_OUTCOME[o].replace("\n", " ") for o in outcomes], rotation=30, ha="right")
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features[::-1])
    ax.grid(alpha=0.18)
    cb = plt.colorbar(sc, ax=ax, fraction=0.050, pad=0.02)
    cb.set_label("Mean coefficient", fontsize=6)
    cb.ax.tick_params(labelsize=5)
    ax.text(0.98, 1.02, "Bubble size = selection frequency", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5, color="#555555")
    clean_axes(ax)


def plot_figure2(best: pd.DataFrame, importance: pd.DataFrame, prediction_store: Dict[Tuple[str, str], Dict[str, object]], outdir: Path) -> None:
    pred_results = make_prediction_subsets(best, prediction_store)
    outcomes = list(pred_results.keys())
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN), dpi=300)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[0.56, 0.44], width_ratios=[0.46, 0.54], left=0.06, right=0.99, top=0.97, bottom=0.07, hspace=0.25, wspace=0.22)
    scatter_grid(fig, gs[0, :], pred_results)
    network_panel(fig.add_subplot(gs[1, 0]), importance, outcomes)
    bubble_panel(fig.add_subplot(gs[1, 1]), importance, outcomes)
    fig.savefig(outdir / "Figure_2_BA_prediction_accuracy_hive.png", dpi=300)
    fig.savefig(outdir / "Figure_2_BA_prediction_accuracy_hive_text.svg")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 3
# -----------------------------------------------------------------------------
def figure3_panel_a(fig, subspec, importance: pd.DataFrame) -> None:
    host = fig.add_subplot(subspec)
    host.set_axis_off()
    add_panel_letter(host, "A")

    outcomes = ["Storage hemolysis", "Osmotic hemolysis", "Oxidative hemolysis", "All EVs", "RBC EV CD235+CD108-", "Platelet EV CD41+CD62p-"]
    full = importance[importance["block"] == "Full model"].copy()
    full["rank_score"] = full["selection_frequency"] * full["mean_abs_coef"]

    gs = GridSpecFromSubplotSpec(3, 2, subplot_spec=subspec, wspace=0.36, hspace=0.55)
    for idx, outcome in enumerate(outcomes):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        dd = full[full["outcome"] == outcome].sort_values("rank_score", ascending=False).head(6).iloc[::-1]
        colors = [GROUP_COLORS.get(g, "#777777") for g in dd["feature_group"]]
        ax.barh(np.arange(len(dd)), dd["mean_coef"], color=colors, edgecolor="none")
        ax.axvline(0, color="#777777", lw=0.7)
        ax.set_yticks(np.arange(len(dd)))
        ax.set_yticklabels([wrap_text(f, 18) for f in dd["feature"]])
        ax.set_title(f"{chr(65 + idx)}  {outcome}", fontsize=6.2, loc="left", pad=2)
        ax.set_xlabel("Mean ridge coefficient")
        ax.grid(axis="x", alpha=0.18)
        clean_axes(ax)


def figure3_panel_b(ax, importance: pd.DataFrame) -> None:
    add_panel_letter(ax, "B")
    ax.set_title("Predictor-class composition", loc="left", pad=4)

    outcomes = ["Delta all EVs", "Platelet EV CD41+CD62p-", "Storage hemolysis", "RBC EV CD235+CD108-", "Oxidative hemolysis", "Osmotic hemolysis"]
    full = importance[importance["block"] == "Full model"].copy()
    full["rank_score"] = full["selection_frequency"] * full["mean_abs_coef"]

    ypos = np.arange(len(outcomes))[::-1]
    left = np.zeros(len(outcomes))
    for group in ["Metabolomics", "Lipidomics", "Proteomics", "CBC/Ferritin", "Trace elements", "Metadata"]:
        counts = []
        for outcome in outcomes:
            dd = full[full["outcome"] == outcome].sort_values("rank_score", ascending=False).head(15)
            counts.append((dd["feature_group"] == group).sum())
        counts = np.asarray(counts)
        ax.barh(ypos, counts, left=left, color=GROUP_COLORS[group], edgecolor="none", height=0.72, label=group)
        left += counts

    ax.set_yticks(ypos)
    ax.set_yticklabels([SHORT_OUTCOME[o] for o in outcomes])
    ax.set_xlabel("Count among top 15 stable predictors")
    ax.legend(frameon=False, fontsize=5.2, ncols=3, loc="lower right", handlelength=1.2, columnspacing=0.8)
    ax.grid(axis="x", alpha=0.18)
    clean_axes(ax)


def figure3_panel_c(ax, best: pd.DataFrame) -> None:
    add_panel_letter(ax, "C")
    ax.set_title("Best-model performance", loc="left", pad=4)

    keep = ["Osmotic hemolysis", "Oxidative hemolysis", "RBC EV CD235+CD108-", "Storage hemolysis", "Platelet EV CD41+CD62p-", "Delta all EVs", "All EVs"]
    b = best[best["outcome"].isin(keep)].copy()
    order = keep[::-1]
    b["outcome"] = pd.Categorical(b["outcome"], order, ordered=True)
    b = b.sort_values("outcome")
    ypos = np.arange(len(b))
    sizes = 50 + 160 * np.maximum(b["cv_R2"], 0)
    colors = [BLOCK_COLORS.get(x, "#777777") for x in b["block"]]

    ax.scatter(b["cv_pearson_r"], ypos, s=sizes, c=colors, edgecolor="white", lw=0.7)
    ax.set_yticks(ypos)
    ax.set_yticklabels([SHORT_OUTCOME[str(o)].replace("\n", " ") for o in b["outcome"]])
    ax.set_xlabel("Cross-validated Pearson r")
    ax.set_xlim(0, 0.66)

    for x, y, r2, n in zip(b["cv_pearson_r"], ypos, b["cv_R2"], b["n"]):
        ax.text(x + 0.02, y, f"R²={r2:.2f}; n={int(n)}", va="center", fontsize=5.5)

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=BLOCK_COLORS[k], markersize=6) for k in ["Metabolomics", "Proteomics", "Full model"]]
    ax.legend(handles, ["Metabolomics", "Proteomics", "Full model"], frameon=False, fontsize=5.2, loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    clean_axes(ax)


def plot_figure3(best: pd.DataFrame, importance: pd.DataFrame, outdir: Path) -> None:
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN), dpi=300)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[0.56, 0.44], height_ratios=[0.58, 0.42], left=0.05, right=0.99, top=0.97, bottom=0.07, wspace=0.24, hspace=0.26)
    figure3_panel_a(fig, gs[:, 0], importance)
    figure3_panel_b(fig.add_subplot(gs[0, 1]), importance)
    figure3_panel_c(fig.add_subplot(gs[1, 1]), best)
    fig.savefig(outdir / "Figure_3_BA_predictor_architecture.png", dpi=300)
    fig.savefig(outdir / "Figure_3_BA_predictor_architecture_text.svg")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Supplementary full-page hive plot
# -----------------------------------------------------------------------------
def plot_supplementary_hive(best: pd.DataFrame, importance: pd.DataFrame, outdir: Path) -> None:
    fig = plt.figure(figsize=(8.5, 11), dpi=300)
    ax = fig.add_axes([0.06, 0.06, 0.88, 0.88])
    ax.axis("off")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.text(-1.10, 1.08, "Supplementary Figure 1. Global hive plot of stable predictors", fontsize=12, fontweight="bold", color=DARK, va="top")
    ax.text(-1.10, 1.00, "Feature classes are shown on the left, phenotypes on the right, and recurrent predictors in the center.", fontsize=8, color=MID, va="top")

    outcomes = best.sort_values("cv_pearson_r", ascending=False)["outcome"].tolist()
    groups = ["Metadata", "CBC/Ferritin", "Trace elements", "Metabolomics", "Lipidomics", "Proteomics"]
    gy = np.linspace(0.88, -0.88, len(groups))
    oy = np.linspace(0.88, -0.88, len(outcomes))
    gpos = {g: (-0.98, y) for g, y in zip(groups, gy)}
    opos = {o: (0.98, y) for o, y in zip(outcomes, oy)}

    use = importance.copy()
    use["score"] = use["selection_frequency"] * use["mean_abs_coef"]
    top_features = (
        use.groupby("feature")
        .agg(n_out=("outcome", "nunique"), max_score=("score", "max"))
        .sort_values(["n_out", "max_score"], ascending=[False, False])
        .head(24)
        .index.tolist()
    )
    fy = np.linspace(0.92, -0.92, len(top_features))
    fpos = {f: (0.0, y) for f, y in zip(top_features, fy)}

    edges = []
    for outcome in outcomes:
        sub = use[use["outcome"] == outcome].sort_values("score", ascending=False)
        count = 0
        for _, row in sub.iterrows():
            if row["feature"] in fpos and count < 5:
                edges.append((row["feature_group"], outcome, row["feature"], row["selection_frequency"]))
                count += 1

    def curve(p0, p1, color, lw=1.0, alpha=0.65):
        x0, y0 = p0
        x1, y1 = p1
        path = MplPath([p0, (x0 + (x1 - x0) * 0.35, y0), (x0 + (x1 - x0) * 0.65, y1), p1], [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=color, lw=lw, alpha=alpha))

    for g, (x, y) in gpos.items():
        ax.add_patch(Circle((x, y), 0.016, fc=GROUP_COLORS[g], ec="none"))
        ax.text(x + 0.035, y, g, fontsize=9, va="center", ha="left", color=DARK, fontweight="bold")
    for outcome, (x, y) in opos.items():
        ax.add_patch(Circle((x, y), 0.013, fc=PINK, ec="none"))
        ax.text(x - 0.035, y, SHORT_OUTCOME.get(outcome, outcome).replace("\n", " "), fontsize=8, va="center", ha="right", color=DARK)
    for feature, (x, y) in fpos.items():
        ax.add_patch(Circle((x, y), 0.012, fc="white", ec=DARK, lw=0.7))
        ax.text(x, y + 0.022, textwrap.shorten(feature, width=26, placeholder="…"), fontsize=6.8, ha="center", va="bottom", color=DARK)

    for group, outcome, feature, sel in edges:
        color = GROUP_COLORS.get(group, GREY)
        curve(gpos[group], fpos[feature], color, lw=0.6 + 1.0 * sel)
        curve(fpos[feature], opos[outcome], color, lw=0.6 + 1.0 * sel)

    ax.text(-1.10, -1.05, "Color legend:", fontsize=8, fontweight="bold", color=DARK, va="center")
    x = -0.83
    for name, col in GROUP_COLORS.items():
        ax.plot([x, x + 0.05], [-1.05, -1.05], color=col, lw=2)
        ax.text(x + 0.06, -1.05, name.replace("CBC/Ferritin", "CBC/Fer"), fontsize=7.5, va="center", color=DARK)
        x += 0.28

    fig.savefig(outdir / "Supplementary_Figure_1_fullpage_hive.png", dpi=300)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Review PDF and package
# -----------------------------------------------------------------------------
def make_review_pdf(outdir: Path) -> None:
    pdf_path = outdir / "BloodAdvances_prediction_figures_review.pdf"
    with PdfPages(pdf_path) as pdf:
        for fn in [
            "Figure_1_BA_prediction_workflow_performance.png",
            "Figure_2_BA_prediction_accuracy_hive.png",
            "Figure_3_BA_predictor_architecture.png",
            "Supplementary_Figure_1_fullpage_hive.png",
        ]:
            img = plt.imread(outdir / fn)
            fig = plt.figure(figsize=(8.27, 8.27))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig, dpi=300)
            plt.close(fig)


def make_zip(outdir: Path) -> Path:
    zip_path = outdir.parent / f"{outdir.name}_package.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fp in outdir.rglob("*"):
            if fp.is_file():
                z.write(fp, arcname=str(fp.relative_to(outdir.parent)))
    return zip_path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate REDS prediction atlas figures from recalled donor spreadsheet.")
    parser.add_argument("--input", required=True, type=Path, help="Path to Lead - REDS Recalled.csv or equivalent submitted spreadsheet.")
    parser.add_argument("--outdir", default=Path("BloodAdvances_prediction_outputs"), type=Path, help="Output directory.")
    parser.add_argument("--skip-modeling", action="store_true", help="Reuse existing CSV outputs in --outdir if present.")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df, donor = read_recalled_spreadsheet(args.input)
    blocks = get_feature_blocks(donor)

    if args.skip_modeling and (args.outdir / "prediction_performance_by_outcome_and_feature_block.csv").exists():
        performance = pd.read_csv(args.outdir / "prediction_performance_by_outcome_and_feature_block.csv")
        best = pd.read_csv(args.outdir / "best_prediction_model_per_outcome.csv")
        importance = pd.read_csv(args.outdir / "model_feature_importance_stability.csv")
        raise RuntimeError("--skip-modeling currently cannot recreate observed-vs-predicted panels because prediction_store is not serialized. Run without --skip-modeling.")
    else:
        performance, best, importance, prediction_store = run_prediction_atlas(donor, blocks, args.outdir)

    donor_n = donor["Recalled ID"].nunique() if "Recalled ID" in donor.columns else donor.shape[0]
    plot_figure1(performance, best, donor_n, args.outdir)
    plot_figure2(best, importance, prediction_store, args.outdir)
    plot_figure3(best, importance, args.outdir)
    plot_supplementary_hive(best, importance, args.outdir)
    make_review_pdf(args.outdir)
    zip_path = make_zip(args.outdir)

    print("\nDone. Outputs written to:")
    print(args.outdir.resolve())
    print("Package:", zip_path.resolve())


if __name__ == "__main__":
    main()
