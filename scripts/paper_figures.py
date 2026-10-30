#!/usr/bin/env python
"""Publication figures for the EXP-002 crossover paper.

Reads the committed run directory (results/analysis.json, results/windows.json)
and writes vector PDFs into paper/figures/. Every number in a figure comes
from the analysis output; nothing is recomputed here except unit conversion.

    python scripts/paper_figures.py --run experiments/exp_002_lowrank-crossover/rtx4090-laptop/20260915T180708 --tag laptop
    python scripts/paper_figures.py --run experiments/exp_002_lowrank-crossover/a100-sxm4-40gb/<run-id> --tag a100

With --second-run the combined two-platform figures used in the paper body are
written as well (crossover_map_both, ratio_b1_both, heldout_scatter_row,
heldout_scatter_both).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BATCHES = [1, 4, 16, 64, 256, 1024, 4096]
SHAPES = [512, 1024, 2048, 4096]
FRACTIONS = [0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.375, 0.5, 0.75]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLORS = {512: "#1b6ca8", 1024: "#d1495b", 2048: "#2a9d8f", 4096: "#7b4b94"}
COLW = 3.25   # one MLSys column, inches
FULLW = 6.75  # two columns


def load(run: Path):
    analysis = json.loads((run / "results" / "analysis.json").read_text())
    windows = json.loads((run / "results" / "windows.json").read_text())
    return analysis, windows


def crossover_value(rec):
    """Encode a crossover record as a fraction of d: numeric, 0 for dense
    always, nan for factorized always (never observed)."""
    if rec is None:
        return np.nan
    if rec.get("kind") == "numeric":
        return rec["r_star"] / rec["d"]
    if rec.get("kind") == "none_dense":
        return 0.0
    return np.nan


def fig_crossover_map(a, out, tag):
    cal = set(a["calibration_shapes"])
    fig, axes = plt.subplots(2, 2, figsize=(FULLW, 3.3), sharex=True, sharey=True)
    for ax, d in zip(axes.ravel(), SHAPES):
        meas, lo, hi, pred = [], [], [], []
        for B in BATCHES:
            key = f"d{d}_B{B}"
            m = a["measured_crossovers"].get(key)
            meas.append(crossover_value(m))
            cell = a["coverage"]["cells"].get(key)
            if cell and cell.get("measured") == "numeric" and cell.get("measured_interval"):
                lo.append(cell["measured_interval"][0] / d)
                hi.append(cell["measured_interval"][1] / d)
            else:
                lo.append(np.nan)
                hi.append(np.nan)
            pred.append(crossover_value(a["predicted_crossovers"]["M1"].get(key)))
        meas, lo, hi, pred = map(np.array, (meas, lo, hi, pred))
        c = COLORS[d]
        ax.axhline(0.5, color="0.35", ls="--", lw=0.9, label="FLOPs rule ($r^* = d/2$)")
        ax.plot(BATCHES, pred, color=c, ls=":", marker="x", label="model " + ("(fit)" if d in cal else "(predicted)"))
        yerr = np.vstack([np.nan_to_num(meas - lo), np.nan_to_num(hi - meas)])
        ax.errorbar(BATCHES, meas, yerr=yerr, color=c, marker="o", ls="-", capsize=2, label="measured")
        for B, v in zip(BATCHES, meas):
            if v == 0.0:
                ax.annotate("dense\nalways", (B, 0.0), textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=6, color=c)
        ax.set_xscale("log", base=2)
        ax.set_xticks(BATCHES)
        ax.set_xticklabels([str(b) for b in BATCHES])
        ax.set_ylim(-0.03, 0.62)
        ax.set_title(f"$d = {d}$" + ("  (calibration)" if d in cal else "  (held out)"), loc="left")
        ax.grid(alpha=0.25)
    axes[1, 0].set_xlabel("batch size $B$")
    axes[1, 1].set_xlabel("batch size $B$")
    axes[0, 0].set_ylabel("crossover rank $r^*/d$")
    axes[1, 0].set_ylabel("crossover rank $r^*/d$")
    axes[1, 1].legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"crossover_map_{tag}.pdf")
    plt.close(fig)


def fig_ratio_vs_rank(a, out, tag):
    fig, axes = plt.subplots(1, 2, figsize=(FULLW, 2.3))
    for ax, B in zip(axes, (1, 4096)):
        f_axis = np.array(FRACTIONS)
        ax.plot(f_axis, 2 * f_axis, color="0.35", ls="--", lw=0.9, label="FLOPs ($2r/d$)")
        for d in SHAPES:
            rec = a["measured_crossovers"].get(f"d{d}_B{B}")
            if not rec:
                continue
            fr = np.array(rec["ranks"]) / d
            ax.plot(fr, rec["ratios"], color=COLORS[d], marker="o", label=f"$d = {d}$")
        ax.axhline(1.0, color="k", lw=0.6)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("rank fraction $r/d$")
        ax.set_title(f"$B = {B}$", loc="left")
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("energy ratio $E_{\\mathrm{fact}} / E_{\\mathrm{dense}}$")
    axes[0].legend(frameon=False, loc="upper left", ncol=1)
    fig.tight_layout()
    fig.savefig(out / f"ratio_vs_rank_{tag}.pdf")
    plt.close(fig)


def fig_heldout_scatter(a, out, tag):
    fig, axes = plt.subplots(1, 2, figsize=(FULLW, 2.5), sharex=True, sharey=True)
    for ax, model in zip(axes, ("P0", "M1")):
        preds = a["predicted_energy_by_key"][model]
        xs, ys, cs = [], [], []
        for key, pred in preds.items():
            d = int(key.split("_")[0][1:])
            if d not in a["heldout_shapes"]:
                continue
            meas = a["rep_stats"][key]["mean"]
            xs.append(meas)
            ys.append(pred if not isinstance(pred, dict) else pred.get("mean", np.nan))
            cs.append(COLORS[d])
        xs, ys = np.array(xs), np.array(ys)
        lim = (min(xs.min(), ys.min()) * 0.7, max(xs.max(), ys.max()) * 1.4)
        ax.plot(lim, lim, color="k", lw=0.6)
        ax.fill_between(lim, np.array(lim) / 1.5, np.array(lim) * 1.5, color="0.85", lw=0, label="factor 1.5")
        ax.scatter(xs, ys, c=cs, s=9, alpha=0.8, edgecolors="none")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        med = a["heldout_prediction"][model]["median_ape"]
        name = "FLOPs" if model == "P0" else "additive model"
        ax.set_title(f"{name}: held-out median error {100 * med:.1f}%", loc="left")
        ax.set_xlabel("measured energy per call (J)")
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("predicted energy per call (J)")
    for d in a["heldout_shapes"]:
        axes[1].scatter([], [], c=COLORS[d], s=12, label=f"$d = {d}$")
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / f"heldout_scatter_{tag}.pdf")
    plt.close(fig)


def fig_regimes(windows, out, tag, f_pick=0.0625):
    """Per-call time and mean power at B = 1 versus d, dense and factorized at
    one rank fraction: the dispatch floor, the memory-bound rise, and the
    power that each regime draws."""
    import collections
    acc = collections.defaultdict(list)
    for w in windows:
        if w["B"] != 1 or abs(w["f"] - f_pick) > 1e-9:
            continue
        acc[(w["d"], w["realization"])].append((w["per_call_s"] * 1e6, w["mean_power_w"], w["energy_per_call_j"] * 1e3))
    fig, axes = plt.subplots(1, 3, figsize=(FULLW, 2.0))
    for real, mk, ls in (("dense", "o", "-"), ("factorized", "s", "--")):
        t = [np.mean([x[0] for x in acc[(d, real)]]) for d in SHAPES]
        p = [np.mean([x[1] for x in acc[(d, real)]]) for d in SHAPES]
        e = [np.mean([x[2] for x in acc[(d, real)]]) for d in SHAPES]
        axes[0].plot(SHAPES, t, marker=mk, ls=ls, color="k", label=real)
        axes[1].plot(SHAPES, p, marker=mk, ls=ls, color="k", label=real)
        axes[2].plot(SHAPES, e, marker=mk, ls=ls, color="k", label=real)
    for ax, lab in zip(axes, ("time per call ($\\mu$s)", "mean power (W)", "energy per call (mJ)")):
        ax.set_xscale("log", base=2)
        ax.set_xticks(SHAPES)
        ax.set_xticklabels([str(d) for d in SHAPES])
        ax.set_xlabel("map size $d$ ($B = 1$, $r/d = 1/16$)")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.25, which="both")
    axes[0].set_yscale("log")
    axes[0].set_yticks([10, 20, 50, 100, 200])
    axes[0].set_yticklabels(["10", "20", "50", "100", "200"])
    axes[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[2].set_yscale("log")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / f"regimes_{tag}.pdf")
    plt.close(fig)


def fig_ratio_b1_both(runs, out):
    """Energy ratio against rank fraction at B = 1 on both platforms, one panel each."""
    fig, axes = plt.subplots(1, len(runs), figsize=(FULLW, 2.1), sharey=True)
    for ax, (label, a) in zip(axes, runs):
        f_axis = np.array(FRACTIONS)
        ax.plot(f_axis, 2 * f_axis, color="0.35", ls="--", lw=0.9, label="FLOPs ($2r/d$)")
        for d in SHAPES:
            rec = a["measured_crossovers"].get(f"d{d}_B1")
            if not rec:
                continue
            ax.plot(np.array(rec["ranks"]) / d, rec["ratios"], color=COLORS[d], marker="o", label=f"$d = {d}$")
        ax.axhline(1.0, color="k", lw=0.6)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("rank fraction $r/d$")
        ax.set_title(f"{label}, $B = 1$", loc="left")
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("energy ratio $E_{\\mathrm{fact}} / E_{\\mathrm{dense}}$")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "ratio_b1_both.pdf")
    plt.close(fig)


def fig_heldout_scatter_both(runs, out):
    """Predicted against measured held-out energy, FLOPs and M1, both platforms (2 x 2)."""
    fig, axes = plt.subplots(2, 2, figsize=(FULLW, 3.9))
    for row, (label, a) in zip(axes, runs):
        for ax, model in zip(row, ("P0", "M1")):
            preds = a["predicted_energy_by_key"][model]
            xs, ys, cs = [], [], []
            for key, pred in preds.items():
                d = int(key.split("_")[0][1:])
                if d not in a["heldout_shapes"]:
                    continue
                xs.append(a["rep_stats"][key]["mean"])
                ys.append(pred if not isinstance(pred, dict) else pred.get("mean", np.nan))
                cs.append(COLORS[d])
            xs, ys = np.array(xs), np.array(ys)
            lim = (xs.min() * 0.5, xs.max() * 1.6)   # axis range from the measurements; predictions below it sit on the edge
            ys_plot = np.clip(ys, lim[0] * 1.08, lim[1] / 1.08)
            ax.plot(lim, lim, color="k", lw=0.6)
            ax.fill_between(lim, np.array(lim) / 1.5, np.array(lim) * 1.5, color="0.85", lw=0, label="factor 1.5")
            ax.scatter(xs, ys_plot, c=cs, s=9, alpha=0.8, edgecolors="none")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            med = a["heldout_prediction"][model]["median_ape"]
            name = "FLOPs" if model == "P0" else "additive model"
            ax.set_title(f"{label}, {name}: median error {100 * med:.1f}%", loc="left")
            ax.grid(alpha=0.25, which="both")
        row[0].set_ylabel("predicted energy per call (J)")
    for ax in axes[1]:
        ax.set_xlabel("measured energy per call (J)")
    for d in runs[0][1]["heldout_shapes"]:
        axes[0][1].scatter([], [], c=COLORS[d], s=12, label=f"$d = {d}$")
    axes[0][1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "heldout_scatter_both.pdf")
    plt.close(fig)


def fig_crossover_map_both(runs, out):
    """Crossover maps of both platforms in one figure: one row per platform, one column per d."""
    fig, axes = plt.subplots(len(runs), 4, figsize=(FULLW, 1.75 * len(runs)), sharex=True, sharey=True)
    for row, (label, a) in zip(axes, runs):
        cal = set(a["calibration_shapes"])
        for ax, d in zip(row, SHAPES):
            meas, lo, hi, pred = [], [], [], []
            for B in BATCHES:
                key = f"d{d}_B{B}"
                meas.append(crossover_value(a["measured_crossovers"].get(key)))
                cell = a["coverage"]["cells"].get(key)
                if cell and cell.get("measured") == "numeric" and cell.get("measured_interval"):
                    lo.append(cell["measured_interval"][0] / d)
                    hi.append(cell["measured_interval"][1] / d)
                else:
                    lo.append(np.nan)
                    hi.append(np.nan)
                pred.append(crossover_value(a["predicted_crossovers"]["M1"].get(key)))
            meas, lo, hi, pred = map(np.array, (meas, lo, hi, pred))
            c = COLORS[d]
            ax.axhline(0.5, color="0.35", ls="--", lw=0.9, label="FLOPs rule")
            ax.plot(BATCHES, pred, color=c, ls=":", marker="x", label="model " + ("(fit)" if d in cal else "(predicted)"))
            yerr = np.vstack([np.nan_to_num(meas - lo), np.nan_to_num(hi - meas)])
            ax.errorbar(BATCHES, meas, yerr=yerr, color=c, marker="o", ls="-", capsize=2, label="measured")
            for B, v in zip(BATCHES, meas):
                if v == 0.0:
                    ax.plot([B], [0.0], marker="v", color=c, ms=4)
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 16, 256, 4096])
            ax.set_xticklabels(["1", "16", "256", "4096"])
            ax.set_ylim(-0.03, 0.62)
            ax.set_title(f"{label}, $d = {d}$" + (" (cal.)" if d in cal else " (held out)"), loc="left", fontsize=7)
            ax.grid(alpha=0.25)
        row[0].set_ylabel("$r^*/d$")
    for ax in axes[-1]:
        ax.set_xlabel("batch size $B$")
    axes[-1][-1].legend(loc="lower left", frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "crossover_map_both.pdf")
    plt.close(fig)


def fig_heldout_scatter_row(runs, out):
    """FLOPs and M1 held-out scatter for both platforms in one row of four panels."""
    fig, axes = plt.subplots(1, 4, figsize=(FULLW, 2.0))
    k = 0
    for label, a in runs:
        for model in ("P0", "M1"):
            ax = axes[k]; k += 1
            preds = a["predicted_energy_by_key"][model]
            xs, ys, cs = [], [], []
            for key, pred in preds.items():
                d = int(key.split("_")[0][1:])
                if d not in a["heldout_shapes"]:
                    continue
                xs.append(a["rep_stats"][key]["mean"])
                ys.append(pred if not isinstance(pred, dict) else pred.get("mean", np.nan))
                cs.append(COLORS[d])
            xs, ys = np.array(xs), np.array(ys)
            lim = (xs.min() * 0.5, xs.max() * 1.6)
            ys_plot = np.clip(ys, lim[0] * 1.08, lim[1] / 1.08)
            ax.plot(lim, lim, color="k", lw=0.6)
            ax.fill_between(lim, np.array(lim) / 1.5, np.array(lim) * 1.5, color="0.85", lw=0)
            ax.scatter(xs, ys_plot, c=cs, s=7, alpha=0.8, edgecolors="none")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(lim); ax.set_ylim(lim)
            med = a["heldout_prediction"][model]["median_ape"]
            name = "FLOPs" if model == "P0" else "additive"
            ax.set_title(f"{label}, {name}: {100 * med:.1f}%", loc="left", fontsize=7)
            ax.set_xlabel("measured (J)")
            ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("predicted (J)")
    fig.tight_layout()
    fig.savefig(out / "heldout_scatter_row.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, default=Path("paper/figures"))
    ap.add_argument("--second-run", type=Path, default=None,
                    help="a second run directory; with it the combined two-platform figures are also written "
                         "(the first run is labeled with --tag, the second with --second-tag)")
    ap.add_argument("--second-tag", default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    analysis, windows = load(args.run)
    fig_crossover_map(analysis, args.out, args.tag)
    fig_ratio_vs_rank(analysis, args.out, args.tag)
    fig_heldout_scatter(analysis, args.out, args.tag)
    fig_regimes(windows, args.out, args.tag)
    if args.second_run is not None:
        analysis2, _ = load(args.second_run)
        runs = [(args.tag, analysis), (args.second_tag or "second", analysis2)]
        fig_ratio_b1_both(runs, args.out)
        fig_heldout_scatter_both(runs, args.out)
        fig_crossover_map_both(runs, args.out)
        fig_heldout_scatter_row(runs, args.out)
    print("wrote figures to", args.out)


if __name__ == "__main__":
    main()
