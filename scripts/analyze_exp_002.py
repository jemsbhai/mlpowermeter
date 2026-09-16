#!/usr/bin/env python
"""Analyze one EXP-002 run directory (pre-registered analysis, criteria E1 to E7).

    python scripts/analyze_exp_002.py --run-dir experiments/exp_002_lowrank-crossover/rtx4090-laptop/<run-id>

Writes results/analysis.json and figures/*.png into the run directory and
prints the criteria table. Re-running overwrites analysis.json and figures
(they are derived artifacts; the measured windows are never touched).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from tomlml.analysis.exp_002 import analyze, criteria_table, encode, load_run
from tomlml.utils import git_info, write_json


def _figures(run: Dict[str, Any], a: Dict[str, Any], out_dir: Path) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    grid = run["summary"]["grid"]
    sel = a["selected_model"]
    means = a["rep_stats"]
    fracs = [float(x) for x in grid["rank_fractions"]]
    batches = [int(x) for x in grid["batches"]]
    shapes = [int(x) for x in grid["shapes"]]
    held = set(a["heldout_shapes"])

    # 1. measured versus predicted energy per call, held-out configurations
    if a["heldout_shapes"]:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
        for ax, model in zip(axes, (sel, "P0")):
            xs, ys, cs = [], [], []
            for key, st in means.items():
                d = int(key.split("_")[0][1:])
                if d not in held:
                    continue
                pred = _predicted_energy(a, model, key)
                if pred is None:
                    continue
                xs.append(st["mean"]); ys.append(pred); cs.append("tab:blue" if key.endswith("_dense") else "tab:orange")
            if xs:
                lo, hi = min(min(xs), min(ys)), max(max(xs), max(ys))
                ax.plot([lo, hi], [lo, hi], "k--", lw=1)
                ax.scatter(xs, ys, c=cs, s=14, alpha=0.8)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("measured energy per call (J)"); ax.set_ylabel("predicted energy per call (J)")
            mape = a["heldout_prediction"][model]["median_ape"]
            ax.set_title(f"{model}: held-out median APE {mape:.1%}" if mape is not None else model)
        fig.suptitle(f"EXP-002 {a['platform_tag']} run {a['run_id']}: held-out shapes {a['heldout_shapes']} "
                     "(blue dense, orange factorized)")
        fig.tight_layout()
        p = out_dir / "energy_pred_vs_meas.png"
        fig.savefig(p, dpi=150); plt.close(fig); written.append(str(p))

    # 2. ratio E_fact / E_dense versus rank fraction, one panel per (d, B)
    fig, axes = plt.subplots(len(shapes), len(batches), figsize=(2.6 * len(batches), 2.4 * len(shapes)),
                             sharex=True, sharey=True, squeeze=False)
    for i, d in enumerate(shapes):
        for j, B in enumerate(batches):
            ax = axes[i][j]
            cell = f"d{d}_B{B}"
            mx, my = [], []
            px, py = [], []
            for f in fracs:
                kd, kf = f"d{d}_f{f:g}_B{B}_dense", f"d{d}_f{f:g}_B{B}_factorized"
                if kd in means and kf in means:
                    mx.append(f); my.append(means[kf]["mean"] / means[kd]["mean"])
                pd_, pf_ = _predicted_energy(a, sel, kd), _predicted_energy(a, sel, kf)
                if pd_ and pf_:
                    px.append(f); py.append(pf_ / pd_)
            ax.axhline(1.0, color="k", lw=0.8)
            ax.plot(fracs, [2 * f for f in fracs], color="gray", lw=1, ls=":", label="FLOPs")
            if px:
                ax.plot(px, py, color="tab:green", lw=1.2, label=sel)
            if mx:
                ax.plot(mx, my, "o", color="tab:red", ms=3.5, label="measured")
            m = a["measured_crossovers"].get(cell)
            if m and m["kind"] == "numeric":
                ax.axvline(m["r_star"] / d, color="tab:red", lw=0.8, ls="--")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_title(f"d={d} B={B}{' (held-out)' if d in held else ''}", fontsize=8)
            if i == len(shapes) - 1:
                ax.set_xlabel("rank fraction r/d", fontsize=8)
            if j == 0:
                ax.set_ylabel("E_fact / E_dense", fontsize=8)
            ax.tick_params(labelsize=7)
    axes[0][0].legend(fontsize=6)
    fig.suptitle("EXP-002: factorized over dense energy ratio; dashed red = measured crossover", fontsize=10)
    fig.tight_layout()
    p = out_dir / "ratio_vs_rank.png"
    fig.savefig(p, dpi=150); plt.close(fig); written.append(str(p))

    # 3. crossover rank fraction versus batch, per shape, with intervals on held-out shapes
    fig, axes = plt.subplots(1, len(shapes), figsize=(3.2 * len(shapes), 3.4), sharey=True, squeeze=False)
    for j, d in enumerate(shapes):
        ax = axes[0][j]
        xs = batches
        meas = [encode(a["measured_crossovers"][f"d{d}_B{B}"]) / d if f"d{d}_B{B}" in a["measured_crossovers"] else None
                for B in xs]
        pred = [encode(a["predicted_crossovers"][sel][f"d{d}_B{B}"]) / d if f"d{d}_B{B}" in a["predicted_crossovers"][sel] else None
                for B in xs]
        floor, ceil = 0.008, 1.5   # plotting positions for none_dense and none_factorized

        def clip(v):
            return None if v is None else (floor if v == 0 else (ceil if math.isinf(v) else v))
        ax.axhline(0.5, color="gray", ls=":", lw=1, label="FLOPs (r* = d/2)")
        ax.plot(xs, [clip(v) for v in pred], "s-", color="tab:green", ms=4, label=f"{sel} predicted")
        ax.plot(xs, [clip(v) for v in meas], "o", color="tab:red", ms=5, label="measured")
        if d in held:
            for B in xs:
                bc = a["bootstrap"]["cells"].get(f"d{d}_B{B}")
                if bc:
                    lo, hi = clip(bc["lo"] / d if bc["lo"] != 0 else 0), clip(bc["hi"] / d if math.isfinite(bc["hi"]) else math.inf)
                    ax.plot([B, B], [lo, hi], color="tab:green", lw=3, alpha=0.3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylim(floor / 1.5, ceil * 1.3)
        ax.axhline(floor, color="k", lw=0.5); ax.axhline(ceil, color="k", lw=0.5)
        ax.text(xs[0], floor, " dense always wins", fontsize=6, va="bottom")
        ax.text(xs[0], ceil, " factorized always wins", fontsize=6, va="top")
        ax.set_title(f"d={d}{' (held-out)' if d in held else ' (calibration)'}", fontsize=9)
        ax.set_xlabel("batch size B")
        if j == 0:
            ax.set_ylabel("crossover rank fraction r*/d")
        ax.tick_params(labelsize=7)
    axes[0][0].legend(fontsize=6, loc="lower right")
    fig.suptitle("EXP-002: crossover rank versus batch (green band = 95% bootstrap interval)", fontsize=10)
    fig.tight_layout()
    p = out_dir / "crossover_vs_batch.png"
    fig.savefig(p, dpi=150); plt.close(fig); written.append(str(p))

    # 4. diagnostics: repetition CV, temperature and clock over the run
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    cvs = [st["cv"] for st in means.values() if st["cv"] is not None]
    axes[0].hist([100 * c for c in cvs], bins=30, color="tab:blue")
    axes[0].set_xlabel("repetition CV of energy per call (%)"); axes[0].set_ylabel("configurations")
    axes[0].set_title(f"median CV {100 * float(np.median(cvs)):.2f}%" if cvs else "no CV")
    ws = run["windows"]
    axes[1].plot([w.get("mean_temp_c") for w in ws], lw=0.7, color="tab:red")
    axes[1].set_xlabel("window index (run order)"); axes[1].set_ylabel("mean temperature (C)")
    axes[2].plot([w.get("mean_sm_clock_mhz") for w in ws], lw=0.7, color="tab:purple")
    axes[2].set_xlabel("window index (run order)"); axes[2].set_ylabel("mean SM clock (MHz)")
    fig.tight_layout()
    p = out_dir / "diagnostics.png"
    fig.savefig(p, dpi=150); plt.close(fig); written.append(str(p))
    return written


def _predicted_energy(a: Dict[str, Any], model: str, key: str) -> Any:
    """Predicted energy for one configuration key from the analysis output."""
    return a.get("predicted_energy_by_key", {}).get(model, {}).get(key)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--resamples", type=int, default=None, help="bootstrap resamples (default: config)")
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    run = load_run(run_dir)
    a = analyze(run, n_resamples=args.resamples)
    a["analysis_code"] = git_info()
    figures = [] if args.no_figures else _figures(run, a, run_dir / "figures")
    a["figures"] = figures
    write_json(run_dir / "results" / "analysis.json", a)
    print(criteria_table(a))
    print(f"analysis written to {run_dir / 'results' / 'analysis.json'}; {len(figures)} figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
