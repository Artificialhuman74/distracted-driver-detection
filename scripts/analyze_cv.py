"""Summarise the driver-grouped cross-validation (runs/cv/results_seed*.jsonl) into outputs/cv/.

Statistics (all on TEST drivers of each fold; selection inside each run used validation drivers only):
  - mean +- std of accuracy / macro-F1 per variant over all fold x seed runs
  - paired comparisons vs baseline at FOLD level (average over seeds first; folds are the independent units):
    mean difference, 95% t-interval, paired t-test, Wilcoxon signed-rank (note: with 5 folds the smallest
    possible two-sided Wilcoxon p is 0.0625), and a bootstrap CI over folds
  - the same at RUN level (fold x seed pairs) for reference
  - identity probes (one-shot identification, verification AUC) per variant, incl. ImageNet-only features
Usage:  .venv/bin/python scripts/analyze_cv.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "cv"; (OUT / "figures").mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for f in sorted((ROOT / "runs" / "cv").glob("results_seed*.jsonl")) for l in f.read_text().splitlines() if l.strip()]
runs = [r for r in rows if r["variant"] != "imagenet"]
ref = [r for r in rows if r["variant"] == "imagenet"]
VARIANTS = ["baseline", "A", "C"]
NAMES = {"baseline": "Baseline (light aug)", "A": "A: scramble appearance", "C": "C: A + adversary (λ 1.0)"}
folds = sorted({r["fold"] for r in runs}); seeds = sorted({r["seed"] for r in runs})
get = lambda v, f, s: next((r for r in runs if r["variant"] == v and r["fold"] == f and r["seed"] == s), None)
complete = [(f, s) for f in folds for s in seeds if all(get(v, f, s) for v in VARIANTS)]
print(f"{len(runs)} runs, folds {folds}, seeds {seeds}, complete fold x seed cells: {len(complete)}")

def ms(x):
    x = np.asarray(x, float); return {"mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0, "n": int(len(x))}

summary = {"n_runs": len(runs), "folds": folds, "seeds": seeds, "n_complete_cells": len(complete),
           "fold_drivers": {str(f): next(r["test_drivers"] for r in runs if r["fold"] == f) for f in folds},
           "variants": {}, "paired_vs_baseline": {}, "identity": {}}
for v in VARIANTS:
    rr = [get(v, f, s) for f, s in complete]
    fold_mean = {f: float(np.mean([get(v, f, s)["test_acc"] for s in seeds if (f, s) in complete])) for f in folds}
    summary["variants"][v] = {
        "name": NAMES[v],
        "test_acc": ms([r["test_acc"] for r in rr]), "test_macro_f1": ms([r["test_macro_f1"] for r in rr]),
        "val_acc": ms([r["val_acc"] for r in rr]), "fold_mean_test_acc": fold_mean,
        "per_class_acc": {c: float(np.mean([r["per_class_acc"][c] for r in rr])) for c in rr[0]["per_class_acc"]},
        "worst_driver": min(((d, a) for r in rr for d, a in r["per_driver_acc"].items()), key=lambda t: t[1]),
    }
    summary["identity"][v] = {"one_shot_id": ms([r["one_shot_id"] for r in rr]), "verif_auc": ms([r["verif_auc"] for r in rr])}
summary["identity"]["imagenet"] = {"one_shot_id": ms([r["one_shot_id"] for r in ref]), "verif_auc": ms([r["verif_auc"] for r in ref])}
summary["identity"]["chance_one_shot"] = float(np.mean([r["chance_one_shot"] for r in ref])) if ref else None

rng = np.random.RandomState(0)
for v in ["A", "C"]:
    fd = np.array([summary["variants"][v]["fold_mean_test_acc"][f] - summary["variants"]["baseline"]["fold_mean_test_acc"][f] for f in folds])
    rd = np.array([get(v, f, s)["test_acc"] - get("baseline", f, s)["test_acc"] for f, s in complete])
    t_f = stats.ttest_rel([summary["variants"][v]["fold_mean_test_acc"][f] for f in folds],
                          [summary["variants"]["baseline"]["fold_mean_test_acc"][f] for f in folds])
    ci = stats.t.interval(0.95, len(fd) - 1, loc=fd.mean(), scale=stats.sem(fd)) if len(fd) > 1 else (float("nan"),) * 2
    boot = np.array([fd[rng.randint(0, len(fd), len(fd))].mean() for _ in range(10000)])
    summary["paired_vs_baseline"][v] = {
        "fold_level": {"diffs": fd.tolist(), "mean_diff": float(fd.mean()), "ci95_t": [float(ci[0]), float(ci[1])],
                       "ci95_bootstrap": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                       "t_test_p": float(t_f.pvalue), "wilcoxon_p": float(stats.wilcoxon(fd).pvalue) if len(fd) > 1 else None,
                       "folds_improved": int((fd > 0).sum()), "n_folds": int(len(fd))},
        "run_level": {"mean_diff": float(rd.mean()), "t_test_p": float(stats.ttest_1samp(rd, 0).pvalue),
                      "wilcoxon_p": float(stats.wilcoxon(rd).pvalue), "runs_improved": int((rd > 0).sum()), "n_runs": int(len(rd))},
    }
# identity: is C's (and A's) identity information lower than the baseline's? paired at fold level
for v in ["A", "C"]:
    for metric in ["one_shot_id", "verif_auc"]:
        fv = np.array([np.mean([get(v, f, s_)[metric] for s_ in seeds]) for f in folds])
        fb = np.array([np.mean([get("baseline", f, s_)[metric] for s_ in seeds]) for f in folds])
        summary["paired_vs_baseline"][v][metric] = {"mean_diff": float((fv - fb).mean()), "t_test_p": float(stats.ttest_rel(fv, fb).pvalue),
                                                    "folds_lower": int((fv < fb).sum()), "n_folds": int(len(folds))}
# safe-driving trade-off per variant (mean per-class accuracy)
summary["safe_driving_c0"] = {v: summary["variants"][v]["per_class_acc"]["c0"] for v in VARIANTS}
(OUT / "cv_summary.json").write_text(json.dumps(summary, indent=1))

# ---------------- figures ----------------
cols = {"baseline": "#8592a3", "A": "#7fb8a8", "C": "#3d8fe0"}
fig, ax = plt.subplots(figsize=(9, 4.4)); w = 0.26
for i, v in enumerate(VARIANTS):
    for j, f in enumerate(folds):
        accs = [get(v, f, s)["test_acc"] for s in seeds if get(v, f, s)]
        x = j + (i - 1) * w
        ax.bar(x, np.mean(accs), w * 0.92, color=cols[v], label=NAMES[v] if j == 0 else None)
        ax.scatter([x] * len(accs), accs, s=10, color="black", zorder=3)
ax.set_xticks(range(len(folds))); ax.set_xticklabels([f"fold {f}\n{len(summary['fold_drivers'][str(f)])} unseen drivers" for f in folds], fontsize=9)
ax.set_ylim(0.5, 1.0); ax.set_ylabel("Test accuracy (unseen drivers)")
ax.set_title("Driver-grouped 5-fold CV: bars = mean over seeds, dots = individual seeds")
ax.legend(fontsize=8, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "figures" / "cv_per_fold.png", dpi=150); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
keys = ["imagenet"] + VARIANTS; labels = ["ImageNet only"] + [NAMES[v] for v in VARIANTS]
for ax, metric, title, chance in [(axes[0], "one_shot_id", "One-shot identification (within action)", summary["identity"]["chance_one_shot"]),
                                  (axes[1], "verif_auc", "Same-person verification AUC (within action)", 0.5)]:
    m = [summary["identity"][k][metric]["mean"] for k in keys]; s = [summary["identity"][k][metric]["std"] for k in keys]
    ax.barh(labels, m, xerr=s, color=["#b4bfcc"] + [cols[v] for v in VARIANTS]); ax.axvline(chance, ls="--", color="k", lw=1)
    for y, val in enumerate(m): ax.text(val + 0.01, y, f"{val:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.08); ax.set_title(title, fontsize=10); ax.invert_yaxis(); ax.spines[["top", "right"]].set_visible(False)
axes[1].set_yticklabels([])
fig.suptitle("How much 'who is this?' is left in the features (lower = less identity; dashed = chance)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "figures" / "cv_identity.png", dpi=150); plt.close(fig)

for v in VARIANTS:
    t = summary["variants"][v]["test_acc"]; i1 = summary["identity"][v]
    print(f"{v:8s} test {t['mean']:.4f} ± {t['std']:.4f} (n={t['n']}) | 1-shot id {i1['one_shot_id']['mean']:.3f} | verif AUC {i1['verif_auc']['mean']:.3f}")
print("imagenet  1-shot id", round(summary["identity"]["imagenet"]["one_shot_id"]["mean"], 3), "verif AUC", round(summary["identity"]["imagenet"]["verif_auc"]["mean"], 3))
for v, p in summary["paired_vs_baseline"].items():
    fl = p["fold_level"]
    print(f"{v} - baseline: fold-level mean {fl['mean_diff']*100:+.2f} pts, 95% CI [{fl['ci95_t'][0]*100:+.2f}, {fl['ci95_t'][1]*100:+.2f}], "
          f"t p={fl['t_test_p']:.3f}, Wilcoxon p={fl['wilcoxon_p']:.3f}, folds improved {fl['folds_improved']}/{fl['n_folds']} | "
          f"run-level p(t)={p['run_level']['t_test_p']:.3f}, runs improved {p['run_level']['runs_improved']}/{p['run_level']['n_runs']}")
for v in ["A", "C"]:
    for metric in ["one_shot_id", "verif_auc"]:
        d = summary["paired_vs_baseline"][v][metric]
        print(f"{v} identity {metric}: diff {d['mean_diff']:+.3f}, t p={d['t_test_p']:.3f}, lower on {d['folds_lower']}/{d['n_folds']} folds")
print("per-class mean acc:", {v: {c: round(a, 3) for c, a in summary["variants"][v]["per_class_acc"].items()} for v in VARIANTS})
