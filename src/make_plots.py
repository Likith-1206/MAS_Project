"""Aggregate results/eval_*.csv -> summary tables, statistical tests and plots (results/*.png, results/*.csv)."""
import glob, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
LABEL = {"fixed_time": "Fixed-time", "max_pressure": "Max-Pressure", "independent": "Independent DQN",
         "coop": "Cooperative DQN", "shared_c0": "Shared, no comm", "shared_c1": "Shared, queue comm", "shared_c2": "Shared, queue+phase comm"}
METRICS = [("avg_wait_s", "Avg waiting time / vehicle (s)"), ("avg_queue", "Avg queue / intersection (veh)"),
           ("max_queue", "Max lane queue (veh)"), ("throughput", "Throughput (veh arrived)"),
           ("total_travel_h", "Total travel time (veh-h)")]
MAIN = ["fixed_time", "max_pressure", "independent", "coop"]
ABL = ["shared_c0", "shared_c1", "shared_c2", "coop"]
COL = {"fixed_time": "#9aa0a6", "max_pressure": "#34a853", "independent": "#fbbc04", "coop": "#1a73e8",
       "shared_c0": "#c5cae9", "shared_c1": "#7986cb", "shared_c2": "#3f51b5"}

df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(R, "eval_*.csv"))], ignore_index=True)
# one number per (method, scenario, eval_seed, train_seed); average over train seeds later
scn_order = ["uniform", "asymmetric", "rush_hour", "corridor"]

# ---- summary table: mean +- std over all (train seed x eval seed x scenario) episodes ----
rows = []
for m in [x for x in LABEL if x in set(df.method)]:
    d = df[df.method == m]
    r = {"method": LABEL[m], "n_episodes": len(d), "train_seeds": d.train_seed.nunique() if m not in ("fixed_time", "max_pressure") else "-"}
    for k, _ in METRICS:
        r[k] = f"{d[k].mean():.1f} ± {d[k].std():.1f}"
    rows.append(r)
pd.DataFrame(rows).to_csv(os.path.join(R, "summary_table.csv"), index=False)
per = df.groupby(["scenario", "method"])[[k for k, _ in METRICS]].mean().round(2)
per.to_csv(os.path.join(R, "summary_per_scenario.csv"))

# ---- statistical tests: paired (same scenario & eval seed) differences in avg waiting time ----
def paired(a, b, metric="avg_wait_s"):
    x = df[df.method == a].groupby(["scenario", "eval_seed"])[metric].mean()
    y = df[df.method == b].groupby(["scenario", "eval_seed"])[metric].mean()
    idx = x.index.intersection(y.index)
    d = (x[idx] - y[idx]).values
    if len(d) < 3: return None
    t = stats.ttest_1samp(d, 0.0); w = stats.wilcoxon(d) if np.any(d != 0) else None
    return dict(A=LABEL[a], B=LABEL[b], metric=metric, n_pairs=len(d), mean_diff_A_minus_B=round(d.mean(), 2),
                rel_change_pct=round(100 * d.mean() / y[idx].mean(), 1), t_p=round(float(t.pvalue), 4),
                wilcoxon_p=round(float(w.pvalue), 4) if w else None)
tests = [paired(a, b) for a, b in [("coop", "independent"), ("coop", "fixed_time"), ("coop", "max_pressure"),
                                   ("independent", "fixed_time"), ("independent", "max_pressure"),
                                   ("shared_c1", "shared_c0"), ("shared_c2", "shared_c0"), ("coop", "shared_c2")]
         if a in set(df.method) and b in set(df.method)]
pd.DataFrame([t for t in tests if t]).to_csv(os.path.join(R, "stat_tests.csv"), index=False)

# ---- bar plots per metric, grouped by scenario ----
def bars(methods, fname, title):
    methods = [m for m in methods if m in set(df.method)]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, (k, lab) in zip(axs, [METRICS[0], METRICS[1], METRICS[3]]):
        w = 0.8 / len(methods)
        for i, m in enumerate(methods):
            g = df[df.method == m].groupby("scenario")[k]
            mu = g.mean().reindex(scn_order); sd = g.std().reindex(scn_order)
            ax.bar(np.arange(4) + i * w, mu, w, yerr=sd, capsize=2, label=LABEL[m], color=COL[m])
        ax.set_xticks(np.arange(4) + 0.4 - w / 2); ax.set_xticklabels(scn_order); ax.set_ylabel(lab); ax.grid(axis="y", alpha=.3)
    axs[0].legend(fontsize=8); fig.suptitle(title); fig.tight_layout(); fig.savefig(os.path.join(R, fname), dpi=130); plt.close(fig)
bars(MAIN, "comparison_main.png", "Fixed-time vs Max-Pressure vs Independent DQN vs Cooperative DQN (mean ± std over seeds)")
bars(ABL, "ablation_communication.png", "RQ2: effect of neighbour information (shared network)")

# ---- learning curves ----
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for m in ["independent", "coop", "shared_c0", "shared_c1", "shared_c2"]:
    fs = glob.glob(os.path.join(R, f"curve_{m}_s*.csv"))
    if not fs: continue
    c = pd.concat([pd.read_csv(f) for f in fs]).groupby("episode")
    for ax, k in zip(axs, ["ret", "avg_wait_s"]):
        y = c[k].mean().rolling(4, min_periods=1).mean()
        ax.plot(y.index + 1, y.values, label=LABEL[m], color=COL[m])
axs[0].set_ylabel("Episode return (mean per agent, raw reward)"); axs[1].set_ylabel("Avg waiting time (s), training episodes")
for ax in axs: ax.set_xlabel("Training episode"); ax.grid(alpha=.3)
axs[1].legend(fontsize=8); fig.suptitle("Learning curves (4-episode moving average; scenario cycles every episode, so curves are noisy)")
fig.tight_layout(); fig.savefig(os.path.join(R, "learning_curves.png"), dpi=130); plt.close(fig)
print(pd.DataFrame(rows).to_string(index=False)); print(pd.DataFrame([t for t in tests if t]).to_string(index=False))
