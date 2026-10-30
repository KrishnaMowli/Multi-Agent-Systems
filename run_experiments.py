import numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cm_simulator import simulate, summarize, CAP

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 8.5,
                     "axes.grid": True, "grid.alpha": 0.3})
TH = {"S1": 0.75, "S2": 1.00, "S3": 1.25, "S4": 1.50}
COL = {"S1": "#2a9d8f", "S2": "#1f4e79", "S3": "#e9a03b", "S4": "#c0392b"}
DCOL = ["#1f4e79", "#2a9d8f", "#c0392b", "#e9a03b", "#7b5ea7"]

# ---------- Part I: baseline (no feedback), 20 seeds ----------
base = {s: [simulate(th, seed) for seed in range(1, 21)] for s, th in TH.items()}
rows = {}
for s in TH:
    sm = [summarize(o) for o in base[s]]
    rows[s] = {k: (float(np.mean([x[k] for x in sm])), float(np.std([x[k] for x in sm]))) for k in sm[0]}
json.dump(rows, open("figures/baseline_stats.json", "w"), indent=1)
for s in TH: print("baseline", s, {k: (round(v[0], 2), round(v[1], 2)) for k, v in rows[s].items()})

# (c) Q_i(d): single run (seed 1), 2x2 panel
fig, axs = plt.subplots(2, 2, figsize=(7.1, 4.3), sharex=True)
for ax, (s, th) in zip(axs.ravel(), TH.items()):
    o = base[s][0]
    for i in range(5):
        ax.plot(np.arange(1, 366), o["Q"][:, i], lw=1.0, color=DCOL[i], label=f"$D_{i+1}$")
    ax.set_title(f"{s} ($\\theta$={th})", fontsize=9); ax.set_ylabel("$Q_i(d)$")
for ax in axs[1]: ax.set_xlabel("Day $d$")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, ncol=5, fontsize=8, loc="upper center", frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig("figures/queues.pdf"); plt.close(fig)

# (d) B_d and R_d : mean of 20 runs with +-1 sd band
for key, ylab, name in [("B", "$B_d$ (unfinished applications)", "backlog"),
                        ("R", "$R_d$ (cumulative reward, $\\times10^3$)", "cumreward")]:
    fig, ax = plt.subplots(figsize=(3.45, 2.7))
    for s in TH:
        M = np.array([o[key] for o in base[s]]) / (1000.0 if key == "R" else 1.0); mu, sd = M.mean(0), M.std(0)
        ax.plot(np.arange(1, 366), mu, color=COL[s], lw=1.2, label=s)
        lo = np.maximum(mu - sd, 0) if key == "B" else mu - sd
        ax.fill_between(np.arange(1, 366), lo, mu + sd, color=COL[s], alpha=0.15, lw=0)
    ax.set_xlabel("Day $d$"); ax.set_ylabel(ylab); ax.legend(frameon=False, fontsize=7.5)
    fig.subplots_adjust(left=0.2, right=0.97, bottom=0.17, top=0.96); fig.savefig(f"figures/{name}.pdf"); plt.close(fig)

# ---------- Part IV: final assessment with feedback, 30 seeds ----------
fin = {s: [simulate(th, seed, feedback=True) for seed in range(101, 131)] for s, th in TH.items()}
frows = {}
for s in TH:
    sm = [summarize(o) for o in fin[s]]
    frows[s] = {k: (float(np.mean([x[k] for x in sm])), float(np.std([x[k] for x in sm]))) for k in sm[0]}
    frows[s]["mean_lambda"] = (float(np.mean([o["lam"].mean() for o in fin[s]])), float(np.std([o["lam"].mean() for o in fin[s]])))
    frows[s]["max_B"] = (float(np.mean([o["B"].max() for o in fin[s]])), 0.0)
    frows[s]["m"] = fin[s][0]["m"].tolist()
json.dump(frows, open("figures/final_stats.json", "w"), indent=1)
for s in TH: print("final", s, {k: (round(v[0], 2), round(v[1], 2)) if k != "m" else v for k, v in frows[s].items()})

# illustration for 'popular scheme can fail': feedback runs, mean q_d and B_d
fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.5))
for s in TH:
    axs[0].plot(np.arange(1, 366), np.mean([o["lam"] for o in fin[s]], 0), color=COL[s], lw=1.1, label=s)
    axs[1].plot(np.arange(1, 366), np.mean([o["B"] for o in fin[s]], 0), color=COL[s], lw=1.1, label=s)
axs[0].set_ylabel("Expected demand $\\lambda_d$"); axs[1].set_ylabel("$B_d$ (mean of 30 runs)")
axs[0].axhline(24, color="k", ls="--", lw=0.8); axs[0].set_ylim(11, 28); axs[0].text(120, 26.6, "$D_3$ capacity = 24/day", fontsize=7.5)
for a in axs: a.set_xlabel("Day $d$")
axs[0].legend(frameon=False, fontsize=7.5, ncol=4, loc="lower right", columnspacing=0.8, handlelength=1.2)
fig.tight_layout(); fig.savefig("figures/feedback_dynamics.pdf"); plt.close(fig)
