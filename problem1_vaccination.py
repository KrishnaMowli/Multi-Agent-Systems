"""
Problem 1 (Question 6): Vaccination strategies on the contact network
Author: G V S S Krishna Mowli (EE25MTECH11019)

Run:  python problem1_vaccination.py   (takes about 2 minutes for 10,000 runs per strategy)
Needs network.csv and epidemic_data.csv in the same folder. Figures go to ./figures
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
rng = np.random.default_rng(2025)

B = 15            # vaccine budget
RUNS = 10000      # number of simulated epidemics per strategy
T_END = 200       # simulation length in days (long enough for the epidemic to finish)

# ------------------------------------------------------------------
# Load the network (edge list is stored once per edge, so make it symmetric)
# ------------------------------------------------------------------
edges = pd.read_csv(HERE / "network.csv")
data = pd.read_csv(HERE / "epidemic_data.csv")
N = int(data["node"].nunique())
A = np.zeros((N, N))
for a, b in zip(edges["source"], edges["target"]):
    A[a - 1, b - 1] = 1
    A[b - 1, a - 1] = 1
deg = A.sum(axis=1)
mean_deg = deg.mean()

# ------------------------------------------------------------------
# Parameters from the data (same estimates as in problem1_epidemic.py)
# ------------------------------------------------------------------
counts = data.groupby(["day", "state"]).size().unstack(fill_value=0)
S, I, R = counts["S"].to_numpy(), counts["I"].to_numpy(), counts["R"].to_numpy()
k = np.arange(len(S) - 1)
K = k[I[k] > 0]
gamma_hat = np.mean((R[K + 1] - R[K]) / I[K])
Kb = k[S[k] * I[k] > 0]
beta_hat = np.mean((S[Kb] - S[Kb + 1]) / (S[Kb] * I[Kb]))

# probability that one infected neighbour passes on the infection in one day.
# A susceptible person has on average mean_deg neighbours, a fraction I/N of them
# infected, so new infections per day = p * mean_deg / N * S * I.  Matching this
# with beta_hat * S * I gives p = beta_hat * N / mean_deg.
p_edge = beta_hat * N / mean_deg
p_recover = gamma_hat

seeds = np.array(sorted(data[(data["day"] == 0) & (data["state"] == "I")]["node"])) - 1

# ------------------------------------------------------------------
# Choosing the nodes to vaccinate
# ------------------------------------------------------------------
vals, vecs = np.linalg.eigh(A)
lam_max = vals[-1]
evec = np.abs(vecs[:, -1])                 # dominant eigenvector (made non-negative)

def top_nodes(score, count):
    score = score.astype(float).copy()
    score[seeds] = -np.inf                 # people already infected at t = 0 are not candidates
    return np.argsort(-score, kind="stable")[:count]

vac_degree = top_nodes(deg, B)
vac_eigen = top_nodes(evec, B)

def lam_after_removal(nodes):
    keep = np.setdiff1d(np.arange(N), nodes)
    return np.linalg.eigvalsh(A[np.ix_(keep, keep)])[-1]

# ------------------------------------------------------------------
# Stochastic SIR on the network (many epidemics are simulated together)
# states: 0 = S, 1 = I, 2 = R, 3 = vaccinated
# ------------------------------------------------------------------
def simulate(vacc_sets, runs=RUNS, t_end=T_END):
    X = np.zeros((runs, N), dtype=int)
    X[:, seeds] = 1
    X[np.arange(runs)[:, None], vacc_sets] = 3
    I_traj = np.zeros((runs, t_end + 1))
    for t in range(t_end + 1):
        infected = (X == 1)
        I_traj[:, t] = infected.sum(axis=1)
        n_inf_nbrs = infected.astype(float) @ A
        p_inf = 1.0 - (1.0 - p_edge) ** n_inf_nbrs
        new_inf = (X == 0) & (rng.random(X.shape) < p_inf)
        recover = infected & (rng.random(X.shape) < p_recover)
        X[new_inf] = 1
        X[recover] = 2
    n_infected = ((X == 1) | (X == 2)).sum(axis=1)        # people who got infected (incl. the 4 seeds)
    return I_traj, n_infected

def random_sets(runs=RUNS):
    score = rng.random((runs, N))
    score[:, seeds] = np.inf
    return np.argsort(score, axis=1)[:, :B]

strategies = {
    "No vaccination": np.zeros((RUNS, 0), dtype=int),
    "Random": random_sets(),
    "Degree-based": np.tile(vac_degree, (RUNS, 1)),
    "Eigenvector-based": np.tile(vac_eigen, (RUNS, 1)),
}

results, mean_curves = {}, {}
for name, sets in strategies.items():
    I_traj, n_inf = simulate(sets)
    imax = I_traj.max(axis=1)
    tpeak = I_traj.argmax(axis=1)
    mean_curves[name] = I_traj.mean(axis=0)
    results[name] = dict(
        Imax_mean=imax.mean(), Imax_std=imax.std(),
        Tpeak_mean=tpeak.mean(), Tpeak_std=tpeak.std(),
        Ninf_mean=n_inf.mean(), Ninf_std=n_inf.std(),
        small_outbreak_pct=100 * np.mean(n_inf < 0.5 * (N - sets.shape[1])),
    )

results["No vaccination"]["lambda_max"] = lam_max
results["Random"]["lambda_max"] = float(np.mean([lam_after_removal(s) for s in strategies["Random"][:200]]))
results["Degree-based"]["lambda_max"] = lam_after_removal(vac_degree)
results["Eigenvector-based"]["lambda_max"] = lam_after_removal(vac_eigen)

# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------
styles = {"No vaccination": ("k", "-"), "Random": ("tab:orange", "-"),
          "Degree-based": ("tab:blue", "-"), "Eigenvector-based": ("tab:red", "-")}
plt.figure(figsize=(6.6, 4.0))
for name, curve in mean_curves.items():
    c, ls = styles[name]
    plt.plot(np.arange(len(curve)), curve, color=c, ls=ls, label=name)
plt.plot(np.arange(len(I)), I, "k.", ms=3, alpha=0.5, label="Observed data (I[k])")
plt.xlim(0, 120)
plt.xlabel("Day")
plt.ylabel("Number of infected individuals")
plt.title("Average infection curves for B = 15 vaccinations")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "fig_vaccination.png", dpi=200)
plt.close()

chosen_deg = np.zeros(N, bool); chosen_deg[vac_degree] = True
chosen_eig = np.zeros(N, bool); chosen_eig[vac_eigen] = True
plt.figure(figsize=(5.6, 4.0))
plt.scatter(deg[~chosen_deg & ~chosen_eig], evec[~chosen_deg & ~chosen_eig], s=10, c="lightgray", label="Other nodes")
plt.scatter(deg[chosen_deg & chosen_eig], evec[chosen_deg & chosen_eig], s=28, c="tab:purple", label="Chosen by both")
plt.scatter(deg[chosen_deg & ~chosen_eig], evec[chosen_deg & ~chosen_eig], s=34, c="tab:blue", marker="s", label="Only degree")
plt.scatter(deg[~chosen_deg & chosen_eig], evec[~chosen_deg & chosen_eig], s=34, c="tab:red", marker="^", label="Only eigenvector")
plt.xlabel("Degree of the node")
plt.ylabel("Eigenvector centrality")
plt.title("Degree and eigenvector centrality of each node")
plt.grid(alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "fig_centrality.png", dpi=200)
plt.close()

# ------------------------------------------------------------------
# Print and save results
# ------------------------------------------------------------------
only_deg = sorted(set(vac_degree) - set(vac_eigen))
only_eig = sorted(set(vac_eigen) - set(vac_degree))
extra = dict(
    N=N, mean_degree=mean_deg, lambda_max=lam_max, p_edge=p_edge, p_recover=p_recover,
    beta_hat=beta_hat, gamma_hat=gamma_hat, seeds=[int(s) + 1 for s in seeds],
    vac_degree=[int(v) + 1 for v in vac_degree], vac_eigen=[int(v) + 1 for v in vac_eigen],
    overlap=len(set(vac_degree) & set(vac_eigen)),
    only_degree=[(int(v) + 1, int(deg[v]), round(float(evec[v]), 4),
                  int(np.sum(evec > evec[v]) + 1)) for v in only_deg],
    only_eigen=[(int(v) + 1, int(deg[v]), round(float(evec[v]), 4),
                 int(np.sum(deg > deg[v]) + 1)) for v in only_eig],
    degree_of_15th=int(np.sort(deg)[::-1][14]), degree_of_16th=int(np.sort(deg)[::-1][15]),
    sum_deg_degree=int(deg[vac_degree].sum()), sum_deg_eigen=int(deg[vac_eigen].sum()),
    edges_removed_degree=int(A[vac_degree].sum() - A[np.ix_(vac_degree, vac_degree)].sum() / 2),
    edges_removed_eigen=int(A[vac_eigen].sum() - A[np.ix_(vac_eigen, vac_eigen)].sum() / 2),
    mean_deg_random=float(deg.mean()),
)
for name, r in results.items():
    print(name, {kk: round(float(vv), 2) for kk, vv in r.items()})
for kk, vv in extra.items():
    print(kk, vv)
with open(HERE / "results_vaccination.json", "w") as f:
    json.dump(dict(results=results, extra=extra), f, indent=2, default=float)
