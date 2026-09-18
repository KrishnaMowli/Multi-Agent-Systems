"""
==============================================================================
Q3_Task_Allocation_ADMM.py
==============================================================================
AI3403: Multi-Agent Systems -- Assignment 3, Problem 3
IIT Hyderabad

WHAT THIS FILE DOES
--------------------
N agents, connected by an undirected graph G(V,E), must collaboratively
execute M > N tasks. Agent i has capacity b_i (number of tasks it can take),
with sum_i b_i = M, and a private cost row C_i giving its cost for every
task. Each task must be assigned to exactly one agent.

1. Formulate the binary assignment problem, and its convex (box) relaxation.
2. Show this relaxed problem is exactly a "resource sharing" problem (Boyd,
   ADMM survey, Sec. 7.3): minimize sum_i f_i(x_i) subject to sum_i x_i = 1_M,
   where x_i in R^M is agent i's own row of the assignment matrix and
   f_i(x_i) = C_i^T x_i + indicator{ x_i in local capacity simplex }.
3. Solve it with the standard **Sharing-ADMM** algorithm. The one quantity
   that needs to be shared across agents (the running average x_bar) is
   computed in a genuinely DISTRIBUTED way: agents run several rounds of
   average-consensus (gossip) over the communication graph G every ADMM
   iteration, rather than any agent ever seeing another agent's cost row.
4. Generate a random connected graph, random capacities, and a random
   positive cost matrix; run ADMM; plot the graph and the resulting
   task-allocation matrix and convergence.
5. Repeat with a second, independently-drawn cost matrix (same M, N, graph,
   capacities) and compare, as requested by the problem statement.

HOW TO RUN
----------
    python Q3_Task_Allocation_ADMM.py

OUTPUTS (written to ./figures/)
--------------------------------
    figures/q3_communication_graph.png
    figures/q3_task_allocation.png            (cost matrix run #1)
    figures/q3_task_allocation_run2.png       (cost matrix run #2, for comparison)
    figures/q3_admm_convergence.png
"""

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------
# 0. Reproducibility and output folder
# -----------------------------------------------------------------------
SEED = 3
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# 1. Problem size: choose M > N, fix capacities so that sum_i b_i = M
# -----------------------------------------------------------------------
N = 6                 # number of agents
M = 17                # number of tasks (M > N, as required)
P_ER = 0.5            # Erdos-Renyi edge probability for the comm. graph

# Random capacities b_i >= 0 with sum_i b_i = M: draw M-N "extra" tasks and
# distribute them randomly on top of a base capacity of 1 per agent.
base = np.ones(N, dtype=int)
extra_counts = np.bincount(rng.integers(0, N, size=M - N), minlength=N)
capacities = base + extra_counts
assert capacities.sum() == M
print(f"N={N} agents, M={M} tasks, capacities b_i = {capacities.tolist()} "
      f"(sum = {capacities.sum()})")

# Random positive cost matrix C in R^{N x M}, C_ij > 0
C = rng.uniform(1.0, 10.0, size=(N, M))

# -----------------------------------------------------------------------
# 2. Connected communication graph (Erdos-Renyi) + Metropolis weights
# -----------------------------------------------------------------------
def make_connected_er_graph(n, p, seed):
    trial = 0
    while True:
        g = nx.erdos_renyi_graph(n, p, seed=seed + trial)
        if nx.is_connected(g):
            return g
        trial += 1

G = make_connected_er_graph(N, P_ER, SEED)
print(f"Communication graph: N={N}, p={P_ER}, |E|={G.number_of_edges()}, "
      f"connected={nx.is_connected(G)}")

degree = dict(G.degree())
Wmat = np.zeros((N, N))
for i, j in G.edges():
    Wmat[i, j] = 1.0 / (1 + max(degree[i], degree[j]))
    Wmat[j, i] = Wmat[i, j]
for i in range(N):
    Wmat[i, i] = 1.0 - Wmat[i, :].sum()

pos_graph = nx.spring_layout(G, seed=SEED)
plt.figure(figsize=(5.5, 5.5))
nx.draw(G, pos_graph, with_labels=True, node_color="salmon",
        edge_color="gray", node_size=550, font_size=10)
plt.title(f"Q3: Agent Communication Graph (N={N})")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "q3_communication_graph.png"), dpi=150)
plt.close()

GOSSIP_ROUNDS = 25   # average-consensus rounds run per ADMM iteration, so
                     # that the network-wide average x_bar used below is
                     # obtained in a genuinely distributed way (each agent
                     # only ever exchanges its own current x_i with its
                     # graph neighbours).

def distributed_average(X):
    """
    Approximate the true column-wise average of X (one row per agent) using
    only graph-neighbour gossip (X <- W X), which provably converges to the
    exact average for a connected graph with a doubly-stochastic W.
    """
    Xc = X.copy()
    for _ in range(GOSSIP_ROUNDS):
        Xc = Wmat @ Xc
    return Xc  # every row already (approximately) equals the true average

# -----------------------------------------------------------------------
# 3. Projection of a point onto the local feasible set
#        Delta_i = { x in R^M : sum_j x_j = b_i ,  0 <= x_j <= 1 }
#    via a standard bisection on the simplex/box Lagrange multiplier.
# -----------------------------------------------------------------------
def project_capacity_simplex(v, b, lo=0.0, hi=1.0, tol=1e-10, max_iter=100):
    """Euclidean projection of vector v onto {x : sum x = b, lo<=x<=hi}."""
    def clipped_sum(mu):
        return np.clip(v - mu, lo, hi).sum()

    mu_lo, mu_hi = v.min() - hi, v.max() - lo
    for _ in range(max_iter):
        mu_mid = 0.5 * (mu_lo + mu_hi)
        if clipped_sum(mu_mid) > b:
            mu_lo = mu_mid
        else:
            mu_hi = mu_mid
        if mu_hi - mu_lo < tol:
            break
    mu_star = 0.5 * (mu_lo + mu_hi)
    return np.clip(v - mu_star, lo, hi)

# -----------------------------------------------------------------------
# 4. Sharing-ADMM for the task-allocation problem
#
#    minimize   sum_i  C_i^T x_i                       (total cost)
#    subject to x_i in Delta_i   (local capacity/box)   for every agent i
#               sum_i x_i = 1_M                          (each task assigned once)
#
#    Standard "Sharing" ADMM update (Boyd et al., ADMM survey Sec. 7.3),
#    scaled dual variable u (identical at every agent once gossip has mixed
#    it -- see report for the full derivation):
#
#       x_i^{k+1} = argmin_{x_i in Delta_i}
#                       C_i^T x_i + (rho/2)|| x_i - x_i^k + xbar^k - 1_M/N + u^k ||^2
#                 = Proj_{Delta_i}( x_i^k - xbar^k + 1_M/N - u^k - C_i/rho )
#
#       xbar^{k+1} = (1/N) sum_i x_i^{k+1}      (via distributed gossip)
#       u^{k+1}    = u^k + xbar^{k+1} - 1_M/N
# -----------------------------------------------------------------------
def run_admm_task_allocation(C, capacities, rho=1.0, n_iter=150):
    N_, M_ = C.shape
    X = np.zeros((N_, M_))
    for i in range(N_):
        X[i] = capacities[i] * np.ones(M_) / M_   # feasible-ish initial guess
    u = np.zeros(M_)

    primal_residual_hist = []   # || sum_i x_i - 1_M ||
    dual_residual_hist = []     # rho * || xbar^{k} - xbar^{k-1} ||
    objective_hist = []         # sum_ij C_ij X_ij

    xbar_prev = X.mean(axis=0)
    for k in range(n_iter):
        xbar = distributed_average(X)   # (N_, M_), every row ~ true average
        X_new = np.zeros_like(X)
        for i in range(N_):
            target = X[i] - xbar[i] + np.ones(M_) / N_ - u - C[i] / rho
            X_new[i] = project_capacity_simplex(target, capacities[i])
        X = X_new

        xbar_new = X.mean(axis=0)
        u = u + xbar_new - np.ones(M_) / N_

        primal_residual_hist.append(np.linalg.norm(X.sum(axis=0) - 1.0))
        dual_residual_hist.append(rho * np.linalg.norm(xbar_new - xbar_prev))
        objective_hist.append(np.sum(C * X))
        xbar_prev = xbar_new

    return X, primal_residual_hist, dual_residual_hist, objective_hist

X_final, primal_res, dual_res, obj_hist = run_admm_task_allocation(C, capacities)
print(f"ADMM finished. Final primal residual ||sum_i x_i - 1||: {primal_res[-1]:.3e}")
print(f"Final total (relaxed) cost sum_ij C_ij X_ij: {obj_hist[-1]:.4f}")
print(f"Row sums (should match capacities): {X_final.sum(axis=1).round(3)}")
print(f"Column sums (should be ~1 for every task): "
      f"min={X_final.sum(axis=0).min():.3f}, max={X_final.sum(axis=0).max():.3f}")

# -----------------------------------------------------------------------
# 5. Plot: task-allocation matrix heatmap (run #1)
# -----------------------------------------------------------------------
plt.figure(figsize=(9, 4.5))
plt.imshow(X_final, aspect="auto", cmap="viridis", vmin=0, vmax=1)
plt.colorbar(label=r"$X_{ij}$ (fraction of task $j$ given to agent $i$)")
plt.xlabel("Task index $j$"); plt.ylabel("Agent index $i$")
plt.title("Q3: ADMM Task-Allocation Matrix (cost matrix #1)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "q3_task_allocation.png"), dpi=150)
plt.close()

# -----------------------------------------------------------------------
# 6. Plot: ADMM convergence (primal/dual residuals + objective)
# -----------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
axs[0].plot(primal_res, label="Primal residual $\\|\\sum_i x_i - 1\\|$")
axs[0].plot(dual_res, label="Dual residual")
axs[0].set_yscale("log")
axs[0].set_xlabel("ADMM iteration $k$")
axs[0].set_ylabel("Residual norm (log scale)")
axs[0].set_title("Q3: ADMM Primal/Dual Residuals")
axs[0].legend()
axs[0].grid(alpha=0.3)

axs[1].plot(obj_hist, color="tab:blue")
axs[1].set_xlabel("ADMM iteration $k$")
axs[1].set_ylabel(r"$\sum_{ij} C_{ij} X_{ij}$")
axs[1].set_title("Q3: Relaxed Objective Value vs. Iteration")
axs[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "q3_admm_convergence.png"), dpi=150)
plt.close()

# -----------------------------------------------------------------------
# 7. Repeat with a SECOND, independently-drawn cost matrix (same M, N,
#    graph, capacities) and compare -- as requested by the problem.
# -----------------------------------------------------------------------
rng2 = np.random.default_rng(SEED + 100)
C2 = rng2.uniform(1.0, 10.0, size=(N, M))
X_final2, primal_res2, dual_res2, obj_hist2 = run_admm_task_allocation(C2, capacities)

plt.figure(figsize=(9, 4.5))
plt.imshow(X_final2, aspect="auto", cmap="viridis", vmin=0, vmax=1)
plt.colorbar(label=r"$X_{ij}$ (fraction of task $j$ given to agent $i$)")
plt.xlabel("Task index $j$"); plt.ylabel("Agent index $i$")
plt.title("Q3: ADMM Task-Allocation Matrix (cost matrix #2, same $M,N,$graph,capacities)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "q3_task_allocation_run2.png"), dpi=150)
plt.close()

# a quick numeric comparison of how "hard"/binary each solution is, and cost
def hardness(X):
    """Average distance of every entry from the nearest of {0,1} -- lower
    means the relaxed solution is closer to a genuine binary assignment."""
    return np.mean(np.minimum(X, 1 - X))

print(f"\nRun #1: total cost = {obj_hist[-1]:.3f},  relaxation hardness = {hardness(X_final):.4f}")
print(f"Run #2: total cost = {obj_hist2[-1]:.3f}, relaxation hardness = {hardness(X_final2):.4f}")

print("\nQ3 complete. All figures saved under ./figures/")
