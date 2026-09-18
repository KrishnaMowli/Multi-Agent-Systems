"""
==============================================================================
Q2_Distributed_State_Estimation.py
==============================================================================
AI3403: Multi-Agent Systems -- Assignment 3, Problem 2
IIT Hyderabad

WHAT THIS FILE DOES
--------------------
A swarm of N drones, connected by a communication graph G(V,E), must
collaboratively track a randomly-walking intruder z(t) in R^3 using noisy,
sensor-local, linear measurements. We:

1. Build a connected Erdos-Renyi communication graph over N (<20) drones.
2. Fix random sensor locations x_i, the intruder's prior N(z0_bar, Sigma0),
   the process noise covariance Sigma_w, and each sensor's own measurement
   noise covariance Sigma_v^(i).
3. Simulate the intruder's true random-walk trajectory z(t), t = 0..Tmax.
4. Simulate each sensor's noisy local measurement y_i(t) = x_i - z(t) + v_i.
5. Formulate distributed state estimation as a consensus-optimization
   problem: at every time t, all drones cooperatively minimise
        sum_i f_i(z) ,     f_i(z) = (1/2)(z - m_i(t))^T Sigma_v^(i)^-1 (z - m_i(t))
   where m_i(t) = x_i - y_i(t) is agent i's own noisy local estimate of z(t).
6. Solve this consensus problem in a fully DISTRIBUTED way with Distributed
   Gradient Descent (DGD): each agent only ever talks to its graph
   neighbours and only ever uses its own f_i.
7. Plot: communication graph, true vs. estimated 3-D trajectory, per-axis
   estimation error e(t) = z_hat(t) - z(t), and the DGD inner-loop
   convergence (consensus disagreement + objective value) at a
   representative time step.

HOW TO RUN
----------
    python Q2_Distributed_State_Estimation.py

OUTPUTS (written to ./figures/)
--------------------------------
    figures/q2_communication_graph.png
    figures/q2_true_and_estimated_trajectory.png
    figures/q2_estimation_error.png
    figures/q2_convergence.png
"""

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3-D projection)

# -----------------------------------------------------------------------
# 0. Reproducibility and output folder
# -----------------------------------------------------------------------
SEED = 7
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# 1. Problem parameters (all "assumed and fixed" as invited by the problem)
# -----------------------------------------------------------------------
N = 8                      # number of drones (< 20, as required)
P_ER = 0.4                 # Erdos-Renyi edge probability
T_MAX = 40                 # number of discrete time steps to simulate
DIM = 3                    # z(t) in R^3

# Intruder prior z(0) ~ N(z0_bar, Sigma0)
z0_bar = np.array([0.0, 0.0, 0.0])
Sigma0 = 0.5 * np.eye(DIM)

# Process noise covariance for the random walk z(t+1) = z(t) + w
Sigma_w = 0.05 * np.eye(DIM)

# Sensor locations x_i in R^3 (fixed, randomly placed around the origin)
sensor_locations = rng.uniform(-5, 5, size=(N, DIM))

# Per-sensor measurement noise covariance Sigma_v^(i) (each sensor has its
# own, generally different, noise level -> heterogeneous sensor quality)
sensor_noise_scale = rng.uniform(0.1, 0.6, size=N)
Sigma_v = [scale * np.eye(DIM) for scale in sensor_noise_scale]
Sigma_v_inv = [np.linalg.inv(S) for S in Sigma_v]

# -----------------------------------------------------------------------
# 2. Communication graph: connected Erdos-Renyi, contains a spanning tree
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
      f"connected={nx.is_connected(G)} (spanning tree exists <=> connected)")

# Metropolis-Hastings weights -> a symmetric, doubly-stochastic consensus
# matrix W (row/col sums = 1), the standard choice for distributed
# averaging / gradient algorithms on a fixed graph.
degree = dict(G.degree())
Wmat = np.zeros((N, N))
for i, j in G.edges():
    Wmat[i, j] = 1.0 / (1 + max(degree[i], degree[j]))
    Wmat[j, i] = Wmat[i, j]
for i in range(N):
    Wmat[i, i] = 1.0 - Wmat[i, :].sum()
assert np.allclose(Wmat.sum(axis=0), 1.0) and np.allclose(Wmat.sum(axis=1), 1.0)

pos_graph = nx.spring_layout(G, seed=SEED)
plt.figure(figsize=(6, 6))
nx.draw(G, pos_graph, with_labels=True, node_color="lightgreen",
        edge_color="gray", node_size=500, font_size=9)
plt.title(f"Q2: Drone Communication Graph (N={N} drones)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "q2_communication_graph.png"), dpi=150)
plt.close()

# -----------------------------------------------------------------------
# 3. Simulate the true intruder trajectory (random walk) and measurements
# -----------------------------------------------------------------------
z_true = np.zeros((T_MAX + 1, DIM))
z_true[0] = rng.multivariate_normal(z0_bar, Sigma0)
for t in range(T_MAX):
    w = rng.multivariate_normal(np.zeros(DIM), Sigma_w)
    z_true[t + 1] = z_true[t] + w

# y_i(t) = x_i - z(t) + v_i  ->  local noisy estimate m_i(t) = x_i - y_i(t) = z(t) - v_i
y_meas = np.zeros((T_MAX + 1, N, DIM))
for t in range(T_MAX + 1):
    for i in range(N):
        v_i = rng.multivariate_normal(np.zeros(DIM), Sigma_v[i])
        y_meas[t, i] = sensor_locations[i] - z_true[t] + v_i

# -----------------------------------------------------------------------
# 4. Distributed Gradient Descent (DGD) for the consensus optimisation
#
#    Local objective:  f_i(z) = 1/2 (z - m_i)^T Sigma_v^(i)^-1 (z - m_i)
#    Global problem :  min_z  sum_i f_i(z)
#    Gradient       :  grad f_i(z) = Sigma_v^(i)^-1 (z - m_i)
#
#    DGD update (each agent i, using ONLY neighbour information via W):
#       z_i(k+1) = sum_{j} W_ij z_j(k)  -  alpha * grad f_i(z_i(k))
#
#    We run K_INNER DGD iterations at every outer time step t (warm-started
#    from the previous time step's consensus estimate), which fuses that
#    time step's N noisy local measurements into a distributed estimate of
#    z(t). This turns the (memoryless) weighted-least-squares fusion at
#    each t into a simple recursive distributed filter across t = 0..Tmax.
# -----------------------------------------------------------------------
ALPHA = 0.3          # DGD step size
K_INNER = 60         # inner consensus/gradient iterations per outer time step

z_hat_agents = np.tile(z0_bar, (N, 1))   # each agent's local estimate, init at prior mean
z_hat_history = np.zeros((T_MAX + 1, DIM))          # consensus (averaged) estimate over time
inner_disagreement_example = None                     # convergence trace saved for one t
inner_objective_example = None

def local_measurement_targets(t):
    """m_i(t) = x_i - y_i(t), agent i's own noisy point estimate of z(t)."""
    return sensor_locations - y_meas[t]

for t in range(T_MAX + 1):
    m_t = local_measurement_targets(t)     # (N, DIM)
    disagreement_trace = []
    objective_trace = []
    for k in range(K_INNER):
        # --- consensus step (each agent mixes with its graph neighbours) ---
        z_mixed = Wmat @ z_hat_agents
        # --- local gradient-descent step (each agent uses only its own f_i) ---
        grad = np.array([Sigma_v_inv[i] @ (z_mixed[i] - m_t[i]) for i in range(N)])
        z_hat_agents = z_mixed - ALPHA * grad

        avg_z = z_hat_agents.mean(axis=0)
        disagreement = np.linalg.norm(z_hat_agents - avg_z)
        objective = sum(
            0.5 * (z_hat_agents[i] - m_t[i]) @ Sigma_v_inv[i] @ (z_hat_agents[i] - m_t[i])
            for i in range(N)
        )
        disagreement_trace.append(disagreement)
        objective_trace.append(objective)

    z_hat_history[t] = z_hat_agents.mean(axis=0)

    if t == T_MAX // 2:   # keep one representative inner-loop trace for plotting
        inner_disagreement_example = disagreement_trace
        inner_objective_example = objective_trace

estimation_error = z_hat_history - z_true    # e(t) = z_hat(t) - z(t)
rmse_per_axis = np.sqrt(np.mean(estimation_error ** 2, axis=0))
rmse_overall = np.sqrt(np.mean(np.sum(estimation_error ** 2, axis=1)))
print(f"Per-axis RMSE (x,y,z) over t=0..{T_MAX}: {rmse_per_axis}")
print(f"Overall RMSE ||e(t)||: {rmse_overall:.4f}")

# -----------------------------------------------------------------------
# 5. Plot: true vs. estimated 3-D trajectory (with fixed sensor locations)
# -----------------------------------------------------------------------
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot(z_true[:, 0], z_true[:, 1], z_true[:, 2], "-o", color="tab:red",
        markersize=3, label="True intruder trajectory $z(t)$")
ax.plot(z_hat_history[:, 0], z_hat_history[:, 1], z_hat_history[:, 2], "-^",
        color="tab:blue", markersize=3, label=r"Distributed estimate $\hat{z}(t)$")
ax.scatter(sensor_locations[:, 0], sensor_locations[:, 1], sensor_locations[:, 2],
           color="black", marker="s", s=50, label="Drone/sensor locations $x_i$")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title("Q2: True vs. Distributed Estimate of the Intruder Trajectory")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "q2_true_and_estimated_trajectory.png"), dpi=150)
plt.close()

# -----------------------------------------------------------------------
# 6. Plot: estimation error e(t) = z_hat(t) - z(t), per axis and norm
# -----------------------------------------------------------------------
fig, axs = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
axs[0].plot(estimation_error[:, 0], label="$e_x(t)$")
axs[0].plot(estimation_error[:, 1], label="$e_y(t)$")
axs[0].plot(estimation_error[:, 2], label="$e_z(t)$")
axs[0].axhline(0, color="gray", linewidth=0.7)
axs[0].set_ylabel("Per-axis error")
axs[0].set_title("Q2: Estimation Error $e(t) = \\hat{z}(t) - z(t)$")
axs[0].legend()
axs[0].grid(alpha=0.3)

axs[1].plot(np.linalg.norm(estimation_error, axis=1), color="tab:purple")
axs[1].set_xlabel("Time step $t$")
axs[1].set_ylabel(r"$\|e(t)\|_2$")
axs[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "q2_estimation_error.png"), dpi=150)
plt.close()

# -----------------------------------------------------------------------
# 7. Plot: DGD inner-loop convergence at a representative time step
# -----------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
axs[0].plot(inner_disagreement_example, color="tab:orange")
axs[0].set_yscale("log")
axs[0].set_xlabel("DGD inner iteration $k$")
axs[0].set_ylabel(r"Consensus disagreement $\|z_i(k)-\bar z(k)\|$ (log scale)")
axs[0].set_title(f"Consensus disagreement (t = {T_MAX // 2})")
axs[0].grid(alpha=0.3)

axs[1].plot(inner_objective_example, color="tab:green")
axs[1].set_xlabel("DGD inner iteration $k$")
axs[1].set_ylabel(r"$\sum_i f_i(z_i(k))$")
axs[1].set_title(f"Objective value (t = {T_MAX // 2})")
axs[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "q2_convergence.png"), dpi=150)
plt.close()

print("\nQ2 complete. All figures saved under ./figures/")
