"""
==============================================================================
Q1_Formation_Control_Krishna.py
==============================================================================
AI3403: Multi-Agent Systems -- Assignment 3, Problem 1
IIT Hyderabad

WHAT THIS FILE DOES
--------------------
1. Generates a connected Erdos-Renyi communication graph over N = 20 agents.
2. Assigns each agent a random initial position in R^2.
3. Uses a distributed, graph-based formation-control law to steer the agents
   so that, collectively, their positions spell out "KRISHNA", letter by
   letter: K -> R -> I -> S -> H -> N -> A.
4. Animates the whole letter-formation sequence and saves it as an MP4 (or a
   GIF if an MP4 encoder is not available on the machine that runs this).
5. Saves the required static figures (communication graph, formation
   snapshots for every letter, and a formation-error convergence curve).

HOW TO RUN
----------
    python Q1_Formation_Control_Krishna.py

OUTPUTS (all written to ./figures/, created automatically if missing)
-----------------------------------------------------------------------
    figures/q1_krishna_communication_graph.png
    figures/q1_krishna_formation_snapshots.png
    figures/q1_krishna_convergence.png
    figures/q1_krishna_formation_animation.mp4  (falls back to .gif if no ffmpeg)
"""

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from scipy.optimize import linear_sum_assignment

# -----------------------------------------------------------------------
# 0. Reproducibility and output folder
# -----------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)
OUT_PREFIX = "q1_krishna"   # namespaces every output file for this name

# -----------------------------------------------------------------------
# 1. Name to simulate
# -----------------------------------------------------------------------
NAME_TO_SIMULATE = "KRISHNA"
print(f"Simulating letter formation sequence for: {NAME_TO_SIMULATE}")

# -----------------------------------------------------------------------
# 2. Problem parameters
# -----------------------------------------------------------------------
N = 20                        # number of agents (fixed by the assignment)
P_ER = 0.25                   # Erdos-Renyi edge probability
FORMATION_HALF_WIDTH = 5.0    # controls the physical size of each letter
K_GAIN = 1.0                  # formation control gain
SAFETY_FACTOR = 0.6           # fraction of the stability limit used for dt
STEPS_PER_LETTER = 250        # discrete-time steps allotted to each letter

# -----------------------------------------------------------------------
# 3. Build a CONNECTED Erdos-Renyi communication graph
# -----------------------------------------------------------------------
def make_connected_er_graph(n, p, seed):
    """Keep resampling an Erdos-Renyi graph until it is connected."""
    trial = 0
    while True:
        g = nx.erdos_renyi_graph(n, p, seed=seed + trial)
        if nx.is_connected(g):
            return g
        trial += 1

G = make_connected_er_graph(N, P_ER, SEED)
L = nx.laplacian_matrix(G).toarray().astype(float)   # graph Laplacian
print(f"Erdos-Renyi graph: N={N}, p={P_ER}, |E|={G.number_of_edges()}, "
      f"connected={nx.is_connected(G)}")

# Save the communication graph figure
pos_graph = nx.spring_layout(G, seed=SEED)
plt.figure(figsize=(6, 6))
nx.draw(G, pos_graph, with_labels=True, node_color="skyblue",
        edge_color="gray", node_size=450, font_size=8)
plt.title(f"Q1: Erdos-Renyi Communication Graph (N={N}, p={P_ER})")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, f"{OUT_PREFIX}_communication_graph.png"), dpi=150)
plt.close()

# -----------------------------------------------------------------------
# 4. Letter target-point generation using the font's own vector outline
#
#    Each letter glyph is traced by matplotlib's TextPath. Points are
#    sampled at equal arc-length spacing ALONG the outline (proportionally
#    across multiple contours, e.g. the counters/holes in 'A', 'D', 'O').
#    Sampling along the outline (rather than filling the interior) is what
#    keeps the letter recognisable with only N = 20 agents: the eye can
#    trace the boundary even from a sparse set of points, especially once
#    we also draw a faint connecting line between geometrically adjacent
#    agents (done in the plotting/animation code below).
# -----------------------------------------------------------------------
def sample_letter_outline(letter, n_points, half_width=FORMATION_HALF_WIDTH):
    """
    Return:
        pts         : (n_points, 2) array, centred at the origin, scaled to
                      half_width, sampled at equal arc-length spacing along
                      the glyph outline.
        contour_id  : (n_points,) array labelling which contour (sub-path)
                      each point belongs to, in traversal order. Used only
                      for drawing a "connect-the-dots" outline in the plots.
    """
    fp = FontProperties(family="DejaVu Sans", weight="bold")
    tp = TextPath((0, 0), letter, size=100, prop=fp)
    polygons = [p for p in tp.to_polygons() if len(p) > 1]

    lengths = np.array([
        np.sum(np.linalg.norm(np.diff(poly, axis=0), axis=1)) for poly in polygons
    ])
    n_per_contour = np.maximum(2, np.round(n_points * lengths / lengths.sum()).astype(int))
    while n_per_contour.sum() != n_points:
        diff = n_points - n_per_contour.sum()
        idx = np.argmax(lengths) if diff > 0 else np.argmax(n_per_contour)
        n_per_contour[idx] += 1 if diff > 0 else -1

    all_pts, contour_id = [], []
    for c_idx, (poly, npts) in enumerate(zip(polygons, n_per_contour)):
        seg = np.diff(poly, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        cum_len = np.concatenate([[0], np.cumsum(seg_len)])
        sample_at = np.linspace(0, cum_len[-1], npts, endpoint=False)
        xs = np.interp(sample_at, cum_len, poly[:, 0])
        ys = np.interp(sample_at, cum_len, poly[:, 1])
        all_pts.append(np.column_stack([xs, ys]))
        contour_id += [c_idx] * npts
    pts = np.vstack(all_pts)[:n_points]
    contour_id = np.array(contour_id[:n_points])

    pts = pts - pts.mean(axis=0)
    span = np.max(np.abs(pts)) if np.max(np.abs(pts)) > 0 else 1.0
    pts = pts * (half_width / span)
    return pts, contour_id

letters = list(NAME_TO_SIMULATE)
raw_targets = {Lc: sample_letter_outline(Lc, N) for Lc in set(letters)}

# -----------------------------------------------------------------------
# 5. Re-order target points between consecutive letters (and from the
#    initial positions to the first letter) via optimal assignment, so that
#    agents follow short, non-crossing paths -> a visually clean animation.
#    We also keep track of, for every letter, which AGENT ends up at each
#    position along each contour, purely so we can draw a faithful
#    "connect-the-dots" outline of the letter in the figures/animation.
# -----------------------------------------------------------------------
x0 = np.random.uniform(-8, 8, size=(N, 2))   # random initial positions

def reorder_by_assignment(reference_pts, target_pts):
    cost = np.linalg.norm(reference_pts[:, None, :] - target_pts[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    return target_pts[col_ind], col_ind

ordered_targets = []          # ordered_targets[k] : (N,2) target for letters[k], indexed by agent
contour_agent_order = []      # contour_agent_order[k] : list of agent-index arrays (one per contour)
prev_pts = x0
for Lch in letters:
    raw_pts, raw_cid = raw_targets[Lch]
    tgt, col_ind = reorder_by_assignment(prev_pts, raw_pts)
    ordered_targets.append(tgt)

    # invert the permutation: inv[original_point_index] = agent_index
    inv = np.empty(N, dtype=int)
    inv[col_ind] = np.arange(N)
    per_contour_agents = [inv[np.where(raw_cid == c)[0]] for c in np.unique(raw_cid)]
    contour_agent_order.append(per_contour_agents)

    prev_pts = tgt

# -----------------------------------------------------------------------
# 6. Formation control law (Laplacian form)
#
#    Continuous time:  xdot_i = -k * sum_{j in N_i} [ (x_i - x_j) - (h_i - h_j) ]
#    where h_i is agent i's desired position within the current letter.
#
#    Writing e = x - h, the right-hand side is exactly -k * (L e)_i, so in
#    matrix form:           xdot = -k * L (x - h)
#
#    Discretising with step dt (forward Euler):
#           x[t+1] = x[t] - dt*k*L*(x[t]-h) = (I - dt*k*L) x[t] + dt*k*L*h
#
#    This is a stable linear recursion whenever  dt*k*lambda_max(L) < 2.
# -----------------------------------------------------------------------
eigvals = np.linalg.eigvalsh(L)
lambda_max = eigvals[-1]
dt = SAFETY_FACTOR * 2.0 / (K_GAIN * lambda_max)
print(f"lambda_max(L) = {lambda_max:.4f}  ->  using dt = {dt:.4f} (stability-safe)")

I_N = np.eye(N)
A_step = I_N - dt * K_GAIN * L    # x[t+1] = A_step @ x[t] + dt*K_GAIN*L @ h

# -----------------------------------------------------------------------
# 7. Run the simulation over the whole letter sequence
# -----------------------------------------------------------------------
trajectory = [x0.copy()]
letter_of_step = []                 # which letter each stored frame belongs to
formation_error_history = []        # ||x - h|| history (mean-removed) for convergence plot
phase_boundaries = [0]              # frame index where each letter's phase starts

x = x0.copy()
for Lch, h in zip(letters, ordered_targets):
    for step in range(STEPS_PER_LETTER):
        x = A_step @ x + dt * K_GAIN * (L @ h)
        trajectory.append(x.copy())
        letter_of_step.append(Lch)
        e = (x - h)
        e_centered = e - e.mean(axis=0)     # formation error, common translation removed
        formation_error_history.append(np.linalg.norm(e_centered))
    phase_boundaries.append(len(trajectory) - 1)

trajectory = np.array(trajectory)   # (T+1, N, 2)
print(f"Total simulated frames: {trajectory.shape[0]} across {len(letters)} letters")
print(f"Final formation error (last letter): {formation_error_history[-1]:.6f}")

# -----------------------------------------------------------------------
# 8. Save the formation-error convergence plot
# -----------------------------------------------------------------------
plt.figure(figsize=(8, 4.5))
plt.plot(formation_error_history, color="tab:blue")
for b in phase_boundaries[1:-1]:
    plt.axvline(b, color="gray", linestyle="--", linewidth=0.7)
plt.xlabel("Discrete time step (concatenated across letters)")
plt.ylabel(r"Formation error $\|(x-h) - \mathrm{mean}(x-h)\|$")
plt.title("Q1: Formation-Control Convergence Across the Letter Sequence")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, f"{OUT_PREFIX}_convergence.png"), dpi=150)
plt.close()

# -----------------------------------------------------------------------
# helper: draw the connect-the-dots outline for a given letter/positions
# -----------------------------------------------------------------------
def draw_letter_outline(ax, positions, per_contour_agents, **line_kwargs):
    for agent_seq in per_contour_agents:
        loop = np.vstack([positions[agent_seq], positions[agent_seq[0]]])
        ax.plot(loop[:, 0], loop[:, 1], **line_kwargs)

# -----------------------------------------------------------------------
# 9. Save a snapshot grid: final converged formation for every letter
# -----------------------------------------------------------------------
n_letters = len(letters)
ncols = min(4, n_letters)
nrows = int(np.ceil(n_letters / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
axes = np.atleast_1d(axes).reshape(-1)
for i, (Lch, boundary) in enumerate(zip(letters, phase_boundaries[1:])):
    final_positions = trajectory[boundary]
    ax = axes[i]
    draw_letter_outline(ax, final_positions, contour_agent_order[i],
                         color="tab:blue", alpha=0.5, linewidth=1.2)
    ax.scatter(final_positions[:, 0], final_positions[:, 1], c="tab:blue", s=40, zorder=3)
    ax.set_title(f"Letter: {Lch}")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
for j in range(n_letters, len(axes)):
    axes[j].axis("off")
fig.suptitle(f"Q1: Converged Formations Spelling '{NAME_TO_SIMULATE}'")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, f"{OUT_PREFIX}_formation_snapshots.png"), dpi=150)
plt.close()

# -----------------------------------------------------------------------
# 10. Animate the full sequence and save as MP4 (falls back to GIF)
# -----------------------------------------------------------------------
FRAME_STRIDE = 3   # sub-sample frames so the video is a reasonable length
frames_idx = list(range(0, trajectory.shape[0], FRAME_STRIDE))
frame_letter_label = ["start"] + letter_of_step   # label for every RAW frame index
MAX_CONTOURS = max(len(c) for c in contour_agent_order)

fig_anim, ax_anim = plt.subplots(figsize=(6, 6))
all_pts = trajectory.reshape(-1, 2)
margin = 2.0
ax_anim.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
ax_anim.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
ax_anim.set_aspect("equal")
ax_anim.grid(alpha=0.3)
scat = ax_anim.scatter([], [], c="tab:blue", s=50, zorder=3)
outline_lines = [ax_anim.plot([], [], color="tab:blue", alpha=0.5, linewidth=1.2)[0]
                 for _ in range(MAX_CONTOURS)]
title_txt = ax_anim.set_title("")

def init_anim():
    scat.set_offsets(np.empty((0, 2)))
    for ln in outline_lines:
        ln.set_data([], [])
    return [scat, title_txt] + outline_lines

def update_anim(frame_number):
    idx = frames_idx[frame_number]
    positions = trajectory[idx]
    scat.set_offsets(positions)

    current_letter = frame_letter_label[idx]
    if current_letter == "start":
        letter_pos_in_seq = None
    else:
        letter_pos_in_seq = letter_of_step.index(current_letter) if False else None
    # find which phase (letter index) this raw frame idx belongs to
    phase_idx = None
    for k in range(len(letters)):
        if phase_boundaries[k] < idx <= phase_boundaries[k + 1]:
            phase_idx = k
            break

    if phase_idx is None:
        for ln in outline_lines:
            ln.set_data([], [])
    else:
        agent_orders = contour_agent_order[phase_idx]
        for j, ln in enumerate(outline_lines):
            if j < len(agent_orders):
                seq = agent_orders[j]
                loop_idx = np.append(seq, seq[0])
                ln.set_data(positions[loop_idx, 0], positions[loop_idx, 1])
            else:
                ln.set_data([], [])

    title_txt.set_text(
        f"Forming '{NAME_TO_SIMULATE}'  |  letter: {current_letter}  "
        f"|  frame {idx}/{trajectory.shape[0]-1}"
    )
    return [scat, title_txt] + outline_lines

anim = animation.FuncAnimation(
    fig_anim, update_anim, frames=len(frames_idx),
    init_func=init_anim, interval=40, blit=False
)

mp4_path = os.path.join(FIG_DIR, f"{OUT_PREFIX}_formation_animation.mp4")
gif_path = os.path.join(FIG_DIR, f"{OUT_PREFIX}_formation_animation.gif")
try:
    writer = animation.FFMpegWriter(fps=25, bitrate=1800)
    anim.save(mp4_path, writer=writer)
    print(f"Saved animation to {mp4_path}")
except Exception as e:
    print(f"FFMpeg writer unavailable ({e}); falling back to GIF.")
    writer = animation.PillowWriter(fps=20)
    anim.save(gif_path, writer=writer)
    print(f"Saved animation to {gif_path}")
plt.close(fig_anim)

print("\nQ1 complete. All figures saved under ./figures/")
