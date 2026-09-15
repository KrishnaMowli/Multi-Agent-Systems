# AI3403 – Multi-Agent Systems: Assignment 3, Problem 1
## Formation Control — Spelling **KRISHNA** with a 20-Agent Multi-Robot System

**Author:** Krishna Mowli
**Roll No:** EE25MTECH11019
**Course:** AI3403 – Multi-Agent Systems, IIT Hyderabad

---

## 1. Problem Statement

Fix the number of agents at **N = 20**. Generate a **connected Erdos–Renyi graph**
`G(N, p)` for some `p ∈ (0,1)` to serve as the inter-agent communication topology.
Assign each agent a **random initial position** in `R^2`. Using a **formation
control strategy**, drive the agents so that their positions spell out the
author's commonly-used name, **letter by letter**, in sequence:

```
K → R → I → S → H → N → A
```

i.e. starting from the random initial positions, the swarm first converges to
the letter **K**, then re-forms into **R**, and so on until the final letter
**A**. The full sequence is recorded as an animation and the code + video are
submitted as a single public GitHub repository link.

---

## 2. Approach

**Communication graph.** An Erdos–Renyi graph is sampled with edge
probability `p = 0.25` and re-sampled (incrementing the seed) until it is
connected, guaranteeing the graph Laplacian `L` has exactly one zero
eigenvalue — the standard requirement for Laplacian-based consensus/formation
control to converge.

**Letter target generation.** Each letter is traced using Matplotlib's
`TextPath` on a bold DejaVu Sans glyph, then sampled at **20 points spaced at
equal arc length along the glyph outline** (proportionally split across
multiple sub-contours, so letters with enclosed loops — e.g. the counter in
`R` — are handled correctly). Tracing the outline rather than filling the
interior is what keeps each letter legible with only 20 agents.

**Inter-letter assignment.** Between the initial positions and the first
letter, and between every consecutive pair of letters, target points are
re-ordered using the **Hungarian algorithm**
(`scipy.optimize.linear_sum_assignment`) to minimize total agent travel
distance. This keeps agent paths short and largely non-crossing, giving a
visually clean transition animation.

**Control law.** A distributed Laplacian feedback law is used:

```
ẋᵢ = -k · Σ_{j ∈ Nᵢ} [ (xᵢ - xⱼ) - (hᵢ - hⱼ) ]      ⇒      ẋ = -k L (x - h)
```

where `h` is the current letter's target formation. This is discretized by
forward Euler with step size `dt` set to 60% of the theoretical stability
limit `dt < 2 / (k·λ_max(L))`, guaranteeing a stable linear recursion
`x[t+1] = (I - dt·k·L) x[t] + dt·k·L·h`.

**Simulation.** The controller is run for 250 discrete steps per letter,
sequentially, across the whole 7-letter sequence, with the converged
formation of each letter feeding in as the initial condition (after
re-assignment) for the next.

---

## 3. Repository Contents

```
.
├── Q1_Formation_Control_Krishna.py
├── figures/
│   ├── q1_krishna_communication_graph.png     # Erdos–Renyi topology, N=20
│   ├── q1_krishna_formation_snapshots.png     # converged formation per letter
│   ├── q1_krishna_convergence.png             # formation-error vs. time
│   └── q1_krishna_formation_animation.mp4     # full sequence animation (falls back to .gif)
└── README.md
```

---

## 4. Requirements

- Python 3.x
- `numpy`
- `networkx`
- `matplotlib`
- `scipy`
- `ffmpeg` (optional — enables MP4 export; if unavailable, the script falls
  back to a GIF via Pillow automatically)

Install with:

```bash
pip install numpy networkx matplotlib scipy
```

---

## 5. How to Run

```bash
python Q1_Formation_Control_Krishna.py
```

All figures and the animation are written automatically to `./figures/`
(created if it does not already exist).

---

## 6. Key Parameters

| Parameter | Value | Description |
|---|---|---|
| `N` | 20 | Number of agents (fixed by the assignment) |
| `P_ER` | 0.25 | Erdos–Renyi edge probability |
| `K_GAIN` | 1.0 | Formation control gain `k` |
| `SAFETY_FACTOR` | 0.6 | Fraction of the stability limit used for `dt` |
| `STEPS_PER_LETTER` | 250 | Discrete-time steps allotted per letter |
| `SEED` | 42 | Random seed, for reproducibility |

---

## 7. Output

- **Communication graph** — the connected Erdos–Renyi topology used for
  distributed control.
- **Formation snapshots** — final converged agent positions for every letter
  of `KRISHNA`.
- **Convergence plot** — the mean-removed formation error
  `‖(x-h) - mean(x-h)‖` over the whole concatenated sequence, with dashed
  lines marking letter transitions.
- **Animation** (`q1_krishna_formation_animation.mp4`) — the full letter-by-letter
  transition of the swarm from random initial positions through
  K → R → I → S → H → N → A.

---

## 8. Submission Note

Per the assignment instructions, this repository is made **public** and
contains both the source code and the recorded animation video; the GitHub
repository link is submitted as the complete solution to Problem 1.
