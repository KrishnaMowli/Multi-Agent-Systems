"""AI3403 - Problem 1 (Q4): Independent Q-Learning on the two-agent 5x5 cooperative navigation gridworld."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- parameters ----------------
S = 2718                      # chosen random positive integer < 10^4
k = S % 5 + 1                 # k = mod(S,5)+1
G1, G2 = (1, k), (5, 6 - k)   # landmarks
START = ((1, 1), (5, 5))
ALPHA, GAMMA = 0.2, 0.95
EPS0, EPS_MIN, DECAY = 1.0, 0.05, 0.995
N_EP, H, WIN = 2000, 30, 50
MOVES = [(0, 1), (0, -1), (-1, 0), (1, 0), (0, 0)]          # Up, Down, Left, Right, Stay
NAMES = ["Up", "Down", "Left", "Right", "Stay"]
rng = np.random.default_rng(S)

def move(p, a):
    x, y = p[0] + MOVES[a][0], p[1] + MOVES[a][1]
    return (x, y) if (1 <= x <= 5 and 1 <= y <= 5) else p      # wall: stay in place

def covered(p1, p2):
    return (p1 == G1 and p2 == G2) or (p1 == G2 and p2 == G1)  # agents may swap roles

def sidx(p1, p2):
    return (((p1[0]-1)*5 + p1[1]-1) * 5 + p2[0]-1) * 5 + p2[1]-1

def env_step(p1, p2, a1, a2):
    n1, n2 = move(p1, a1), move(p2, a2)
    if covered(n1, n2):
        return n1, n2, 20 - 1, True
    if n1 == n2:
        return n1, n2, -5, False
    return n1, n2, -1, False

Q = np.zeros((2, 625, 5))     # Q[i][s, a_i], s = joint position

def act(i, s, eps):
    if rng.random() < eps:
        return int(rng.integers(5))
    q = Q[i, s]
    return int(rng.choice(np.flatnonzero(q == q.max())))        # greedy, random tie-break

# ---------------- training ----------------
eps, returns, succ = EPS0, [], []
for ep in range(N_EP):
    p1, p2 = START
    G, ok = 0.0, 0
    for t in range(H):
        s = sidx(p1, p2)
        a = (act(0, s, eps), act(1, s, eps))
        n1, n2, r, done = env_step(p1, p2, *a)
        s2 = sidx(n1, n2)
        for i in range(2):                                       # independent updates, shared reward
            target = r + (0.0 if done else GAMMA * Q[i, s2].max())
            Q[i, s, a[i]] += ALPHA * (target - Q[i, s, a[i]])
        G += r; p1, p2 = n1, n2
        if done:
            ok = 1; break
    returns.append(G); succ.append(ok)
    eps = max(EPS_MIN, DECAY * eps)

ma = lambda x: np.convolve(x, np.ones(WIN)/WIN, mode="valid")
np.save("figures/iql_returns.npy", np.array(returns)); np.save("figures/iql_success.npy", np.array(succ))

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
for name, y, lab in [("iql_return", ma(returns), "Moving-average return"),
                     ("iql_success", ma(succ), "Moving-average success rate")]:
    fig, ax = plt.subplots(figsize=(3.45, 2.5))
    ax.plot(np.arange(WIN, N_EP + 1), y, color="#1f4e79", lw=1.3)
    ax.set_xlabel("Episode"); ax.set_ylabel(lab); ax.grid(alpha=0.3)
    ax.set_title(lab + " (window 50)", fontsize=9)
    fig.subplots_adjust(left=0.2, right=0.97, bottom=0.18, top=0.9); fig.savefig(f"figures/{name}.pdf"); plt.close(fig)

# ---------------- greedy evaluation ----------------
print(f"S = {S}, k = {k}, g1 = {G1}, g2 = {G2}")
print(f"Final MA(50) return = {ma(returns)[-1]:.2f}, final MA(50) success = {ma(succ)[-1]:.2f}")
out = []
for e in range(1, 4):
    p1, p2 = START
    traj, R = [(p1, p2)], 0
    for t in range(H):
        s = sidx(p1, p2)
        a = (act(0, s, 0.0), act(1, s, 0.0))                     # eps = 0 -> greedy
        p1, p2, r, done = env_step(p1, p2, *a)
        R += r; traj.append((p1, p2))
        out.append((e, t + 1, NAMES[a[0]], NAMES[a[1]]))
        if done: break
    print(f"Greedy episode {e} (return = {R}, steps = {len(traj)-1})")
    print("  Agent 1: " + " -> ".join(f"({a},{b})" for (a, b), _ in traj))
    print("  Agent 2: " + " -> ".join(f"({a},{b})" for _, (a, b) in traj))
