"""AI3403 - Problem 2: numerical verification of the analytical results (Q4, Q5)."""
import numpy as np
from scipy.optimize import linprog

p0 = np.array([[0,0],[1,0],[1,1],[0,1]], float)
E = [(0,1),(1,2),(2,3),(3,0)]
d = {e: p0[e[0]] - p0[e[1]] for e in E}
g1 = p0 + np.array([0,2]); g2 = np.array([[0,2],[2,2],[2,3],[0,3]], float)
H, UMAX = 6, 0.5

# --- Q5: explicit controls for phi_1
u = np.zeros((4, H, 2)); u[:, :4, 1] = 0.5
P = np.zeros((4, H+1, 2)); P[:, 0] = p0
for l in range(H): P[:, l+1] = P[:, l] + u[:, l]
print("max |u|_inf =", np.abs(u).max())
print(" l   p1           p2           p3           p4           max Fij-dev   max goal-dev")
for l in range(H+1):
    dev = max(np.abs((P[i, l]-P[j, l]) - d[(i, j)]).max() for i, j in E)
    gd = np.abs(P[:, l]-g1).max()
    print(l, *[str(tuple(P[i, l])) for i in range(4)], round(dev,3), round(gd,3))

# --- Q4: LP feasibility of the joint problem for each target time t* (phi_1 and phi_2)
def feasible(goal, tstar):
    # variables: u[i,l,c] flattened; positions affine in u
    n = 4*H*2
    idx = lambda i,l,c: (i*H + l)*2 + c
    def pos(i, l):                 # returns (A, b) with p_i[l]_c = A_c @ x + b_c
        A = np.zeros((2, n)); 
        for m in range(l):
            for c in range(2): A[c, idx(i,m,c)] = 1
        return A, p0[i]
    A_ub, b_ub = [], []
    for l in range(0, H+1):
        for (i, j) in E:
            Ai, bi = pos(i, l); Aj, bj = pos(j, l)
            for c in range(2):
                row = Ai[c] - Aj[c]; rhs = d[(i, j)][c] - (bi[c] - bj[c])
                A_ub.append(row);  b_ub.append(0.2 + rhs)     # (pi-pj-d)_c <= 0.2
                A_ub.append(-row); b_ub.append(0.2 - rhs)     # >= -0.2
    for i in range(4):
        Ai, bi = pos(i, tstar)
        for c in range(2):
            A_ub.append(Ai[c]);  b_ub.append(goal[i][c] + 0.1 - bi[c])
            A_ub.append(-Ai[c]); b_ub.append(-(goal[i][c] - 0.1 - bi[c]))
    r = linprog(np.zeros(n), A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=[(-UMAX, UMAX)]*n, method="highs")
    return r.status == 0
for name, goal in [("phi_1", g1), ("phi_2", g2)]:
    print(name, {t: feasible(goal, t) for t in range(0, H+1)})
