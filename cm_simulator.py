"""AI3403 - Problem 3: stochastic government-administration simulator (5 FIFO departments, 365 days)."""
import heapq
import numpy as np

P = [20, 30, 60, 40, 30]      # minutes per application at D1..D5
H_DAY, ND, DAYS = 480, 5, 365
CAP = [3 * H_DAY // t for t in P]   # apps/day with 3 officers: 72, 48, 24, 36, 48


def simulate(theta, seed, feedback=False, days=DAYS, m=(3, 3, 3, 3, 3)):
    """One 365-day run. Every application is tracked individually (arrival day, stage,
    remaining requirement, completion day). All new applications arrive at minute 0."""
    rng = np.random.default_rng(seed)
    arr, stage, rem, comp = [], [], [], []           # per-application records
    queue = [[] for _ in range(ND)]                  # FIFO queue of app ids at each department
    q_prev, p_prev = 0.0, 0.5
    out = {k: np.zeros(days) for k in ["A", "Csame", "Ccomp", "B", "r", "q", "lam"]}
    out["Q"] = np.zeros((days, ND))
    for d in range(1, days + 1):
        lam = 16 * theta * (1 + 0.5 * q_prev) if feedback else 16 * theta
        A = rng.poisson(lam)
        new = list(range(len(arr), len(arr) + A))
        for j in new:
            arr.append(d); stage.append(0); rem.append(P[0]); comp.append(0)
        inc = [(0, j) for j in new]                  # arrivals at D1 (ready at minute 0)
        c_same = c_all = 0
        for i in range(ND):
            seq = [(0, j) for j in queue[i]] + inc   # carry-over first, then new arrivals (FIFO)
            free = [0] * m[i]                        # officer free-times (min-heap)
            newq, done_here = [], []
            for idx, (ready, j) in enumerate(seq):
                f = heapq.heappop(free)
                start = ready if ready > f else f
                if start >= H_DAY:                   # nobody can start any more work today
                    newq.extend(x[1] for x in seq[idx:]); break
                if start + rem[j] <= H_DAY:          # completes at this department today
                    fin = start + rem[j]
                    heapq.heappush(free, fin)
                    if i == ND - 1:
                        rem[j] = 0; stage[j] = ND; comp[j] = d; c_all += 1
                        c_same += (arr[j] == d)
                    else:
                        rem[j] = P[i + 1]; stage[j] = i + 1; done_here.append((fin, j))
                else:                                # cut off at minute 480: effort preserved
                    rem[j] -= H_DAY - start
                    heapq.heappush(free, H_DAY); newq.append(j)
            queue[i] = newq
            inc = sorted(done_here)                  # FIFO into next department by finish time
        B = sum(len(x) for x in queue)
        p = c_same / A if A > 0 else p_prev
        q = 0.8 * q_prev + 0.2 * (2 * p - 1)
        out["A"][d-1], out["Csame"][d-1], out["Ccomp"][d-1], out["B"][d-1] = A, c_same, c_all, B
        out["r"][d-1] = 5 * c_same - 10 * B; out["q"][d-1] = q; out["lam"][d-1] = lam
        out["Q"][d-1] = [len(x) for x in queue]
        p_prev, q_prev = p, q
    out["R"] = np.cumsum(out["r"])
    arr, comp = np.array(arr), np.array(comp)
    done = comp > 0
    out["delay"] = float((comp[done] - arr[done]).mean()) if done.any() else 0.0
    out["records"] = (arr, np.array(stage), np.array(rem), comp)
    out["m"] = np.array(m, dtype=float)
    return out


def summarize(o):
    return dict(received=o["A"].sum(), completed=o["Ccomp"].sum(),
                same_frac=o["Csame"].sum() / o["A"].sum(), delay=o["delay"],
                unfinished=o["B"][-1], R365=o["R"][-1], q365=o["q"][-1])

if __name__ == "__main__":
    print("capacity per day:", CAP)
    for th in [0.75, 1.0, 1.25, 1.5]:
        o = simulate(th, 1)
        s = summarize(o)
        # consistency: received = completed + unfinished
        assert s["received"] == s["completed"] + s["unfinished"]
        print(th, {k: round(float(v), 2) for k, v in s.items()})
