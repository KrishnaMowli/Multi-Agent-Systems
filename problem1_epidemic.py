"""
Problem 1 (Questions 1-5): Data-driven SIR modelling
Author: G V S S Krishna Mowli (EE25MTECH11019)

Run:  python problem1_epidemic.py
Needs epidemic_data.csv in the same folder. Figures go to ./figures
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# Question 1: S[k], I[k], R[k] from the individual-level data
# ------------------------------------------------------------------
data = pd.read_csv(HERE / "epidemic_data.csv")
counts = data.groupby(["day", "state"]).size().unstack(fill_value=0)
S = counts["S"].to_numpy()
I = counts["I"].to_numpy()
R = counts["R"].to_numpy()
days = counts.index.to_numpy()
N = int(S[0] + I[0] + R[0])

plt.figure(figsize=(6.4, 3.8))
plt.plot(days, S, label="S[k]", color="tab:blue")
plt.plot(days, I, label="I[k]", color="tab:red")
plt.plot(days, R, label="R[k]", color="tab:green")
plt.xlabel("Day k")
plt.ylabel("Number of individuals")
plt.title("Susceptible, infected and recovered counts (data)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "fig_data_sir.png", dpi=200)
plt.close()

I_max = int(I.max())
T_peak = int(days[I.argmax()])
ever_infected = int(I[-1] + R[-1])          # = N - S[end]
frac_infected = ever_infected / N

# ------------------------------------------------------------------
# Question 2: recovery rate gamma (forward Euler, dt = 1 day)
# ------------------------------------------------------------------
dt = 1.0
k_all = np.arange(len(days) - 1)            # k = 0 ... last-1 (k+1 must exist)
K = k_all[I[k_all] > 0]
gamma_k = (R[K + 1] - R[K]) / (I[K] * dt)
gamma_hat = gamma_k.mean()
mean_duration = 1.0 / gamma_hat

# ------------------------------------------------------------------
# Question 3: transmission rate beta and reproduction number R0
# ------------------------------------------------------------------
Kb = k_all[(S[k_all] * I[k_all]) > 0]
beta_k = (S[Kb] - S[Kb + 1]) / (S[Kb] * I[Kb] * dt)
beta_hat = beta_k.mean()
R0 = beta_hat * S[0] / gamma_hat
R0_N = beta_hat * N / gamma_hat             # same thing with S[0] replaced by N

# ------------------------------------------------------------------
# Question 4: simulate the continuous-time SIR model
# ------------------------------------------------------------------
def sir(t, y, beta, gamma):
    s, i, r = y
    return [-beta * s * i, beta * s * i - gamma * i, gamma * i]

sol = solve_ivp(sir, (0, days[-1]), [S[0], I[0], R[0]], args=(beta_hat, gamma_hat),
                t_eval=np.linspace(0, days[-1], 1001), rtol=1e-8, atol=1e-8)
t_sim, S_sim, I_sim, R_sim = sol.t, sol.y[0], sol.y[1], sol.y[2]

# the model evaluated on the same daily grid as the data
sol_d = solve_ivp(sir, (0, days[-1]), [S[0], I[0], R[0]], args=(beta_hat, gamma_hat),
                  t_eval=days.astype(float), rtol=1e-8, atol=1e-8)
S_d, I_d, R_d = sol_d.y

rmse_S = np.sqrt(np.mean((S_d - S) ** 2))
rmse_I = np.sqrt(np.mean((I_d - I) ** 2))
rmse_R = np.sqrt(np.mean((R_d - R) ** 2))
I_max_model = float(I_sim.max())
T_peak_model = float(t_sim[I_sim.argmax()])
S_end_model = float(S_sim[-1])

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(t_sim, S_sim, "b-", label="S(t) model")
ax[0].plot(t_sim, I_sim, "r-", label="I(t) model")
ax[0].plot(t_sim, R_sim, "g-", label="R(t) model")
ax[0].set_title("Simulated SIR model")
ax[1].plot(t_sim, S_sim, "b-", lw=1.2, label="S model")
ax[1].plot(t_sim, I_sim, "r-", lw=1.2, label="I model")
ax[1].plot(t_sim, R_sim, "g-", lw=1.2, label="R model")
ax[1].plot(days, S, "b.", ms=3, alpha=0.6, label="S data")
ax[1].plot(days, I, "r.", ms=3, alpha=0.6, label="I data")
ax[1].plot(days, R, "g.", ms=3, alpha=0.6, label="R data")
ax[1].set_title("Model compared with data")
for a in ax:
    a.set_xlabel("Time (days)")
    a.set_ylabel("Number of individuals")
    a.grid(alpha=0.3)
ax[0].legend()
ax[1].legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "fig_sir_model.png", dpi=200)
plt.close()

# ------------------------------------------------------------------
# Print and save the numbers used in the report
# ------------------------------------------------------------------
res = dict(N=N, S0=int(S[0]), I0=int(I[0]), R0_count=int(R[0]),
           I_max=I_max, T_peak=T_peak, ever_infected=ever_infected,
           frac_infected=frac_infected, S_end=int(S[-1]),
           gamma_hat=gamma_hat, mean_duration=mean_duration, num_K=int(len(K)),
           beta_hat=beta_hat, num_Kbeta=int(len(Kb)), R0_hat=R0, R0_with_N=R0_N,
           I_max_model=I_max_model, T_peak_model=T_peak_model,
           S_end_model=S_end_model, rmse_S=rmse_S, rmse_I=rmse_I, rmse_R=rmse_R)
for key, val in res.items():
    print(f"{key:15s} {val}")
with open(HERE / "results_epidemic.json", "w") as f:
    json.dump(res, f, indent=2)
