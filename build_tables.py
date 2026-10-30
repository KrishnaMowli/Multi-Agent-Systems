import json, numpy as np
B = json.load(open("figures/baseline_stats.json")); F = json.load(open("figures/final_stats.json"))
TH = {"S1": 0.75, "S2": 1.00, "S3": 1.25, "S4": 1.50}
def pm(v, nd=0, scale=1.0):
    m, s = v[0]/scale, v[1]/scale
    t = f"{m:,.{nd}f}".replace(",", "{,}"); u = f"{s:,.{nd}f}".replace(",", "{,}")
    if t.startswith("-"): t = "-" + t[1:]
    return f"${t}\\pm{u}$"

# ---- baseline table (Part I, e)
L = [r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
     r"Scenario & $\lambda\!=\!16\theta$ & $\rho_3\!=\!\lambda/24$ & Received & Completed & Same-day (\%) & Delay (days) & Unfinished at $d\!=\!365$ \\", r"\midrule"]
for s, th in TH.items():
    r = B[s]
    L.append(f"{s} ($\\theta\\!=\\!{th:.2f}$) & {16*th:.0f} & {16*th/24:.2f} & {pm(r['received'])} & {pm(r['completed'])} & "
             f"${100*r['same_frac'][0]:.1f}$ & ${r['delay'][0]:.2f}$ & {pm(r['unfinished'],1)} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open("tables/baseline_table.tex", "w").write("\n".join(L))

# ---- final assessment table (Part IV)
rows = [("(a) Applications received", lambda r: pm(r["received"])),
        ("(b) Applications completed", lambda r: pm(r["completed"])),
        ("(c) Same-day completion fraction (\\%)", lambda r: f"${100*r['same_frac'][0]:.1f}\\pm{100*r['same_frac'][1]:.1f}$"),
        ("(d) Average completion delay (days)", lambda r: f"${r['delay'][0]:.2f}\\pm{r['delay'][1]:.2f}$"),
        ("(e) Unfinished at end of day 365", lambda r: pm(r["unfinished"],1)),
        ("(f) Cumulative reward $R_{365}$", lambda r: pm(r["R365"])),
        ("(g) Final public satisfaction $q_{365}$", lambda r: f"${r['q365'][0]:.2f}\\pm{r['q365'][1]:.2f}$"),
        ("(h) Avg.\\ staffing at $D_1,\\dots,D_5$", lambda r: "3, 3, 3, 3, 3")]
L = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
     r"Metric & S1 ($\theta\!=\!0.75$) & S2 ($\theta\!=\!1.00$) & S3 ($\theta\!=\!1.25$) & S4 ($\theta\!=\!1.50$) \\", r"\midrule"]
for name, fn in rows:
    L.append(name + " & " + " & ".join(fn(F[s]) for s in TH) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open("tables/final_table.tex", "w").write("\n".join(L))
print(open("tables/baseline_table.tex").read()); print(open("tables/final_table.tex").read())
for s in TH: print(s, "mean lambda", F[s]["mean_lambda"][0], "util", F[s]["mean_lambda"][0]/24)
