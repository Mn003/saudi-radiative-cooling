"""
reproduce_last_year.py
======================
Purpose
-------
Re-runs every numerical result reported in the Ibdaa 2026 Project Detail Form
(last year's Qalib) using the physics code that actually exists in the
repository, and prints a side-by-side comparison:

    reported value  |  value from current PRC.py  |  value from July-27 PRC.py

It does NOT change any model parameter to make results match. Its only job is
to show which reported numbers can be regenerated from code and which cannot.

Usage (from the repository root)
--------------------------------
    python reproduce_last_year.py                       # current engine only
    python reproduce_last_year.py --old path/to/old/PRC.py   # also July-27 engine

Outputs
-------
    Console report + reproduction_report.csv (one row per reported number).

"Match" rule: |code - reported| <= 0.02 degC (rounding tolerance of 2-decimal
reporting). Anything larger is flagged NOT REPRODUCED.
"""
import argparse
import ast
import csv
import os
import sys

import numpy as np
from scipy.optimize import fsolve

TOL = 0.02  # degC, rounding tolerance for 2-decimal reported values

# ---------------------------------------------------------------------------
# Numbers transcribed from last year's report (IBDAA Electronic Stage PDF).
# Every entry cites the table / section it came from.
# ---------------------------------------------------------------------------
# Table IV: (study, Tamb C, RH %, wind m/s, GHI W/m2, alpha, eps, exp dT, reported sim dT)
TABLE_IV = [
    ("Mandal et al. [1]",       27.0, 45, 1.8,  890, 0.04,  0.96,  -6.00, -5.78),
    ("Raman et al. [2]",        21.5, 40, 1.2,  850, 0.03,  0.96,  -4.90, -4.72),
    ("Zhai et al. [3]",         24.0, 22, 1.8, 1000, 0.04,  0.93,  -5.00, -5.24),
    ("Zhao et al. [4]",         25.5, 28, 1.5,  960, 0.03,  0.95,  -6.20, -5.91),
    ("Gentle & Smith [5]",      28.0, 55, 2.5,  920, 0.04,  0.91,  -2.00, -2.11),
    ("Li et al. [6] BaSO4",     26.0, 42, 1.5,  950, 0.019, 0.95,  -4.50, -4.68),
    ("Li et al. [7] CaCO3",     28.5, 50, 1.6,  910, 0.045, 0.955, -5.00, -4.81),
    ("Li et al. [8] wood",      25.0, 35, 1.4,  880, 0.04,  0.92,  -4.00, -4.21),
]
REPORTED_MAE = 0.20
REPORTED_ACCURACY = 95.6

# Table V (Purdue BaSO4, alpha=0.019, eps=0.95):
# (location, Tamb, RH, wind, GHI, reported "Sim. Value", reported dT, EPW city key)
TABLE_V = [
    ("Al Baha (Aqiq AP)",   34.0, 25.1, 3.5, 920, 0.81, -9.78, "Al Baha / Aqiq (High Elevation)"),
    ("Tabuk (Wajh AP)",     42.0, 15.2, 4.2, 950, 0.82, -6.88, "Al Wajh (Northern Red Sea Coast, Tabuk Province)"),
    ("Dhahran (Abdulaziz)", 44.5, 45.3, 4.1, 980, 0.92, -3.85, "Dhahran (Eastern Province)"),
    ("Jeddah (Abdulaziz)",  41.0, 65.2, 3.2, 940, 0.94, -5.21, "Jeddah (Red Sea Coast)"),
]

# Section 6.4 / Fig. 3: three materials, reported equilibrium temperatures (C).
SEC_6_4 = [
    ("Standard TiO2 paint",      0.20,  0.88, 54.13),
    ("Stanford photonic cooler", 0.03,  0.96, 44.69),
    ("Purdue BaSO4 paint",       0.019, 0.95, 44.15),
]
# Conditions stated in the 6.4 text: Tamb=44.5, GHI=980 (RH/wind taken from Table V Dhahran row)
SEC_6_4_TEXT_COND = (44.5, 45.3, 4.1, 980)

# Section 6.3 claims
CLAIM_DISCONTINUITY = 0.65         # degC jump removed by logistic factor
CLAIM_PRESSURE = dict(p=82800, hc_before=16.2, hc_after=14.1, dT_after=-10.42)


# ---------------------------------------------------------------------------
# Engine loaders
# ---------------------------------------------------------------------------
def load_current_engine(repo_dir):
    sys.path.insert(0, repo_dir)
    import PRC  # noqa: E402
    return PRC


class _OldEngineBase:
    pass


def load_old_engine(old_prc_path):
    """Extract the three physics methods from the July-27 Tkinter class
    WITHOUT importing tkinter, so they run headless and unmodified."""
    src = open(old_prc_path, encoding="utf-8").read()
    tree = ast.parse(src)
    wanted = {"calculateSkyEmissivity", "calculateConvectiveCoefficient",
              "solveEquilibriumTemperature"}
    funcs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "RadiativeCoolingGui":
            funcs = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    if len(funcs) != 3:
        raise RuntimeError("Could not find the 3 physics methods in old PRC.py")
    cls = ast.ClassDef(name="OldEngine", bases=[], keywords=[], body=funcs, decorator_list=[])
    mod = ast.Module(body=[cls], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {"np": np, "fsolve": fsolve}
    exec(compile(mod, old_prc_path, "exec"), ns)
    return ns["OldEngine"]()


def current_solve(PRC, T, rh, w, g, a, e):
    return PRC.solve_equilibrium_temperature(T + 273.15, g, w, rh, e, a, 1.0)


def old_solve(old, T, rh, w, g, a, e):
    return float(old.solveEquilibriumTemperature(T + 273.15, g, w, rh, e, a))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.dirname(os.path.abspath(__file__)),
                    help="folder containing current PRC.py, epw_files/ (default: this script's folder)")
    ap.add_argument("--old", default=None, help="path to July-27 PRC.py (optional)")
    ap.add_argument("--out", default="reproduction_report.csv")
    args = ap.parse_args()

    PRC = load_current_engine(args.repo)
    old = load_old_engine(args.old) if args.old else None
    rows = []

    def record(item, reported, cur, oldv, note=""):
        def status(v):
            if v is None or reported is None:
                return "n/a"
            return "MATCH" if abs(v - reported) <= TOL else "NOT REPRODUCED"
        rows.append(dict(item=item, reported=reported,
                         current=None if cur is None else round(cur, 2),
                         current_status=status(cur),
                         july27=None if oldv is None else round(oldv, 2),
                         july27_status=status(oldv) if old else "not run",
                         note=note))

    # ---------------- Table IV ----------------
    print("\n=== TABLE IV: validation against 8 published experiments (L = 1 m) ===")
    print(f"{'study':22s} {'exp':>6s} {'reported':>8s} {'current':>8s} {'july27':>8s}")
    cur_err, old_err = [], []
    for s, T, rh, w, g, a, e, exp, rep in TABLE_IV:
        c = current_solve(PRC, T, rh, w, g, a, e) - T
        o = old_solve(old, T, rh, w, g, a, e) - T if old else None
        cur_err.append(abs(c - exp))
        if old:
            old_err.append(abs(o - exp))
        print(f"{s:22s} {exp:6.2f} {rep:8.2f} {c:8.2f} {o if o is None else round(o,2):>8}")
        record(f"Table IV sim dT - {s}", rep, c, o)
    acc = lambda errs: 100 * np.mean([1 - er / abs(r[7]) for er, r in zip(errs, TABLE_IV)])
    print(f"MAE      reported {REPORTED_MAE:.2f} | current {np.mean(cur_err):.2f}"
          + (f" | july27 {np.mean(old_err):.2f}" if old else ""))
    print(f"Accuracy reported {REPORTED_ACCURACY:.1f}% | current {acc(cur_err):.1f}%"
          + (f" | july27 {acc(old_err):.1f}%" if old else ""))
    record("Table IV MAE (degC)", REPORTED_MAE, float(np.mean(cur_err)),
           float(np.mean(old_err)) if old else None)
    # internal arithmetic check of the reported table itself
    rep_errs = [abs(r[8] - r[7]) for r in TABLE_IV]
    print(f"Arithmetic check of reported table: mean |reported sim - exp| = {np.mean(rep_errs):.4f}"
          " (consistent with 0.20 => the table's arithmetic is fine; the question is where 'sim' came from)")

    # Diagnostic only: is there ANY single plate length L that reproduces Table IV?
    best = min(((L, np.mean([abs(PRC.solve_equilibrium_temperature(T+273.15, g, w, rh, e, a, L) - T - rep)
                             for _, T, rh, w, g, a, e, _, rep in TABLE_IV]))
                for L in np.geomspace(0.01, 10, 61)), key=lambda x: x[1])
    print(f"Diagnostic: best single L in 0.01-10 m gives mean |current - reported sim| = {best[1]:.2f} degC "
          f"at L = {best[0]:.2f} m (diagnostic only, NOT a fitted model setting)")

    # ---------------- Table V ----------------
    print("\n=== TABLE V: regional results, Purdue BaSO4 (L = 1 m) ===")
    for loc, T, rh, w, g, rep_sky, rep_dT, city in TABLE_V:
        c = current_solve(PRC, T, rh, w, g, 0.019, 0.95) - T
        o = old_solve(old, T, rh, w, g, 0.019, 0.95) - T if old else None
        sky = PRC.calculate_sky_emissivity(T, rh)
        df = PRC.load_epw_weather(city)
        hits = df[(abs(df[6] - T) <= 0.5) & (abs(df[8] - rh) <= 3) & (abs(df[13] - g) <= 40)]
        print(f"{loc:22s} reported dT {rep_dT:+.2f} | current {c:+.2f}"
              + (f" | july27 {o:+.2f}" if old else "")
              + f" | sky-eps code {sky:.3f} vs reported 'Sim. Value' {rep_sky:.2f}"
              + f" | EPW hours matching these conditions: {len(hits)}")
        record(f"Table V dT - {loc}", rep_dT, c, o,
               note=f"EPW rows matching stated conditions: {len(hits)}")
        record(f"Table V 'Sim. Value' (sky emissivity?) - {loc}", rep_sky, sky, None,
               note="tolerance 0.02 on a 2-decimal value")

    # ---------------- Section 6.4 / Fig. 3 ----------------
    print("\n=== SECTION 6.4 / FIG. 3: Dhahran three-material comparison ===")
    T, rh, w, g = SEC_6_4_TEXT_COND
    print(f"(a) Conditions stated in the text: Tamb={T}, RH={rh}, wind={w}, GHI={g}")
    for n, a, e, rep in SEC_6_4:
        c = current_solve(PRC, T, rh, w, g, a, e)
        o = old_solve(old, T, rh, w, g, a, e) if old else None
        print(f"   {n:26s} reported {rep:.2f} | current {c:.2f}" + (f" | july27 {o:.2f}" if old else ""))
        record(f"6.4 Teq (text conditions) - {n}", rep, c, o)
    df = PRC.load_epw_weather("Dhahran (Eastern Province)")
    r = df.loc[df[6].idxmax()]
    T, rh, w, g = float(r[6]), float(r[8]), float(r[21]), float(r[13])
    print(f"(b) Dhahran EPW hottest hour (month {int(r[1])}, day {int(r[2])}, hour {int(r[3])}): "
          f"Tamb={T}, RH={rh}, wind={w}, GHI={g}")
    for n, a, e, rep in SEC_6_4:
        c = current_solve(PRC, T, rh, w, g, a, e)
        o = old_solve(old, T, rh, w, g, a, e) if old else None
        print(f"   {n:26s} reported {rep:.2f} | current {c:.2f}" + (f" | july27 {o:.2f}" if old else ""))
        record(f"6.4 Teq (EPW hottest hour) - {n}", rep, c, o,
               note="Fig. 3 screenshot conditions (app default scenario)")

    # ---------------- Section 6.3: discontinuity claim ----------------
    print("\n=== SECTION 6.3: convective discontinuity claim ===")
    if old:
        # Jump in Teq for BaSO4 at the Re = 5e5 switch in the July-27 code, Dhahran hottest-hour air
        winds = np.linspace(0.1, 15, 3000)
        teq_old = np.array([old_solve(old, T, rh, wv, g, 0.019, 0.95) for wv in winds])
        teq_cur = np.array([current_solve(PRC, T, rh, wv, g, 0.019, 0.95) for wv in winds])
        jump_old = np.max(np.abs(np.diff(teq_old)))
        jump_cur = np.max(np.abs(np.diff(teq_cur)))
        print(f"Largest step between adjacent wind values (dv={winds[1]-winds[0]:.4f} m/s), BaSO4:")
        print(f"   July-27 hard switch : {jump_old:.2f} degC   | current logistic blend: {jump_cur:.3f} degC")
        record("6.3 discontinuity removed (degC)", CLAIM_DISCONTINUITY, None, jump_old,
               note="july27 column = size of jump in old code under Dhahran hottest-hour air; depends on conditions")
    # Standard textbook mixed-BL correlation continuity check (no blending)
    Pr = 0.71
    Re = 5e5
    nu_lam = 0.664 * Re**0.5 * Pr**(1/3)
    nu_mixed = (0.037 * Re**0.8 - 871) * Pr**(1/3)
    nu_turb_only = 0.037 * Re**0.8 * Pr**(1/3)
    print(f"At Re=5e5: laminar Nu={nu_lam:.0f}, textbook mixed Nu (with -871)={nu_mixed:.0f}, "
          f"fully-turbulent Nu (no -871, as in July-27)={nu_turb_only:.0f}")

    # ---------------- Section 6.3: pressure claim ----------------
    print("\n=== SECTION 6.3: altitude pressure claim (Al Baha) ===")
    print("Neither the current nor the July-27 PRC.py has any pressure input (101325 Pa is hard-coded).")
    T, rh, w, g = TABLE_V[0][1:5]
    hc_cur = PRC.calculate_convective_coefficient(w, T, rh, 1.0)
    print(f"Code h_c at Table V Al Baha conditions (101325 Pa, L=1): {hc_cur:.2f} W/m2K "
          f"(report says {CLAIM_PRESSURE['hc_before']} before pressure correction)")
    record("6.3 h_c before pressure correction (W/m2K)", CLAIM_PRESSURE["hc_before"], hc_cur,
           old.calculateConvectiveCoefficient(w, T, rh) if old else None)
    record("6.3 dT after pressure correction (degC)", CLAIM_PRESSURE["dT_after"], None, None,
           note="NOT TESTABLE: no version of the code contains a pressure input")

    # ---------------- write CSV ----------------
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    n_match = sum(r["current_status"] == "MATCH" for r in rows)
    n_match_old = sum(r["july27_status"] == "MATCH" for r in rows)
    n_test = sum(r["current_status"] != "n/a" for r in rows)
    print(f"\nSUMMARY: current engine reproduces {n_match}/{n_test} testable reported numbers"
          + (f"; July-27 engine reproduces {n_match_old}" if old else "")
          + f". Full table written to {args.out}")


if __name__ == "__main__":
    main()
