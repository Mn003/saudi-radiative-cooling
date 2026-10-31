"""
analysis_summer.py  -  Objectives 2 and 3 of the Ibdaa 2027 Project Detail Form
================================================================================
Run from the repository root:

    python analysis_summer.py                 # default: GHI > 600 W/m2, June-August
    python analysis_summer.py --ghi-min 400   # different irradiance threshold
    python analysis_summer.py --sky all       # all selected hours, not only verified clear-sky hours

What it does
------------
1. Loads the four TMYx weather files (epw_files/) and the verified materials
   (materials_verified.csv). Nothing is tuned or fitted.
2. Selects "summer midday" hours: June-August hours with GHI above --ghi-min.
   By default (--sky clear) only verified clear-sky hours are kept: hours in which
   BOTH the total and the opaque sky-cover fields of the weather file are zero.
   The two fields contradict each other in many hours (opaque cover larger than
   total cover, which is physically impossible), so an hour counts as clear only
   when both agree that there is no cloud. The model itself assumes clear sky.
3. For every selected hour, material and station, solves the surface energy
   balance with PRC.solve_equilibrium_temperature, using that hour's air
   temperature, humidity, wind, solar irradiance and station pressure.
4. Reports, per material and station:
     - median and interquartile range of (surface - air) temperature, dT
     - share of selected hours in which the surface is below air temperature
5. Objective 2: ranks the materials by median dT at each station and reports
   whether the ranking is the same at all stations. Repeats the ranking at
   400, 600 and 800 W/m2 to show whether the threshold changes the answer.
6. Objective 3: for each material and station, the change in median dT for
   alpha +/-0.02 and epsilon +/-0.02, and the epsilon increase needed to offset
   a 0.01 increase in alpha (from central differences).
7. Extra: for a surface with epsilon = 0.95, the highest alpha that keeps the
   median dT at or below zero, per station.

Outputs (folder results/): summary_by_material_station.csv, ranking.csv,
sensitivity.csv, alpha_threshold.csv, run_info.txt
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import PRC  # noqa: E402

COLS = dict(t=PRC.epwDryBulbTempCol, rh=PRC.epwRelHumidityCol, ghi=PRC.epwGhiCol,
            wind=PRC.epwWindSpeedCol, p=PRC.epwPressureCol, month=1, day=2, hour=3,
            tot=PRC.epwTotalSkyCoverCol, opq=PRC.epwOpaqueSkyCoverCol)
SKY = "clear"  # set from the command line in main()


def select_hours(df, ghi_min, months, sky=None):
    """Rows: air temp, RH, GHI, wind, pressure, solar hour (EPW hour - 0.5)."""
    sky = SKY if sky is None else sky
    mask = (df[COLS["month"]].isin(months)) & (df[COLS["ghi"]] > ghi_min)
    if sky == "clear":
        mask &= (df[COLS["tot"]] == 0) & (df[COLS["opq"]] == 0)
    sel = df[mask]
    out = sel[[COLS["t"], COLS["rh"], COLS["ghi"], COLS["wind"], COLS["p"], COLS["hour"]]].to_numpy(dtype=float)
    out[:, 5] -= 0.5  # EPW hour N covers N-1..N; use the middle of the hour
    return out


def delta_t(hours, alpha, eps, length, sky_model="bf1982"):
    """Surface minus air temperature (K) for each selected hour."""
    out = np.empty(len(hours))
    for i, (t, rh, ghi, wind, p, hr) in enumerate(hours):
        out[i] = PRC.solve_equilibrium_temperature(t + 273.15, ghi, wind, rh, eps, alpha,
                                                   length=length, pressure_pa=p, sky_model=sky_model,
                                                   solar_hour=hr) - t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ghi-min", type=float, default=600.0)
    ap.add_argument("--months", type=int, nargs="+", default=[6, 7, 8])
    ap.add_argument("--materials", default="materials_verified.csv")
    ap.add_argument("--length", type=float, default=1.0, help="plate length L in metres")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--sky", choices=["clear", "all"], default="clear",
                    help="clear: only hours with zero total AND zero opaque sky cover (default)")
    args = ap.parse_args()
    global SKY
    SKY = args.sky
    os.makedirs(args.out, exist_ok=True)

    materials = PRC.load_material_database(args.materials)
    weather = {c: PRC.load_epw_weather(c) for c in PRC.city_profiles}
    short = {c: c.split(" (")[0].split(" /")[0] for c in weather}

    # ---- summaries at the main threshold ------------------------------------
    rows = []
    hours_main = {c: select_hours(df, args.ghi_min, args.months) for c, df in weather.items()}
    for c, hrs in hours_main.items():
        for m in materials:
            d = delta_t(hrs, m["alpha"], m["epsilon"], args.length)
            rows.append(dict(station=short[c], material=m["name"], alpha=m["alpha"], epsilon=m["epsilon"],
                             n_hours=len(d), median_dT=np.median(d), p25_dT=np.percentile(d, 25),
                             p75_dT=np.percentile(d, 75), share_below_air=np.mean(d < 0)))
    summary = pd.DataFrame(rows)
    summary.round(3).to_csv(os.path.join(args.out, "summary_by_material_station.csv"), index=False)

    # ---- Objective 2: ranking, and robustness to the threshold --------------
    rank_rows = []
    for thr in sorted({400.0, args.ghi_min, 800.0}):
        for c, df in weather.items():
            hrs = select_hours(df, thr, args.months)
            med = {m["name"]: np.median(delta_t(hrs, m["alpha"], m["epsilon"], args.length)) for m in materials}
            order = sorted(med, key=med.get)  # coolest first
            rank_rows.append(dict(ghi_min=thr, station=short[c], n_hours=len(hrs),
                                  ranking_coolest_first=" > ".join(order)))
    ranking = pd.DataFrame(rank_rows)
    ranking.to_csv(os.path.join(args.out, "ranking.csv"), index=False)

    # ---- Objective 3: sensitivity and alpha-epsilon trade-off ---------------
    sens_rows = []
    h = 0.02
    for c, hrs in hours_main.items():
        for m in materials:
            a, e = m["alpha"], m["epsilon"]
            med = lambda aa, ee: np.median(delta_t(hrs, aa, ee, args.length))  # noqa: E731
            base = med(a, e)
            a_up, a_dn = med(a + h, e), med(max(a - h, 0.0), e)
            e_up, e_dn = med(a, min(e + h, 1.0)), med(a, e - h)
            dT_da = (a_up - a_dn) / ((a + h) - max(a - h, 0.0))
            dT_de = (e_up - e_dn) / (min(e + h, 1.0) - (e - h))
            sens_rows.append(dict(station=short[c], material=m["name"], median_dT=base,
                                  change_alpha_plus_0p02=a_up - base, change_alpha_minus_0p02=a_dn - base,
                                  change_eps_plus_0p02=e_up - base, change_eps_minus_0p02=e_dn - base,
                                  dT_per_unit_alpha=dT_da, dT_per_unit_eps=dT_de,
                                  eps_increase_to_offset_alpha_0p01=-0.01 * dT_da / dT_de))
    pd.DataFrame(sens_rows).round(4).to_csv(os.path.join(args.out, "sensitivity.csv"), index=False)

    # ---- Extra: alpha threshold for eps = 0.95 -------------------------------
    thr_rows = []
    for c, hrs in hours_main.items():
        f = lambda aa: np.median(delta_t(hrs, aa, 0.95, args.length))  # noqa: E731
        if f(0.0) > 0:
            a_star = float("nan")  # even a perfect reflector stays above air
        else:
            a_star = brentq(f, 0.0, 0.5, xtol=1e-4)
        thr_rows.append(dict(station=short[c], epsilon=0.95, max_alpha_for_median_below_air=a_star))
    pd.DataFrame(thr_rows).round(4).to_csv(os.path.join(args.out, "alpha_threshold.csv"), index=False)

    with open(os.path.join(args.out, "run_info.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"ghi_min={args.ghi_min} months={args.months} length={args.length} sky={args.sky} materials={args.materials}\n")
        fh.write("Weather: Climate.OneBuilding TMYx files in epw_files/; station pressure from EPW field 9.\n")
        fh.write("Model: PRC.solve_equilibrium_temperature (grey body, clear sky, insulated back).\n")

    pd.set_option("display.width", 200)
    print(summary.round(2).to_string(index=False))
    print()
    print(ranking.to_string(index=False))
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
