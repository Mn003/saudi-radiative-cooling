"""threshold_uncertainty.py - range of the reflectance threshold across the structural cases.
For a surface with eps = 0.95 (and 0.93, 0.97), the largest alpha that keeps the median
surface-minus-air temperature at or below zero, for each plate length (0.1, 1, 10 m) and
sky model (Berdahl & Fromberg 1982; Berdahl & Martin 1984). Same clear-sky summer hours as
analysis_summer.py. Output: results_uncertainty/threshold_range.csv"""
import itertools, os, sys
import numpy as np, pandas as pd
from scipy.optimize import brentq
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import PRC, analysis_summer as A
rows = []
for c in PRC.city_profiles:
    hrs = A.select_hours(PRC.load_epw_weather(c), 600.0, [6, 7, 8], sky="clear")
    for eps, L, sky in itertools.product([0.93, 0.95, 0.97], [0.1, 1.0, 10.0], ["bf1982", "bm1984"]):
        f = lambda a: np.median(A.delta_t(hrs, a, eps, L, sky))
        a_star = brentq(f, 0.0, 0.5, xtol=1e-4) if f(0.0) < 0 else float("nan")
        rows.append(dict(station=c.split(" (")[0].split(" /")[0], epsilon=eps, L=L, sky=sky, max_alpha=a_star))
df = pd.DataFrame(rows)
os.makedirs(os.path.join(HERE, "results_uncertainty"), exist_ok=True)
df.round(4).to_csv(os.path.join(HERE, "results_uncertainty", "threshold_range.csv"), index=False)
g = df.groupby("station").max_alpha.agg(["min", "max"])
g["reflectance_needed_low"] = 1 - g["max"]; g["reflectance_needed_high"] = 1 - g["min"]
print(g.round(3))
print(df[df.epsilon == 0.95].groupby("station").max_alpha.agg(["min", "max"]).round(3))
