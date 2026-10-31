"""
uncertainty.py  -  combined uncertainty and performance boundaries
==================================================================
Run from the repository root:   python uncertainty.py

Same hours as analysis_summer.py (June-August, GHI > 600 W/m2, verified clear sky:
total AND opaque sky cover both zero).

For every material and station, the median surface-minus-air temperature (dT) is
recomputed under every combination of the uncertainties that can be tested with
this model:

  property uncertainty (independent for each material)
    alpha   : -0.02, 0, +0.02       (about the measurement uncertainty in Mandal et al. 2018)
    epsilon : -0.02, 0, +0.02       (capped at 1)
  structural uncertainty (shared by all materials at a station)
    plate length L : 0.1 m, 1 m, 10 m
    sky model      : Berdahl & Fromberg (1982) and Berdahl & Martin (1984, with diurnal term)

That is 9 property cases x 6 structural cases = 54 cases per material and station.

Outputs (folder results_uncertainty/):
  envelope.csv   min and max median dT over all 54 cases; "robust_subambient" is True
                 when even the warmest case stays below air temperature.
  pairs.csv      for each pair of materials and station: the smallest and largest gap
                 between their medians, over all 6 structural cases and all 81
                 combinations of the two materials' property cases. A pair is
                 "distinguishable" when the gap never changes sign.

Not covered (stated as limitations): cloud, full spectra, angle dependence, the 2.5 W/m2K
natural-convection term, wind height, dust, and anything only an outdoor test can reveal.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import PRC  # noqa: E402
import analysis_summer as A  # noqa: E402

D = 0.02
LENGTHS = [0.1, 1.0, 10.0]
SKIES = ["bf1982", "bm1984"]


def main():
    out = os.path.join(HERE, "results_uncertainty")
    os.makedirs(out, exist_ok=True)
    materials = PRC.load_material_database("materials_verified.csv")
    weather = {c: PRC.load_epw_weather(c) for c in PRC.city_profiles}
    short = {c: c.split(" (")[0].split(" /")[0] for c in weather}

    env_rows, pair_rows = [], []
    for c, df in weather.items():
        hrs = A.select_hours(df, 600.0, [6, 7, 8], sky="clear")
        # med[(material, struct)] = 3x3 array over (alpha case, eps case)
        med = {}
        for m in materials:
            for L, sky in itertools.product(LENGTHS, SKIES):
                arr = np.empty((3, 3))
                for i, da in enumerate((-D, 0.0, D)):
                    for j, de in enumerate((-D, 0.0, D)):
                        a = max(m["alpha"] + da, 0.0)
                        e = min(m["epsilon"] + de, 1.0)
                        arr[i, j] = np.median(A.delta_t(hrs, a, e, L, sky))
                med[(m["name"], L, sky)] = arr
            vals = np.concatenate([med[(m["name"], L, s)].ravel() for L in LENGTHS for s in SKIES])
            base = med[(m["name"], 1.0, "bf1982")][1, 1]
            env_rows.append(dict(station=short[c], material=m["name"], n_hours=len(hrs), central=base,
                                 low=vals.min(), high=vals.max(), robust_subambient=bool(vals.max() < 0)))
        names = [m["name"] for m in materials]
        for a_name, b_name in itertools.combinations(names, 2):
            gaps = []
            for L, sky in itertools.product(LENGTHS, SKIES):
                ga = med[(a_name, L, sky)].ravel()
                gb = med[(b_name, L, sky)].ravel()
                gaps.append((gb[:, None] - ga[None, :]).ravel())  # positive: a cooler than b
            gaps = np.concatenate(gaps)
            central = med[(b_name, 1.0, "bf1982")][1, 1] - med[(a_name, 1.0, "bf1982")][1, 1]
            pair_rows.append(dict(station=short[c], cooler=a_name if central > 0 else b_name,
                                  warmer=b_name if central > 0 else a_name, central_gap=abs(central),
                                  gap_min=gaps.min() * np.sign(central), gap_max=gaps.max() * np.sign(central),
                                  distinguishable=bool((gaps > 0).all() or (gaps < 0).all())))
        print(f"{short[c]}: done ({len(hrs)} hours)")

    env = pd.DataFrame(env_rows)
    pairs = pd.DataFrame(pair_rows)
    # gap_min/gap_max were multiplied by the sign of the central gap; order them
    lo = pairs[["gap_min", "gap_max"]].min(axis=1)
    hi = pairs[["gap_min", "gap_max"]].max(axis=1)
    pairs["gap_min"], pairs["gap_max"] = lo, hi
    env.round(3).to_csv(os.path.join(out, "envelope.csv"), index=False)
    pairs.round(3).to_csv(os.path.join(out, "pairs.csv"), index=False)
    pd.set_option("display.width", 220)
    print(env.round(2).to_string(index=False))
    print(pairs.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
