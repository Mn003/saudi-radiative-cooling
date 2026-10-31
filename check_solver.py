"""
check_solver.py - compares the July 2026 solver (scipy fsolve) with the current one (brentq).

For every hour of all four weather files (4 x 8,760 = 35,040 hours) and every material in
materials_verified.csv, it solves the same energy balance with both solvers and reports:
  - how many times fsolve said it did NOT converge
  - the largest difference between the two answers
Run:  python check_solver.py
"""
import time
import numpy as np
import scipy.optimize as opt
import PRC

SIGMA = 5.67e-8
materials = PRC.load_material_database("materials_verified.csv")
n_cases = n_fail = 0
max_diff = 0.0
t0 = time.time()
for city in PRC.city_profiles:
    df = PRC.load_epw_weather(city)
    for _, r in df.iterrows():
        t_air_c = r[PRC.epwDryBulbTempCol]
        t_air = t_air_c + 273.15
        ghi, wind, rh, p = r[PRC.epwGhiCol], r[PRC.epwWindSpeedCol], r[PRC.epwRelHumidityCol], r[PRC.epwPressureCol]
        h_c = PRC.calculate_convective_coefficient(wind, t_air_c, rh, 1.0, p)
        eps_sky = PRC.calculate_sky_emissivity(t_air_c, rh)
        for m in materials:
            eps, alpha = m["epsilon"], m["alpha"]
            balance = lambda T: eps * SIGMA * T**4 - eps * eps_sky * SIGMA * t_air**4 - alpha * ghi + h_c * (T - t_air)
            new = opt.brentq(balance, t_air - 100, t_air + 100, xtol=1e-6)        # current solver
            old, info, ier, msg = opt.fsolve(balance, x0=t_air, full_output=True)  # July 2026 solver
            n_cases += 1
            if ier != 1:          # fsolve reports ier = 1 only when it has converged
                n_fail += 1
            else:
                max_diff = max(max_diff, abs(old[0] - new))
print(f"cases checked: {n_cases:,}  ({len(PRC.city_profiles)} stations x 8,760 hours x {len(materials)} materials)")
print(f"fsolve did not converge: {n_fail}")
print(f"largest difference between the two solvers: {max_diff:.2e} K")
print(f"time taken: {time.time() - t0:.0f} s")
