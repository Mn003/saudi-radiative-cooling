# Changes made on 23 Oct 2026 (code review for Ibdaa 2027)

PRC.py
- Convection: replaced the logistic blending factor with the standard flat-plate
  correlations (Incropera): laminar Nu = 0.664 Re^0.5 Pr^(1/3) below Re = 5e5, mixed
  boundary layer Nu = (0.037 Re^0.8 - 871) Pr^(1/3) above. These are continuous at 5e5.
  The old blend mixed in a negative turbulent term at low wind.
- Station pressure: humidity ratio and air density now use the hourly station pressure
  from EPW field 9 (default 101325 Pa if none is given). Matters at Al Baha (~83 kPa).
- Solver: fsolve replaced by brentq (bracketed, guaranteed to converge; the energy
  balance has exactly one root).
- Loaders: no more silent fallbacks. A missing/invalid weather file or material file
  now stops with a clear error instead of switching to synthetic weather or a
  6-material preset list.
- Sky emissivity: formula unchanged (0.741 + 0.0062 Tdp); docstring now credits
  Berdahl & Fromberg (1982) instead of Berdahl & Martin, and flags that the
  day/night form of the correlation must be confirmed in the original paper.

app.py
- All solver calls now pass the station pressure of the selected weather hour.

New files
- materials_verified.csv: the materials whose alpha and epsilon were checked in the
  original papers (source and page recorded). materials_database.csv is unchanged.
- analysis_summer.py: Objectives 2-3 (summer-midday distributions, ranking by station,
  threshold robustness 400/600/800 W/m2, sensitivity, alpha-epsilon trade-off,
  alpha threshold). Writes results/ CSV files.
- reproduce_last_year.py, climate_summary.py: reproducibility and weather summary.

## Added 23 Oct 2026
- materials_verified.csv: commercial white paint control (Dutch Boy Maxbond UltraWhite, solar reflectance 87.2% from Li et al. 2020, p.5), run at assumed emissivity 0.90 and 0.95 because no number is reported.
- PRC.py: sky-emissivity docstring now credits Berdahl & Fromberg (1982) and states what was and was not checked.
- make_figures.py: draws Figures 1-5 of the Qalib from results/ and the EPW files (run after analysis_summer.py).

## App and labels (23 Oct 2026, evening; to be reviewed with the student)
- Station "Tabuk / Wajh" renamed "Al Wajh (Northern Red Sea Coast, Tabuk Province)" everywhere; results files now say "Al Wajh". Same weather file, same numbers.
- Station descriptions in PRC.py replaced with values computed from the weather files (elevation, maximum temperature, June–August dew point). The old text had unsourced claims (e.g. "often >90% RH", "~1,700m+").
- app.py:
  - Default materials are now the verified ones (BaSO4 paint, CaCO3 paint, commercial white paint control), matching the Qalib. The old defaults (an unsourced TiO2 paint and the Stanford cooler, which has no reported epsilon value) are no longer used.
  - The material library shows the 6 verified entries by default; the 60 older entries are labelled "Not yet verified" and only shown if the user ticks a box. The "60+ literature materials" label is gone.
  - The results table is recalculated automatically whenever the city, hour, plate length or materials change, so it can no longer show results from a previous selection. The "Run" button is no longer needed and was removed.
  - "Absolute Hottest Day" / "Maximum Solar Peak Day" renamed "Hottest hour of the year" / "Sunniest hour of the year"; the date and hour are shown.
  - Daily-plot hour labels every 3 hours; deprecated use_container_width replaced by width="stretch".
- Physics unchanged: analysis_summer.py gives identical results before and after these changes.

## Control emissivity and solver check (24 Oct 2026, reviewed with the student)
- materials_verified.csv: the control's emissivity is now DERIVED, not assumed. Li et al. (2020) give the figure of merit RC = eps_sky - r(1 - R_solar) (Eq. 3, p.7, r = 10) and RC = -0.33 for the commercial white paint (p.8), with R_solar = 0.872 (p.5). So eps_sky = -0.33 + 10 x 0.128 = 0.95 (+/- about 0.01: RC is rounded to 2 decimals and R to 0.1%, each giving +/- 0.005). The same calculation gives 0.94 for their CaCO3 paint, matching the value they report. The two assumed rows (0.90, 0.95) are replaced by one row (0.95).
- check_solver.py: compares the July 2026 solver (fsolve) with the current one (brentq) on every hour of all four weather files for every verified material. Result: fsolve never failed; largest difference 2.5e-7 K.


## October 31, 2026: clear-sky hours and combined uncertainty
- `analysis_summer.py` now uses only verified clear-sky hours by default: hours in which BOTH the total and the opaque sky-cover fields are zero. The two fields contradict each other in many hours (opaque > total in 193 Jeddah and 383 Al Wajh hours), and the model assumes a clear sky. `--sky all` restores the old selection.
- `PRC.py`: optional second sky model (`sky_model="bm1984"`, Berdahl & Martin 1984, with the diurnal term). The default is unchanged, so the app and `check_solver.py` give the same results as before.
- New `uncertainty.py`: medians for all 54 combinations of alpha and epsilon (+/-0.02), plate length (0.1, 1, 10 m) and sky model; reports robust sub-ambient predictions and which material pairs are distinguishable.
- New `threshold_uncertainty.py`: range of the reflectance threshold over the same uncertainties.
- `make_figures.py`: Figure 4 shows the uncertainty range; Figure 5 shows the threshold range; the flowchart shows the uncertainty analysis and the rooftop test.
