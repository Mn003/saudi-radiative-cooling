import os
import csv
import pandas as pd
import numpy as np
import scipy.optimize as opt

# ==========================================
# EPW WEATHER DATA COLUMN MAPPING (0-indexed)
# ==========================================
epwDryBulbTempCol = 6
epwDewPointCol = 7       # Dew Point Temperature (°C)
epwRelHumidityCol = 8    # Relative Humidity (%)
epwPressureCol = 9       # Station (atmospheric) pressure (Pa)
epwGhiCol = 13           # Global Horizontal Irradiance (W/m²), averaged over the hour
epwWindSpeedCol = 21     # Wind Speed (m/s), measured at ~10 m height
epwTotalSkyCoverCol = 22  # Total sky cover (tenths)
epwOpaqueSkyCoverCol = 23 # Opaque sky cover (tenths)

SEA_LEVEL_PRESSURE_PA = 101325.0


# ==========================================
# METEOROLOGICAL & CITY PROFILES METADATA
# ==========================================
# Descriptions are computed from the TMYx files themselves (see climate_summary.py).
city_profiles = {
    "Dhahran (Eastern Province)": {
        "file": "SAU_SH_Dhahran-Abdulaziz.AB.404160_TMYx.2011-2025.epw",
        "specialty": "Arabian Gulf coast, 26 m elevation. Hottest of the four: maximum 48.0 °C, 986 hours per year at or above 40 °C; mean June–August dew point 19.9 °C."
    },
    "Jeddah (Red Sea Coast)": {
        "file": "SAU_MK_Jeddah-Abdulaziz.Intl.AP.410240_TMYx.2011-2025.epw",
        "specialty": "Central Red Sea coast, 15 m elevation. Humid: mean June–August dew point 22.8 °C; maximum 43.0 °C."
    },
    "Al Wajh (Northern Red Sea Coast, Tabuk Province)": {
        "file": "SAU_TB_Wajh.AP.404000_TMYx.2007-2021.epw",
        "specialty": "Northern Red Sea coast, 20 m elevation. Humid: mean June–August dew point 22.8 °C; maximum 43.0 °C."
    },
    "Al Baha / Aqiq (High Elevation)": {
        "file": "SAU_BA_Aqiq-Abdulaziz.AP.410550_TMYx.2011-2025.epw",
        "specialty": "Sarawat mountains, 1,672 m elevation. Driest summer air of the four: mean June–August dew point 7.3 °C; maximum 38.5 °C."
    }
}


# ==========================================
# THERMODYNAMIC & PHYSICAL SOLVER FUNCTIONS
# ==========================================
def calculate_sky_emissivity(temp_celsius, relative_humidity, sky_model="bf1982", solar_hour=None):
    """
    Clear-sky emissivity from the dew point (linear dew-point correlation,
    eps_sky = 0.741 + 0.0062 * T_dew, T_dew in °C).

    Source: Berdahl & Fromberg (1982), Solar Energy 29, 299-314. Their abstract
    gives eps_sky = 0.741 + 0.0062 T_dp (T_dp in °C), fitted on 2,945 night-time
    measurements, and states that "a similar relationship with almost identical
    coefficients holds during daylight hours". (The original code wrongly credited
    Berdahl & Martin.) The exact daytime coefficients were not checked because the
    full text was not accessible. The dew point is obtained from air temperature and relative
    humidity with the Magnus-Tetens formula. Clouds are NOT included
    (clear-sky assumption). The result is limited to 0.70-0.95.
    """
    a, b = 17.27, 237.7
    alpha = ((a * temp_celsius) / (b + temp_celsius)) + np.log(max(relative_humidity, 1e-3) / 100.0)
    temp_dew = (b * alpha) / (a - alpha)
    if sky_model == "bf1982":
        sky_emissivity = 0.741 + (0.0062 * temp_dew)
    elif sky_model == "bm1984":
        # Berdahl & Martin (1984), Solar Energy 32, 663-664 (LBL-15367 preprint, Eqs. 2-3):
        # 24-hour clear-sky average eps = 0.711 + 0.56 (Tdp/100) + 0.73 (Tdp/100)^2,
        # plus the diurnal term 0.013 cos(2*pi*t/24), t = solar time in hours.
        # Used only as a second, independent sky model for the uncertainty analysis.
        x = temp_dew / 100.0
        sky_emissivity = 0.711 + 0.56 * x + 0.73 * x * x
        if solar_hour is not None:
            sky_emissivity += 0.013 * np.cos(2.0 * np.pi * solar_hour / 24.0)
    else:
        raise ValueError(f"unknown sky_model: {sky_model}")
    return float(np.clip(sky_emissivity, 0.70, 0.95))


def calculate_convective_coefficient(wind_speed, temp_celsius, relative_humidity, length=1.0,
                                     pressure_pa=SEA_LEVEL_PRESSURE_PA):
    """
    Convective heat transfer coefficient h_c (W/m²K) for a flat plate of length L.

    h_c = h_nat + Nu * k / L, with a fixed natural-convection term h_nat = 2.5 W/m²K
    (an assumption of this model).

    Forced convection uses the standard flat-plate correlations (Incropera et al.,
    Fundamentals of Heat and Mass Transfer):
      Re < 5e5 : laminar        Nu = 0.664 Re^0.5 Pr^(1/3)
      Re >= 5e5: mixed boundary layer  Nu = (0.037 Re^0.8 - 871) Pr^(1/3)
    These two expressions are equal at Re = 5e5, so h_c is continuous.
    (Earlier versions blended them with a logistic factor; that is no longer
    needed and has been removed.)

    Air properties: Sutherland-type dry-air viscosity and conductivity with
    linear moist-air modifiers (after Tsilingiris 2008), and moist-air density
    from the ideal-gas law at the given station pressure.
    """
    natural_convection = 2.5
    if wind_speed <= 0.05:
        return natural_convection

    temp_kelvin = temp_celsius + 273.15
    sat_pressure = 610.78 * np.exp((17.27 * temp_celsius) / (temp_celsius + 237.3))
    vapour_pressure = (relative_humidity / 100.0) * sat_pressure
    humidity_ratio = 0.622 * vapour_pressure / (pressure_pa - vapour_pressure)
    air_density = pressure_pa / (287.05 * temp_kelvin * (1.0 + 0.608 * humidity_ratio))

    moist_viscosity = (1.458e-6 * (temp_kelvin**1.5) / (temp_kelvin + 110.4)) * (1.0 + 0.23 * humidity_ratio)
    moist_conductivity = (2.495e-3 * (temp_kelvin**1.5) / (temp_kelvin + 194.4)) * (1.0 + 0.45 * humidity_ratio)
    prandtl_num = ((1005 + 1820 * humidity_ratio) * moist_viscosity) / moist_conductivity
    
    # Reynolds number using specific Characteristic Length
    reynolds_num = (air_density * wind_speed * length) / moist_viscosity

    re_crit = 5e5
    if reynolds_num < re_crit:
        forced_nusselt = 0.664 * (reynolds_num**0.5) * (prandtl_num**(1.0 / 3.0))
    else:
        forced_nusselt = (0.037 * (reynolds_num**0.8) - 871.0) * (prandtl_num**(1.0 / 3.0))

    # Convective Coefficient scaling: h_c = Nu * k / L
    return natural_convection + ((forced_nusselt * moist_conductivity) / length)


def solve_equilibrium_temperature(temp_air_k, ghi_val, wind_speed, rel_hum, emissivity, absorptivity,
                                  length=1.0, pressure_pa=SEA_LEVEL_PRESSURE_PA, sky_model="bf1982",
                                  solar_hour=None):
    """
    Steady-state energy balance of a passive, sky-facing horizontal surface:
        P_rad(T) - P_atm - P_solar + P_conv(T) = 0
      P_rad   = eps * sigma * T^4                 (emitted)
      P_atm   = eps * eps_sky * sigma * T_air^4   (absorbed sky radiation; grey-body)
      P_solar = alpha * GHI                       (absorbed sunlight)
      P_conv  = h_c * (T - T_air)                 (convection with the air)
    No conduction to the building below (the back surface is treated as insulated).

    The left-hand side increases strictly with T, so there is exactly one root.
    It is found with Brent's method inside a bracket of +/-100 K around the air
    temperature (scipy.optimize.brentq), which is guaranteed to converge.
    Returns the surface temperature in °C.
    """
    temp_air_c = temp_air_k - 273.15
    convective_coef = calculate_convective_coefficient(wind_speed, temp_air_c, rel_hum, length, pressure_pa)
    sky_emissivity = calculate_sky_emissivity(temp_air_c, rel_hum, sky_model, solar_hour)
    stefan_boltzmann_const = 5.67e-8

    def heat_balance(temp_k):
        return ((emissivity * stefan_boltzmann_const * (temp_k**4))
                - (emissivity * sky_emissivity * stefan_boltzmann_const * (temp_air_k**4))
                - (absorptivity * ghi_val)
                + (convective_coef * (temp_k - temp_air_k)))

    root = opt.brentq(heat_balance, temp_air_k - 100.0, temp_air_k + 100.0, xtol=1e-6)
    return float(root - 273.15)


# ==========================================
# MATERIAL DATABASE LOADER
# ==========================================
def load_material_database(filename="materials_database.csv"):
    """
    Loads the material library from a CSV file next to this script.
    Raises an error if the file is missing or malformed. (Earlier versions
    silently switched to a built-in 6-material list, which could hide errors.)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, filename)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Material database not found: {csv_path}")

    materials = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            try:
                materials.append({
                    "name": row["Name"],
                    "chemical": row.get("Chemical_Formula", "N/A"),
                    "category": row.get("Category", "General"),
                    "alpha": float(row["Alpha"]),
                    "epsilon": float(row["Epsilon"]),
                    "spec": row.get("Thickness_or_Spec", "N/A"),
                    "university": row.get("University_or_Institution", "N/A"),
                    "reference": row.get("Reference", "N/A"),
                    "epsilon_type": row.get("Epsilon_Type", "not recorded"),
                    "source_location": row.get("Source_Location", "not recorded"),
                })
            except (KeyError, ValueError) as err:
                raise ValueError(f"{csv_path}, line {line_no}: invalid material row ({err})") from err
    if not materials:
        raise ValueError(f"Material database is empty: {csv_path}")
    return materials


# ==========================================
# EPW WEATHER FILE PARSER
# ==========================================
def load_epw_weather(city_name):
    """
    Parses an hourly EnergyPlus Weather (.epw) TMYx file (8,760 rows).
    Raises an error if the file is missing or does not contain 8,760 valid
    hours. (Earlier versions silently returned synthetic sine-wave weather.)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "epw_files", city_profiles[city_name]["file"])
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Weather file not found: {file_path}")

    df = pd.read_csv(file_path, skiprows=8, header=None)
    cols = [epwDryBulbTempCol, epwDewPointCol, epwRelHumidityCol, epwPressureCol, epwGhiCol, epwWindSpeedCol,
            epwTotalSkyCoverCol, epwOpaqueSkyCoverCol]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n_bad = int(df[cols].isna().any(axis=1).sum())
    if len(df) != 8760 or n_bad:
        raise ValueError(f"{file_path}: expected 8760 complete hourly rows, got {len(df)} rows with {n_bad} incomplete")
    return df
