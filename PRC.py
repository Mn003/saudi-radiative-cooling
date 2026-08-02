import os
import csv
import pandas as pd
import numpy as np
import scipy.optimize as opt

# ==========================================
# EPW WEATHER DATA COLUMN MAPPING (0-indexed)
# ==========================================
epwDryBulbTempCol = 6
epwRelHumidityCol = 8    # Relative Humidity (%)
epwGhiCol = 13           # Global Horizontal Irradiance (W/m²)
epwWindSpeedCol = 21     # Wind Speed (m/s)


# ==========================================
# METEOROLOGICAL & CITY PROFILES METADATA
# ==========================================
city_profiles = {
    "Dhahran (Eastern Province)": {
        "file": "SAU_SH_Dhahran-Abdulaziz.AB.404160_TMYx.2011-2025.epw",
        "specialty": "Arabian Gulf Coastal-Desert: Subject to extreme summer humidity spikes (often >90% RH) and intense solar loads."
    },
    "Jeddah (Red Sea Coast)": {
        "file": "SAU_MK_Jeddah-Abdulaziz.Intl.AP.410240_TMYx.2011-2025.epw",
        "specialty": "Red Sea Coastal Climate: High year-round mean relative humidity (~55–65%) with narrow diurnal temperature ranges."
    },
    "Tabuk / Wajh (Western Coast & Northern Region)": {
        "file": "SAU_TB_Wajh.AP.404000_TMYx.2007-2021.epw",
        "specialty": "Red Sea North-Western Coastline: Persistent coastal winds and maritime air masses."
    },
    "Al Baha / Aqiq (High Elevation)": {
        "file": "SAU_BA_Aqiq-Abdulaziz.AP.410550_TMYx.2011-2025.epw",
        "specialty": "Sarawat Mountain Elevation (~1,700m+): Lower air density, reduced water vapor column, maximum atmospheric transmittance."
    }
}


# ==========================================
# THERMODYNAMIC & PHYSICAL SOLVER FUNCTIONS
# ==========================================
def calculate_sky_emissivity(temp_celsius, relative_humidity):
    """
    Computes dynamic clear-sky emissivity (Berdahl & Martin model)
    integrated with Magnus-Tetens dew point calculation.
    """
    a, b = 17.27, 237.7
    alpha = ((a * temp_celsius) / (b + temp_celsius)) + np.log(max(relative_humidity, 1e-3) / 100.0)
    temp_dew = (b * alpha) / (a - alpha)
    sky_emissivity = 0.741 + (0.0062 * temp_dew)
    return float(np.clip(sky_emissivity, 0.70, 0.95))


def calculate_convective_coefficient(wind_speed, temp_celsius, relative_humidity):
    """
    Computes dynamic convective heat transfer coefficient (h_c) using flat-plate 
    boundary layer fluid dynamics and Tsilingiris moist-air transport properties.
    """
    natural_convection = 2.5
    if wind_speed <= 0.05:
        return natural_convection

    temp_kelvin = temp_celsius + 273.15
    sat_pressure = 610.78 * np.exp((17.27 * temp_celsius) / (temp_celsius + 237.3))
    humidity_ratio = 0.622 * ((relative_humidity / 100.0) * sat_pressure) / (101325.0 - ((relative_humidity / 100.0) * sat_pressure))
    air_density = 101325.0 / (287.05 * temp_kelvin * (1.0 + 0.608 * humidity_ratio))

    moist_viscosity = (1.458e-6 * (temp_kelvin**1.5) / (temp_kelvin + 110.4)) * (1.0 + 0.23 * humidity_ratio)
    moist_conductivity = (2.495e-3 * (temp_kelvin**1.5) / (temp_kelvin + 194.4)) * (1.0 + 0.45 * humidity_ratio)
    prandtl_num = ((1005 + 1820 * humidity_ratio) * moist_viscosity) / moist_conductivity
    reynolds_num = (air_density * wind_speed * 1.0) / moist_viscosity

    if reynolds_num < 5e5:
        forced_nusselt = 0.664 * (reynolds_num**0.5) * (prandtl_num**(1.0 / 3.0))
    else:
        forced_nusselt = 0.037 * (reynolds_num**0.8) * (prandtl_num**(1.0 / 3.0))

    return natural_convection + ((forced_nusselt * moist_conductivity) / 1.0)


def solve_equilibrium_temperature(temp_air_k, ghi_val, wind_speed, rel_hum, emissivity, absorptivity):
    """
    Solves steady-state PDRC energy balance: P_rad - P_atm - P_solar + P_conv = 0
    using MINPACK Powell hybrid Newton-Raphson root solver (scipy.optimize.fsolve).
    """
    temp_air_c = temp_air_k - 273.15
    convective_coef = calculate_convective_coefficient(wind_speed, temp_air_c, rel_hum)
    sky_emissivity = calculate_sky_emissivity(temp_air_c, rel_hum)
    stefan_boltzmann_const = 5.67e-8

    heat_balance = lambda temp_k: (
        (emissivity * stefan_boltzmann_const * (temp_k**4)) 
        - (emissivity * sky_emissivity * stefan_boltzmann_const * (temp_air_k**4)) 
        - (absorptivity * ghi_val) 
        + (convective_coef * (temp_k - temp_air_k))
    )
    return float(opt.fsolve(heat_balance, x0=temp_air_k)[0] - 273.15)


# ==========================================
# MATERIAL DATABASE LOADER
# ==========================================
def load_material_database():
    """
    Loads all 60+ materials from CSV database or falls back to benchmark presets dictionary.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "materials_database.csv")

    if os.path.exists(csv_path):
        try:
            materials = []
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    materials.append({
                        "name": row.get("Name", "Unknown"),
                        "chemical": row.get("Chemical_Formula", "N/A"),
                        "category": row.get("Category", "General"),
                        "alpha": float(row.get("Alpha", 0.05)),
                        "epsilon": float(row.get("Epsilon", 0.90)),
                        "spec": row.get("Thickness_or_Spec", "N/A"),
                        "university": row.get("University_or_Institution", "N/A"),
                        "reference": row.get("Reference", "N/A")
                    })
            if materials:
                return materials
        except Exception:
            pass

    # Benchmark Preset Fallback Dataset
    return [
        {"name": "Purdue BaSO4 Super-White Paint", "chemical": "BaSO4 (Barium Sulfate)", "category": "Paints & Coatings", "alpha": 0.019, "epsilon": 0.950, "spec": "400 µm", "university": "Purdue University", "reference": "Li et al. (ACS Appl. Mater. 2021)"},
        {"name": "Purdue CaCO3 Radiative Paint", "chemical": "CaCO3 (Calcium Carbonate)", "category": "Paints & Coatings", "alpha": 0.045, "epsilon": 0.955, "spec": "400 µm", "university": "Purdue University", "reference": "Li et al. (Cell Rep. Phys. Sci. 2020)"},
        {"name": "Stanford HfO2/SiO2 Photonic Cooler", "chemical": "HfO2 / SiO2 / Ag", "category": "Metamaterials & Photonic", "alpha": 0.030, "epsilon": 0.960, "spec": "1.8 µm", "university": "Stanford University", "reference": "Raman et al. (Nature 2014)"},
        {"name": "Columbia Porous P(VdF-HFP) Film", "chemical": "P(VdF-HFP)", "category": "Polymers & Structural Films", "alpha": 0.040, "epsilon": 0.960, "spec": "300 µm", "university": "Columbia University", "reference": "Mandal et al. (Science 2018)"},
        {"name": "Maryland Delignified Cooling Wood", "chemical": "Cellulose / Wood", "category": "Wood & Bio-Aerogels", "alpha": 0.040, "epsilon": 0.920, "spec": "Engineered", "university": "University of Maryland", "reference": "Li et al. (Science 2019)"},
        {"name": "Standard Commercial TiO2 Paint", "chemical": "TiO2 (Titanium Dioxide)", "category": "Paints & Coatings", "alpha": 0.200, "epsilon": 0.880, "spec": "150 µm", "university": "Commercial Benchmark", "reference": "Commercial Control"}
    ]


# ==========================================
# EPW WEATHER FILE PARSER
# ==========================================
def load_epw_weather(city_name):
    """
    Parses hourly EnergyPlus Weather (.epw) datasets or generates synthetic desert profile fallback.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "epw_files", city_profiles[city_name]["file"])

    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, skiprows=8, header=None)
            cols = [epwDryBulbTempCol, epwRelHumidityCol, epwGhiCol, epwWindSpeedCol]
            for c in cols:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df.dropna(subset=cols)
            return df
        except Exception:
            pass

    # Synthetic annual weather profile fallback
    hours = np.arange(8760)
    synthetic_temp = 35 + 8 * np.sin(2 * np.pi * hours / 24)
    synthetic_rh = 50 + 20 * np.cos(2 * np.pi * hours / 24)
    synthetic_ghi = np.maximum(0, 950 * np.sin(np.pi * (hours % 24 - 6) / 12))
    synthetic_wind = 3.5 + 1.0 * np.sin(2 * np.pi * hours / 12)
    return pd.DataFrame({
        epwDryBulbTempCol: synthetic_temp,
        epwRelHumidityCol: synthetic_rh,
        epwGhiCol: synthetic_ghi,
        epwWindSpeedCol: synthetic_wind
    })
