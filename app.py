import os
import csv
import pandas as pd
import numpy as np
import scipy.optimize as opt

# Web Framework & Plotting
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# DIRECT IMPORT FROM prc.py
# ==========================================
try:
    from prc import (
        epwDryBulbTempCol, 
        epwRelHumidityCol, 
        epwGhiCol, 
        epwWindSpeedCol
    )
except ImportError:
    epwDryBulbTempCol = 6
    epwRelHumidityCol = 8
    epwGhiCol = 13
    epwWindSpeedCol = 21


# ==========================================
# PAGE CONFIG & PROFESSIONAL STYLING
# ==========================================
st.set_page_config(
    page_title="Saudi Arabia Radiative Cooling Simulator",
    page_icon="❄️",
    layout="wide"
)

# Custom CSS for High-Contrast, Professional Design
st.markdown("""
<style>
    /* Main Background */
    body, .stApp {
        background-color: #f8fafc !important;
        color: #0f172a !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Header Styling */
    .app-header-title {
        font-size: 24px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 2px;
    }
    .app-header-subtitle {
        font-size: 12.5px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 20px;
    }

    /* Section Cards */
    .card-container {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    
    .section-header {
        font-size: 15px;
        font-weight: 700;
        color: #0f172a;
        border-left: 4px solid #0f172a;
        padding-left: 10px;
        margin-bottom: 14px;
    }

    /* Executive KPI Cards */
    .kpi-card {
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 12px;
        text-align: center;
    }
    .kpi-label {
        font-size: 11px;
        font-weight: 700;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 20px;
        font-weight: 800;
        color: #0f172a;
    }

    /* Primary Buttons */
    .stButton > button {
        background-color: #0f172a !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        border-radius: 4px !important;
        border: 1px solid #0f172a !important;
        padding: 8px 16px !important;
        width: 100%;
        transition: all 0.15s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #334155 !important;
        border-color: #334155 !important;
        color: #ffffff !important;
    }

    /* FIX FOR 60+ POPOVER BUTTON & TEXT CONTRAST */
    div[data-testid="stPopover"] button,
    div[data-testid="stPopover"] button:hover,
    div[data-testid="stPopover"] button:focus,
    div[data-testid="stPopover"] button:active {
        background-color: #0f172a !important;
        color: #ffffff !important;
        border: 1px solid #0f172a !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        box-shadow: none !important;
    }
    div[data-testid="stPopover"] button p {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* Modal Popover Interior High-Contrast Fix */
    div[data-testid="stPopoverBody"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.12) !important;
    }
    div[data-testid="stPopoverBody"] p, 
    div[data-testid="stPopoverBody"] h4, 
    div[data-testid="stPopoverBody"] h5, 
    div[data-testid="stPopoverBody"] span, 
    div[data-testid="stPopoverBody"] label {
        color: #0f172a !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# THERMAL PHYSICS CALCULATIONS (FROM prc.py)
# ==========================================
def calculate_sky_emissivity(temp_celsius, relative_humidity):
    a, b = 17.27, 237.7
    alpha = ((a * temp_celsius) / (b + temp_celsius)) + np.log(max(relative_humidity, 1e-3) / 100.0)
    temp_dew = (b * alpha) / (a - alpha)
    sky_emissivity = 0.741 + (0.0062 * temp_dew)
    return float(np.clip(sky_emissivity, 0.70, 0.95))

def calculate_convective_coefficient(wind_speed, temp_celsius, relative_humidity):
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
        forced_nusselt = 0.664 * (reynolds_num**0.5) * (prandtl_num**(1.0/3.0))
    else:
        forced_nusselt = 0.037 * (reynolds_num**0.8) * (prandtl_num**(1.0/3.0))

    return natural_convection + ((forced_nusselt * moist_conductivity) / 1.0)

def solve_equilibrium_temperature(temp_air_k, ghi_val, wind_speed, rel_hum, emissivity, absorptivity):
    temp_air_c = temp_air_k - 273.15
    convective_coef = calculate_convective_coefficient(wind_speed, temp_air_c, rel_hum)
    sky_emissivity = calculate_sky_emissivity(temp_air_c, rel_hum)
    stefan_boltzmann_const = 5.67e-8

    heat_balance = lambda temp_k: (emissivity * stefan_boltzmann_const * (temp_k**4)) - (emissivity * sky_emissivity * stefan_boltzmann_const * (temp_air_k**4)) - (absorptivity * ghi_val) + (convective_coef * (temp_k - temp_air_k))
    return float(opt.fsolve(heat_balance, x0=temp_air_k)[0] - 273.15)


# ==========================================
# DATABASE & WEATHER FILE LOADERS
# ==========================================
@st.cache_data
def load_material_database():
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

    # Embedded Fallback
    return [
        {"name": "Purdue BaSO4 Super-White Paint", "chemical": "BaSO4 (Barium Sulfate)", "category": "Paints & Coatings", "alpha": 0.019, "epsilon": 0.950, "spec": "400 µm", "university": "Purdue University", "reference": "Li et al. (ACS Appl. Mater. 2021)"},
        {"name": "Purdue CaCO3 Radiative Paint", "chemical": "CaCO3 (Calcium Carbonate)", "category": "Paints & Coatings", "alpha": 0.045, "epsilon": 0.955, "spec": "400 µm", "university": "Purdue University", "reference": "Li et al. (Cell Rep. Phys. Sci. 2020)"},
        {"name": "Stanford HfO2/SiO2 Photonic Cooler", "chemical": "HfO2 / SiO2 / Ag", "category": "Metamaterials & Photonic", "alpha": 0.030, "epsilon": 0.960, "spec": "1.8 µm", "university": "Stanford University", "reference": "Raman et al. (Nature 2014)"},
        {"name": "Columbia Porous P(VdF-HFP) Film", "chemical": "P(VdF-HFP)", "category": "Polymers & Structural Films", "alpha": 0.040, "epsilon": 0.960, "spec": "300 µm", "university": "Columbia University", "reference": "Mandal et al. (Science 2018)"},
        {"name": "Maryland Delignified Cooling Wood", "chemical": "Cellulose / Wood", "category": "Wood & Bio-Aerogels", "alpha": 0.040, "epsilon": 0.920, "spec": "Engineered", "university": "University of Maryland", "reference": "Li et al. (Science 2019)"},
        {"name": "Standard Commercial TiO2 Paint", "chemical": "TiO2 (Titanium Dioxide)", "category": "Paints & Coatings", "alpha": 0.200, "epsilon": 0.880, "spec": "150 µm", "university": "Commercial Benchmark", "reference": "Commercial Control"}
    ]

materials_db = load_material_database()

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

@st.cache_data
def load_epw_weather(city_name):
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

    # Synthetic fallback profile
    hours = np.arange(8760)
    synthetic_temp = 35 + 8 * np.sin(2 * np.pi * hours / 24)
    synthetic_rh = 50 + 20 * np.cos(2 * np.pi * hours / 24)
    synthetic_ghi = np.maximum(0, 950 * np.sin(np.pi * (hours % 24 - 6) / 12))
    synthetic_wind = 3.5 + 1.0 * np.sin(2 * np.pi * hours / 12)
    return pd.DataFrame({epwDryBulbTempCol: synthetic_temp, epwRelHumidityCol: synthetic_rh, epwGhiCol: synthetic_ghi, epwWindSpeedCol: synthetic_wind})


# ==========================================
# INITIAL STATE MANAGEMENT
# ==========================================
if "slot_0_name" not in st.session_state:
    st.session_state["slot_0_name"] = "Standard Commercial TiO2 Paint [TiO2 (Titanium Dioxide)]"
    st.session_state["slot_0_eps"] = "0.88"
    st.session_state["slot_0_alp"] = "0.20"

if "slot_1_name" not in st.session_state:
    st.session_state["slot_1_name"] = "Purdue BaSO4 Super-White Paint [BaSO4 (Barium Sulfate)]"
    st.session_state["slot_1_eps"] = "0.95"
    st.session_state["slot_1_alp"] = "0.019"

if "slot_2_name" not in st.session_state:
    st.session_state["slot_2_name"] = "Stanford HfO2/SiO2 Photonic Cooler [HfO2 / SiO2 / Ag]"
    st.session_state["slot_2_eps"] = "0.96"
    st.session_state["slot_2_alp"] = "0.03"

if "results_data" not in st.session_state:
    st.session_state["results_data"] = None

if "active_plot" not in st.session_state:
    st.session_state["active_plot"] = None


# ==========================================
# HEADER SECTION
# ==========================================
st.markdown('<div class="app-header-title">Saudi Arabia Radiative Cooling Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="app-header-subtitle">Passive Daytime Radiative Cooling (PDRC) Thermal Workstation</div>', unsafe_allow_html=True)


# --- SECTION 1: Regional Climate Setup ---
st.markdown("""
<div class="card-container">
    <div class="section-header">Regional Climate Setup</div>
</div>
""", unsafe_allow_html=True)

f1_col1, f1_col2 = st.columns([1.3, 2.7])
with f1_col1:
    current_city = st.selectbox("Select Meteorological Region:", list(city_profiles.keys()), key="city_select")

city_info = city_profiles[current_city]
st.markdown(f"<p style='color: #475569; font-size: 12.5px; font-style: italic; margin-top: -10px; margin-bottom: 15px;'>{city_info['specialty']}</p>", unsafe_allow_html=True)

weather_df = load_epw_weather(current_city)
climate_scenarios = {
    "Absolute Hottest Day": weather_df[epwDryBulbTempCol].idxmax(),
    "Maximum Solar Peak Day": weather_df[epwGhiCol].idxmax()
}


# --- SECTION 2: Target Environmental Evaluation Scenario ---
st.markdown("""
<div class="card-container">
    <div class="section-header">Evaluation Scenario Case Study</div>
</div>
""", unsafe_allow_html=True)

current_scenario = st.selectbox("Select Profile Case Study:", list(climate_scenarios.keys()), key="scenario_select")

scenario_row = weather_df.loc[climate_scenarios[current_scenario]]
tair_c = float(scenario_row[epwDryBulbTempCol])
rh_pct = float(scenario_row[epwRelHumidityCol])
ghi_wm2 = float(scenario_row[epwGhiCol])
wind_ms = float(scenario_row[epwWindSpeedCol])
tair_k = tair_c + 273.15


# --- SECTION 3: Peak Regional Climate Benchmarks ---
st.markdown("""
<div class="card-container">
    <div class="section-header">Peak Climate Benchmarks</div>
</div>
""", unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Solar Irradiance (GHI)</div>
        <div class="kpi-value">{ghi_wm2:.1f} W/m²</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Ambient Air Temp</div>
        <div class="kpi-value">{tair_c:.1f} °C</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Wind Speed</div>
        <div class="kpi-value">{wind_ms:.1f} m/s</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Relative Humidity</div>
        <div class="kpi-value">{rh_pct:.1f} %</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")


# --- SECTION 4: Core Materials Matrix & Database ---
st.markdown("""
<div class="card-container">
    <div class="section-header">Core Materials Matrix & Database</div>
</div>
""", unsafe_allow_html=True)

hdr_col1, hdr_col2, hdr_col3 = st.columns([3, 1, 1])
hdr_col1.markdown("**Material / Surface Label Selection**")
hdr_col2.markdown("<div style='text-align: center;'><b>Emissivity (ε)</b></div>", unsafe_allow_html=True)
hdr_col3.markdown("<div style='text-align: center;'><b>Absorptivity (α)</b></div>", unsafe_allow_html=True)

# Row 1
m1_col1, m1_col2, m1_col3 = st.columns([3, 1, 1])
mat0_name = m1_col1.text_input("Name 1", value=st.session_state["slot_0_name"], label_visibility="collapsed")
mat0_eps = m1_col2.text_input("Eps 1", value=st.session_state["slot_0_eps"], label_visibility="collapsed")
mat0_alp = m1_col3.text_input("Alp 1", value=st.session_state["slot_0_alp"], label_visibility="collapsed")

# Row 2
m2_col1, m2_col2, m2_col3 = st.columns([3, 1, 1])
mat1_name = m2_col1.text_input("Name 2", value=st.session_state["slot_1_name"], label_visibility="collapsed")
mat1_eps = m2_col2.text_input("Eps 2", value=st.session_state["slot_1_eps"], label_visibility="collapsed")
mat1_alp = m2_col3.text_input("Alp 2", value=st.session_state["slot_1_alp"], label_visibility="collapsed")

# Row 3
m3_col1, m3_col2, m3_col3 = st.columns([3, 1, 1])
mat2_name = m3_col1.text_input("Name 3", value=st.session_state["slot_2_name"], label_visibility="collapsed")
mat2_eps = m3_col2.text_input("Eps 3", value=st.session_state["slot_2_eps"], label_visibility="collapsed")
mat2_alp = m3_col3.text_input("Alp 3", value=st.session_state["slot_2_alp"], label_visibility="collapsed")

# Search Database Popover
with st.popover("🔍 Search & Filter 60+ Literature Material Database (University, Chemical, Alpha, Epsilon)", use_container_width=True):
    st.markdown("<h4 style='color:#0f172a !important;'>Global Radiative Cooling Material Library</h4>", unsafe_allow_html=True)
    
    p_col1, p_col2 = st.columns(2)
    search_q = p_col1.text_input("Search Name / Chemical Formula:", value="")
    
    univ_list = ["All Institutions"] + sorted(list(set([m["university"] for m in materials_db])))
    selected_univ = p_col2.selectbox("Filter by Institution:", univ_list)
    
    p_col3, p_col4 = st.columns(2)
    a_min, a_max = p_col3.slider("Absorptivity (α) Range:", 0.0, 1.0, (0.0, 1.0), step=0.01)
    e_min, e_max = p_col4.slider("Emissivity (ε) Range:", 0.0, 1.0, (0.0, 1.0), step=0.01)

    # Filter Data
    filtered_mats = []
    for m in materials_db:
        u_match = (selected_univ == "All Institutions") or (m["university"] == selected_univ)
        t_search = f"{m['name']} {m['chemical']} {m['reference']}".lower()
        t_match = (not search_q) or (search_q.lower() in t_search)
        num_match = (a_min <= m["alpha"] <= a_max) and (e_min <= m["epsilon"] <= e_max)
        if u_match and t_match and num_match:
            filtered_mats.append(m)

    df_display = pd.DataFrame(filtered_mats)
    if not df_display.empty:
        st.dataframe(df_display[["name", "chemical", "spec", "university", "alpha", "epsilon", "reference"]], use_container_width=True, hide_index=True)
        
        st.markdown("<b style='color:#0f172a !important;'>Assign Selected Material to Slot:</b>", unsafe_allow_html=True)
        sel_col1, sel_col2, sel_col3 = st.columns([2, 1, 1])
        chosen_mat_name = sel_col1.selectbox("Select Material", df_display["name"].tolist())
        target_slot = sel_col2.radio("Slot", ["Slot 1", "Slot 2", "Slot 3"], horizontal=True)
        
        if sel_col3.button("Apply Material"):
            match_row = next(m for m in filtered_mats if m["name"] == chosen_mat_name)
            slot_idx = {"Slot 1": "0", "Slot 2": "1", "Slot 3": "2"}[target_slot]
            st.session_state[f"slot_{slot_idx}_name"] = f"{match_row['name']} [{match_row['chemical']}]"
            st.session_state[f"slot_{slot_idx}_eps"] = str(match_row["epsilon"])
            st.session_state[f"slot_{slot_idx}_alp"] = str(match_row["alpha"])
            st.success(f"Loaded '{chosen_mat_name}' into {target_slot}!")
            st.rerun()


# --- SECTION 5: Computed Equilibrium Results Output ---
st.markdown("""
<div class="card-container">
    <div class="section-header">Computed Equilibrium Temperature Results</div>
</div>
""", unsafe_allow_html=True)

if st.session_state["results_data"]:
    # Building HTML table without leading line indentation to avoid markdown code-block escaping
    table_html = (
        '<table style="width:100%; border-collapse: collapse; font-family: sans-serif; font-size: 13.5px; border: 1px solid #cbd5e1; border-radius: 4px; overflow: hidden;">'
        '<thead>'
        '<tr style="background-color: #0f172a; color: #ffffff; text-align: left; font-weight: bold;">'
        '<th style="padding: 10px 14px;">Material Configuration</th>'
        '<th style="padding: 10px 14px; text-align: center;">Emissivity (ε)</th>'
        '<th style="padding: 10px 14px; text-align: center;">Absorptivity (α)</th>'
        '<th style="padding: 10px 14px; text-align: center;">Equilibrium Temp</th>'
        '<th style="padding: 10px 14px; text-align: center;">Thermal Delta</th>'
        '<th style="padding: 10px 14px;">Performance Status</th>'
        '</tr>'
        '</thead>'
        '<tbody>'
    )
    
    for idx, row in enumerate(st.session_state["results_data"]):
        bg_color = "#ffffff" if idx % 2 == 0 else "#f8fafc"
        badge_style = "background-color: #dcfce7; color: #15803d;" if row["is_cooling"] else "background-color: #fee2e2; color: #b91c1c;"
        
        table_html += f'<tr style="background-color: {bg_color}; border-bottom: 1px solid #e2e8f0; color: #0f172a;"><td style="padding: 10px 14px; font-weight: 600;">{row["name"]}</td><td style="padding: 10px 14px; text-align: center;">{row["eps"]:.3f}</td><td style="padding: 10px 14px; text-align: center;">{row["alp"]:.3f}</td><td style="padding: 10px 14px; text-align: center; font-weight: 700;">{row["eq_c"]:.2f} °C</td><td style="padding: 10px 14px; text-align: center; font-weight: 700;">{row["delta"]:+.2f} °C</td><td style="padding: 10px 14px;"><span style="{badge_style} padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 700;">{row["status"]}</span></td></tr>'
        
    table_html += '</tbody></table>'
    st.markdown(table_html, unsafe_allow_html=True)
else:
    st.info("Select a city, set up your materials, and click 'Run Equilibrium Calculation' below.")

st.write("")


# ==========================================
# BOTTOM ACTION BUTTON BAR & LOGIC
# ==========================================
btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

def get_current_materials():
    try:
        return [
            {"name": mat0_name.strip(), "eps": float(mat0_eps), "alp": float(mat0_alp)},
            {"name": mat1_name.strip(), "eps": float(mat1_eps), "alp": float(mat1_alp)},
            {"name": mat2_name.strip(), "eps": float(mat2_eps), "alp": float(mat2_alp)}
        ]
    except ValueError as err:
        st.error(f"Input Validation Error: Please enter valid numeric values for Emissivity and Absorptivity.\n{err}")
        return None

# Action 1: Calculation
if btn_col1.button("Run Equilibrium Calculation"):
    mats = get_current_materials()
    if mats:
        rows = []
        for mat in mats:
            eq_c = solve_equilibrium_temperature(tair_k, ghi_wm2, wind_ms, rh_pct, mat["eps"], mat["alp"])
            delta = eq_c - tair_c
            is_cooling = delta < 0
            status_str = f"Sub-ambient Cooling ({delta:+.2f}°C)" if is_cooling else f"Heating Penalty ({delta:+.2f}°C)"
            
            rows.append({
                "name": mat["name"],
                "eps": mat["eps"],
                "alp": mat["alp"],
                "eq_c": eq_c,
                "delta": delta,
                "is_cooling": is_cooling,
                "status": status_str
            })
            
        st.session_state["results_data"] = rows
        st.session_state["active_plot"] = None
        st.rerun()

# Action 2: Diurnal Plot
if btn_col2.button("Plot Diurnal Performance Profile"):
    st.session_state["active_plot"] = "diurnal"

# Action 3: Sensitivity Sweeps Plot
if btn_col3.button("Plot Sensitivity Sweeps"):
    st.session_state["active_plot"] = "sensitivity"

# Action 4: Wind Sweep Plot
if btn_col4.button("Plot Fixed-Context Wind Sweep"):
    st.session_state["active_plot"] = "wind"


# ==========================================
# MATPLOTLIB GRAPH OUTPUT DISPLAY
# ==========================================
materials_mat = get_current_materials()

if st.session_state["active_plot"] and materials_mat:
    st.divider()
    
    # --- PLOT 1: DIURNAL PROFILE ---
    if st.session_state["active_plot"] == "diurnal":
        st.markdown(f"#### Diurnal Performance Profile - {current_city}")
        
        start_idx = (climate_scenarios[current_scenario] // 24) * 24
        day_df = weather_df.iloc[start_idx : start_idx + 24]
        hours = np.arange(1, len(day_df) + 1)
        amb_temps = day_df[epwDryBulbTempCol].values
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 5.2), dpi=100)
        
        ax1.plot(hours, amb_temps, 'k--', lw=2.5, label='Ambient Air Temp Baseline')
        colors, markers = ['#e63946', '#2a9d8f', '#457b9d'], ['-o', '-s', '-^']
        
        for i, mat in enumerate(materials_mat):
            sim_profile = [
                solve_equilibrium_temperature(row[epwDryBulbTempCol] + 273.15, row[epwGhiCol], row[epwWindSpeedCol], row[epwRelHumidityCol], mat["eps"], mat["alp"])
                for _, row in day_df.iterrows()
            ]
            ax1.plot(hours, sim_profile, markers[i % 3], color=colors[i % 3], lw=1.8, label=f"{mat['name']} (ε={mat['eps']}, α={mat['alp']})")

        ax1.set_xlabel('Hour of Day', fontweight='bold')
        ax1.set_ylabel('Steady State Temp (°C)', fontweight='bold')
        ax1.set_xticks(hours)
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(loc='upper left', fontsize='x-small')
        ax1.set_title("Equilibrium Thermal Response", fontsize=10, fontweight='bold')

        ax2.plot(hours, day_df[epwGhiCol].values, color='#f39c12', lw=2, marker='o', label='Solar GHI (W/m²)')
        ax2.set_xlabel('Hour of Day', fontweight='bold')
        ax2.set_ylabel('Solar Irradiance (GHI) [W/m²]', color='#f39c12', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#f39c12')
        ax2.set_xticks(hours)
        ax2.grid(True, linestyle=':', alpha=0.4)

        twin2 = ax2.twinx()
        twin2.plot(hours, amb_temps, color='#e74c3c', lw=2, linestyle='-.', label='Ambient Temp (°C)')
        twin2.plot(hours, day_df[epwWindSpeedCol].values, color='#3498db', lw=2, linestyle=':', label='Wind Speed (m/s)')
        twin2.plot(hours, day_df[epwRelHumidityCol].values, color='#2ecc71', lw=1.8, label='Relative Humidity (%)')
        twin2.set_ylabel('Temp / Wind / Humidity', fontweight='bold')

        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = twin2.get_legend_handles_labels()
        twin2.legend(h1 + h2, l1 + l2, loc='upper right', fontsize='x-small')
        ax2.set_title("Active Meteorological Variables", fontsize=10, fontweight='bold')

        fig.suptitle(f"Diurnal Performance Profile - {current_city}", fontsize=12, fontweight='bold')
        fig.tight_layout()
        st.pyplot(fig)

    # --- PLOT 2: SENSITIVITY SWEEPS ---
    elif st.session_state["active_plot"] == "sensitivity":
        st.markdown("#### Sensitivity Analysis Dashboard")
        
        ghi_sweep = np.linspace(0, 1000, 50)
        wind_sweep = np.linspace(0.1, 12, 50)
        colors, markers = ['#e63946', '#2a9d8f', '#457b9d'], ['o', 's', '^']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5), dpi=100)

        for i, mat in enumerate(materials_mat):
            ax1.plot(ghi_sweep, [solve_equilibrium_temperature(tair_k, g, wind_ms, rh_pct, mat["eps"], mat["alp"]) for g in ghi_sweep], color=colors[i % 3], marker=markers[i % 3], markevery=5, lw=2, label=mat["name"])
            ax2.plot(wind_sweep, [solve_equilibrium_temperature(tair_k, ghi_wm2, w, rh_pct, mat["eps"], mat["alp"]) for w in wind_sweep], color=colors[i % 3], marker=markers[i % 3], markevery=5, lw=2, label=mat["name"])

        for ax, x_label, title in [(ax1, "Solar Radiation Load (GHI) [W/m²]", "Sensitivity vs. Solar Irradiance Load"), (ax2, "Convective Wind Speed [m/s]", "Sensitivity vs. Wind Convection")]:
            ax.axhline(tair_c, color='black', linestyle='--', alpha=0.7, label="Ambient Baseline")
            ax.set_xlabel(x_label, fontweight='bold')
            ax.set_ylabel("Equilibrium Temperature (°C)", fontweight='bold')
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='lower left', fontsize='small')

        fig.suptitle(f"Parametric Sensitivity Dashboard ({tair_c:.1f}°C Ambient)", fontsize=11, fontweight='bold')
        fig.tight_layout()
        st.pyplot(fig)

    # --- PLOT 3: FIXED WIND SWEEP ---
    elif st.session_state["active_plot"] == "wind":
        st.markdown("#### Wind Speed Sensitivity")
        
        st.info(f"• Solar Irradiance (GHI): {ghi_wm2:.1f} W/m²   |   • Ambient Air Temperature: {tair_c:.1f}°C   |   • Boundary Relative Humidity: {rh_pct:.1f}%")
        
        fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=100)
        wind_sweep = np.linspace(0.1, 12, 50)
        colors, markers = ['#e63946', '#2a9d8f', '#457b9d'], ['o', 's', '^']

        for i, mat in enumerate(materials_mat):
            ax.plot(wind_sweep, [solve_equilibrium_temperature(tair_k, ghi_wm2, w, rh_pct, mat["eps"], mat["alp"]) for w in wind_sweep], color=colors[i % 3], marker=markers[i % 3], markevery=5, lw=2, label=mat["name"])

        ax.axhline(tair_c, color='black', linestyle='--', alpha=0.7, label=f"Ambient Baseline ({tair_c:.1f}°C)")
        ax.set_xlabel("Convective Wind Speed [m/s]", fontweight='bold')
        ax.set_ylabel("Equilibrium Temperature (°C)", fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='lower left', fontsize='small')

        fig.tight_layout()
        st.pyplot(fig)
