import os
import csv
import pandas as pd
import numpy as np

# Web Framework & Plotting
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================
# DIRECT IMPORT FROM PHYSICS ENGINE (PRC.py)
# ==========================================
from PRC import (
    epwDryBulbTempCol,
    epwRelHumidityCol,
    epwGhiCol,
    epwWindSpeedCol,
    city_profiles,
    calculate_sky_emissivity,
    calculate_convective_coefficient,
    solve_equilibrium_temperature,
    load_material_database,
    load_epw_weather
)


# ==========================================
# PAGE CONFIGURATION & STYLING
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

    /* Streamlit Popover Button Contrast Overrides */
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
# STREAMLIT CACHED WRAPPERS
# ==========================================
@st.cache_data
def get_cached_materials():
    return load_material_database()

@st.cache_data
def get_cached_weather(city_name):
    return load_epw_weather(city_name)

materials_db = get_cached_materials()


# ==========================================
# INITIAL SESSION STATE MANAGEMENT
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
# APPLICATION HEADER
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

weather_df = get_cached_weather(current_city)
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
        <div class="kpi-value">{tair_c:.2f} °C</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Wind Speed</div>
        <div class="kpi-value">{wind_ms:.2f} m/s</div>
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

# Row 1 Slot
m1_col1, m1_col2, m1_col3 = st.columns([3, 1, 1])
mat0_name = m1_col1.text_input("Name 1", value=st.session_state["slot_0_name"], label_visibility="collapsed")
mat0_eps = m1_col2.text_input("Eps 1", value=st.session_state["slot_0_eps"], label_visibility="collapsed")
mat0_alp = m1_col3.text_input("Alp 1", value=st.session_state["slot_0_alp"], label_visibility="collapsed")

# Row 2 Slot
m2_col1, m2_col2, m2_col3 = st.columns([3, 1, 1])
mat1_name = m2_col1.text_input("Name 2", value=st.session_state["slot_1_name"], label_visibility="collapsed")
mat1_eps = m2_col2.text_input("Eps 2", value=st.session_state["slot_1_eps"], label_visibility="collapsed")
mat1_alp = m2_col3.text_input("Alp 2", value=st.session_state["slot_1_alp"], label_visibility="collapsed")

# Row 3 Slot
m3_col1, m3_col2, m3_col3 = st.columns([3, 1, 1])
mat2_name = m3_col1.text_input("Name 3", value=st.session_state["slot_2_name"], label_visibility="collapsed")
mat2_eps = m3_col2.text_input("Eps 3", value=st.session_state["slot_2_eps"], label_visibility="collapsed")
mat2_alp = m3_col3.text_input("Alp 3", value=st.session_state["slot_2_alp"], label_visibility="collapsed")

# 60+ Literature Material Database Popover Dialog
with st.popover("🔍 Search & Filter 60+ Literature Material Database (University, Chemical, Alpha, Epsilon)", use_container_width=True):
    st.markdown("<h4 style='color:#0f172a !important;'>Global Radiative Cooling Material Library</h4>", unsafe_allow_html=True)
    
    p_col1, p_col2 = st.columns(2)
    search_q = p_col1.text_input("Search Name / Chemical Formula:", value="")
    
    univ_list = ["All Institutions"] + sorted(list(set([m["university"] for m in materials_db])))
    selected_univ = p_col2.selectbox("Filter by Institution:", univ_list)
    
    p_col3, p_col4 = st.columns(2)
    a_min, a_max = p_col3.slider("Absorptivity (α) Range:", 0.0, 1.0, (0.0, 1.0), step=0.01)
    e_min, e_max = p_col4.slider("Emissivity (ε) Range:", 0.0, 1.0, (0.0, 1.0), step=0.01)

    # Multi-parametric Filtering
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
        
        table_html += (
            f'<tr style="background-color: {bg_color}; border-bottom: 1px solid #e2e8f0; color: #0f172a;">'
            f'<td style="padding: 10px 14px; font-weight: 600;">{row["name"]}</td>'
            f'<td style="padding: 10px 14px; text-align: center;">{row["eps"]:.3f}</td>'
            f'<td style="padding: 10px 14px; text-align: center;">{row["alp"]:.3f}</td>'
            f'<td style="padding: 10px 14px; text-align: center; font-weight: 700;">{row["eq_c"]:.2f} °C</td>'
            f'<td style="padding: 10px 14px; text-align: center; font-weight: 700;">{row["delta"]:+.2f} °C</td>'
            f'<td style="padding: 10px 14px;"><span style="{badge_style} padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 700;">{row["status"]}</span></td>'
            f'</tr>'
        )
        
    table_html += '</tbody></table>'
    st.markdown(table_html, unsafe_allow_html=True)
else:
    st.info("Select a city, set up your materials, and click 'Run Equilibrium Calculation' below.")

st.write("")


# ==========================================
# 3-BUTTON ACTION BAR & SIMULATION HANDLERS
# ==========================================
btn_col1, btn_col2, btn_col3 = st.columns(3)

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

# Action 2: Diurnal Profile Plot Trigger
if btn_col2.button("Plot Diurnal Performance Profile"):
    st.session_state["active_plot"] = "diurnal"

# Action 3: Sensitivity Sweeps Dashboard Plot Trigger (Dual Panel: GHI & Wind Sweeps)
if btn_col3.button("Plot Sensitivity Sweeps"):
    st.session_state["active_plot"] = "sensitivity"


# ==========================================
# MATPLOTLIB GRAPH VISUALIZATION ENGINE
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
        
        ax1.plot(hours, amb_temps, color='black', linestyle='--', lw=2.5, label='Ambient Air Temp Baseline')
        colors = ['#e63946', '#2a9d8f', '#457b9d']
        
        for i, mat in enumerate(materials_mat):
            sim_profile = [
                solve_equilibrium_temperature(row[epwDryBulbTempCol] + 273.15, row[epwGhiCol], row[epwWindSpeedCol], row[epwRelHumidityCol], mat["eps"], mat["alp"])
                for _, row in day_df.iterrows()
            ]
            ax1.plot(hours, sim_profile, color=colors[i % 3], linestyle='-', lw=2.2, label=f"{mat['name']} (ε={mat['eps']}, α={mat['alp']})")

        ax1.set_xlabel('Hour of Day', fontweight='bold')
        ax1.set_ylabel('Steady State Temp (°C)', fontweight='bold')
        ax1.set_xticks(hours)
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(loc='upper left', fontsize='x-small')
        ax1.set_title("Equilibrium Thermal Response", fontsize=10, fontweight='bold')

        ax2.plot(hours, day_df[epwGhiCol].values, color='#f39c12', lw=2.2, label='Solar GHI (W/m²)')
        ax2.set_xlabel('Hour of Day', fontweight='bold')
        ax2.set_ylabel('Solar Irradiance (GHI) [W/m²]', color='#f39c12', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#f39c12')
        ax2.set_xticks(hours)
        ax2.grid(True, linestyle=':', alpha=0.4)

        twin2 = ax2.twinx()
        twin2.plot(hours, amb_temps, color='#e74c3c', lw=2.0, linestyle='-.', label='Ambient Temp (°C)')
        twin2.plot(hours, day_df[epwWindSpeedCol].values, color='#3498db', lw=2.0, linestyle=':', label='Wind Speed (m/s)')
        twin2.plot(hours, day_df[epwRelHumidityCol].values, color='#2ecc71', lw=1.8, linestyle='-', label='Relative Humidity (%)')
        twin2.set_ylabel('Temp / Wind / Humidity', fontweight='bold')

        h1, l1 = ax2.get_legend_handles_labels()
        h2, l2 = twin2.get_legend_handles_labels()
        twin2.legend(h1 + h2, l1 + l2, loc='upper right', fontsize='x-small')
        ax2.set_title("Active Meteorological Variables", fontsize=10, fontweight='bold')

        fig.suptitle(f"Diurnal Performance Profile - {current_city}", fontsize=12, fontweight='bold')
        fig.tight_layout()
        st.pyplot(fig)

    # --- PLOT 2: DUAL-PANEL SENSITIVITY SWEEPS DASHBOARD ---
    elif st.session_state["active_plot"] == "sensitivity":
        st.markdown("#### Parametric Sensitivity Dashboard")
        st.info(f"• Baseline Solar Irradiance (GHI): {ghi_wm2:.1f} W/m²   |   • Baseline Air Temperature: {tair_c:.2f}°C   |   • Boundary Relative Humidity: {rh_pct:.1f}%")

        # 100 evaluation points for high resolution smooth curves
        ghi_sweep = np.linspace(0.0, 1000.0, 100)
        wind_sweep = np.linspace(0.1, 12.0, 100)
        colors = ['#e63946', '#2a9d8f', '#457b9d']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.0), dpi=100)

        for i, mat in enumerate(materials_mat):
            # Left Subplot: Solar Irradiance Load Sweep
            ghi_results = [solve_equilibrium_temperature(tair_k, g, wind_ms, rh_pct, mat["eps"], mat["alp"]) for g in ghi_sweep]
            ax1.plot(ghi_sweep, ghi_results, color=colors[i % 3], linestyle='-', lw=2.2, label=mat["name"])

            # Right Subplot: Wind Speed Convection Sweep (Incropera Standard Model)
            wind_results = [solve_equilibrium_temperature(tair_k, ghi_wm2, w, rh_pct, mat["eps"], mat["alp"]) for w in wind_sweep]
            ax2.plot(wind_sweep, wind_results, color=colors[i % 3], linestyle='-', lw=2.2, label=mat["name"])

        # Format Subplot 1 (GHI)
        ax1.axhline(tair_c, color='black', linestyle='--', alpha=0.7, label=f"Ambient Baseline ({tair_c:.2f}°C)")
        ax1.set_xlabel("Solar Radiation Load (GHI) [W/m²]", fontweight='bold')
        ax1.set_ylabel("Equilibrium Temperature (°C)", fontweight='bold')
        ax1.set_title("Sensitivity vs. Solar Irradiance Load", fontsize=10, fontweight='bold')
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(loc='lower left', fontsize='small')

        # Format Subplot 2 (Wind Speed Convection)
        ax2.axhline(tair_c, color='black', linestyle='--', alpha=0.7, label=f"Ambient Baseline ({tair_c:.2f}°C)")
        ax2.set_xlabel("Convective Wind Speed [m/s]", fontweight='bold')
        ax2.set_ylabel("Equilibrium Temperature (°C)", fontweight='bold')
        ax2.set_title("Sensitivity vs. Wind Convection", fontsize=10, fontweight='bold')
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend(loc='lower left', fontsize='small')

        fig.suptitle(f"Parametric Sensitivity Dashboard ({current_city})", fontsize=11, fontweight='bold')
        fig.tight_layout()
        st.pyplot(fig)
