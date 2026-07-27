# Saudi Arabia Passive Daytime Radiative Cooling (PDRC) Simulator

An engineering workstation and web-based simulation engine for modeling sub-ambient passive daytime radiative cooling performance across Saudi Arabian microclimates using official EPW (EnergyPlus Weather) datasets.

---

## 🌐 Live Web Application
Access the deployed interactive simulator here:  
👉 **[Click Here to Launch Live Website]([[https://saudi-radiative-cooling-aefttvn5kmgbjvrhyblppm.streamlit.app]])**

---

## Project Overview
Passive Daytime Radiative Cooling (PDRC) permits surfaces to cool below ambient air temperature without external energy usage through solar radiation reflection ($0.3–2.5\ \mu\text{m}$) and emission of thermal radiation through the atmospheric transparency window ($8–13\ \mu\text{m}$).

This simulator utilizes steady-state thermal heat balance to model the thermal performance of custom and benchmark PDRC surfaces across the unique microclimates within Saudi Arabia. These include regions of extremely humid coastal conditions, high-elevation mountains with clear transmittance and low convection, northern maritime-convection, and others.

---

## Key Features
- **Regional Microclimates:** Investigates EPW weather data files for Dhahran, Jeddah, Tabuk/Wajh, and Al Baha/Aqiq in peak solar and thermal load contexts.
- **Global Material Library (60+ Literature Presets):** Implements verified solar absorptivity ($\alpha$) and thermal emissivity ($\epsilon$) values documented in high-impact literature such as *Nature*, *Science*, *ACS Materials*, and others.
- **Multi-Parameter Search & Filter:** Filters materials by university/institution, chemical formula, absorptivity range, and emissivity range.
- **Thermal Equilibrium Physics Engine:** Solves nonlinear equations defining thermal balance at equilibrium via numerical root-finding with `scipy.optimize.fsolve`.
- **Parametric Sweeps:** Develops 24-hour diurnal temperature profiles and performs GHI sensitivity sweeps and convective wind sensitivity plots.

---

## Thermal Heat Balance Equations
The equilibrium surface temperature of the PDRC surface ($T_{\text{eq}}$) is solved by equating thermal emissions, atmospheric and absorbed solar irradiation, and convective heat transfer. The resulting equation is given as:

$$P_{\text{net}}(T) = P_{\text{rad}}(T) - P_{\text{atm}}(T_{\text{amb}}) - P_{\text{solar}} + P_{\text{conv}}(T) = 0$$

1. **Emissive Radiative Power:**
$$P_{\text{rad}}(T) = \epsilon \cdot \sigma \cdot T^4$$

2. **Absorbed Atmospheric Radiative Power:**
$$P_{\text{atm}}(T_{\text{amb}}) = \epsilon \cdot \epsilon_{\text{sky}} \cdot \sigma \cdot T_{\text{amb}}^4$$

3. **Absorbed Solar Radiative Power:**
$$P_{\text{solar}} = \alpha \cdot \text{GHI}$$

4. **Convective Heat Transfer Power:**
$$P_{\text{conv}} = h_c \cdot (T - T_{\text{amb}})$$

Here, $h_c$ (convective heat transfer coefficient) is developed using moist air density, dynamic viscosity, thermal conductivity, and the Nusselt and Prandtl numbers (all dependent on temperature and pressure).

---

## Project Structure
```text
app.py                 # Streamlit Web Application Interface
prc.py                 # Core Desktop Physics Engine & EPW Parser
materials_database.csv # Comprehensive 60+ Literature PDRC Materials Database
requirements.txt       # Python Package Dependencies
epw_files/             # EPW Weather Datasets for Saudi Arabia Regions
README.md              # Project Documentation
