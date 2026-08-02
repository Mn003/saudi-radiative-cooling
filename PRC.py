import os
import csv
import tkinter as tk
from tkinter import messagebox, ttk
import pandas as pd
import numpy as np

# Force TkAgg rendering backend before any matplotlib submodules load
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.optimize import fsolve

# EPW Data Column Standard Mapping (0-indexed)
epwDryBulbTempCol = 6
epwRelHumidityCol = 8   # Relative Humidity %
epwGhiCol = 13
epwWindSpeedCol = 21


class MaterialLibraryDialog(tk.Toplevel):
    """
    Searchable and Multi-Filtered Dialog Window for 60+ PDRC Materials.
    Supports live sorting, searching by Material/Chemical Name, University, and Range Filtering.
    """
    def __init__(self, parent, materials_list, on_select_callback):
        super().__init__(parent)
        self.title("Global Radiative Cooling Material Library (60+ Benchmark Presets)")
        self.geometry("1150x670")
        self.lift()
        self.focus_force()
        
        self.on_select_callback = on_select_callback
        self.raw_materials = materials_list

        self.create_widgets()
        self.populate_table(self.raw_materials)

    def create_widgets(self):
        # --- Top Filter Panel ---
        filter_frame = ttk.LabelFrame(self, text=" Advanced Search & Sorting Controls ", padding=10)
        filter_frame.pack(fill="x", padx=15, pady=8)

        # Row 0: Text Search & University Filter
        tk.Label(filter_frame, text="Search (Name / Formula):", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, padx=5, pady=4, sticky="e")
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", self.filter_data)
        search_entry = ttk.Entry(filter_frame, textvariable=self.search_var, width=28)
        search_entry.grid(row=0, column=1, padx=5, pady=4, sticky="w")

        tk.Label(filter_frame, text="University / Institution:", font=("Segoe UI", 9, "bold")).grid(row=0, column=2, padx=5, pady=4, sticky="e")
        universities = ["All Institutions"] + sorted(list(set([m.get("university", "N/A") for m in self.raw_materials])))
        self.univ_var = tk.StringVar(value="All Institutions")
        univ_combo = ttk.Combobox(filter_frame, textvariable=self.univ_var, values=universities, state="readonly", width=26)
        univ_combo.grid(row=0, column=3, padx=5, pady=4, sticky="w")
        univ_combo.bind("<<ComboboxSelected>>", self.filter_data)

        # Row 1: Absorptivity Range & Emissivity Range
        tk.Label(filter_frame, text="Absorptivity (α) Range:", font=("Segoe UI", 9, "bold")).grid(row=1, column=0, padx=5, pady=4, sticky="e")
        alpha_range_frame = ttk.Frame(filter_frame)
        alpha_range_frame.grid(row=1, column=1, padx=5, pady=4, sticky="w")
        
        self.alpha_min_var = tk.StringVar(value="0.0")
        self.alpha_max_var = tk.StringVar(value="1.0")
        self.alpha_min_var.trace_add("write", self.filter_data)
        self.alpha_max_var.trace_add("write", self.filter_data)
        
        ttk.Entry(alpha_range_frame, textvariable=self.alpha_min_var, width=6).pack(side="left")
        tk.Label(alpha_range_frame, text=" to ").pack(side="left")
        ttk.Entry(alpha_range_frame, textvariable=self.alpha_max_var, width=6).pack(side="left")

        tk.Label(filter_frame, text="Emissivity (ε) Range:", font=("Segoe UI", 9, "bold")).grid(row=1, column=2, padx=5, pady=4, sticky="e")
        eps_range_frame = ttk.Frame(filter_frame)
        eps_range_frame.grid(row=1, column=3, padx=5, pady=4, sticky="w")
        
        self.eps_min_var = tk.StringVar(value="0.0")
        self.eps_max_var = tk.StringVar(value="1.0")
        self.eps_min_var.trace_add("write", self.filter_data)
        self.eps_max_var.trace_add("write", self.filter_data)
        
        ttk.Entry(eps_range_frame, textvariable=self.eps_min_var, width=6).pack(side="left")
        tk.Label(eps_range_frame, text=" to ").pack(side="left")
        ttk.Entry(eps_range_frame, textvariable=self.eps_max_var, width=6).pack(side="left")

        # Reset Button
        ttk.Button(filter_frame, text="Reset All Filters", command=self.reset_filters).grid(row=0, column=4, rowspan=2, padx=15, pady=4)

        # --- Center Table Frame (Treeview) ---
        table_frame = ttk.Frame(self, padding=5)
        table_frame.pack(fill="both", expand=True, padx=15, pady=5)

        columns = ("Name", "Chemical", "Spec", "University", "Alpha", "Epsilon", "Reference")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")
        
        # Configure Headers
        self.tree.heading("Name", text="Material Label", command=lambda: self.sort_column("Name", False))
        self.tree.heading("Chemical", text="Chemical Formula", command=lambda: self.sort_column("Chemical", False))
        self.tree.heading("Spec", text="Thickness / Spec")
        self.tree.heading("University", text="University / Institution", command=lambda: self.sort_column("University", False))
        self.tree.heading("Alpha", text="Solar Abs. (α)", command=lambda: self.sort_column("Alpha", False))
        self.tree.heading("Epsilon", text="Thermal Emis. (ε)", command=lambda: self.sort_column("Epsilon", False))
        self.tree.heading("Reference", text="Scientific Citation")

        self.tree.column("Name", width=210)
        self.tree.column("Chemical", width=180)
        self.tree.column("Spec", width=110)
        self.tree.column("University", width=160)
        self.tree.column("Alpha", width=90, anchor="center")
        self.tree.column("Epsilon", width=90, anchor="center")
        self.tree.column("Reference", width=200)

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # --- Bottom Action Bar ---
        action_frame = ttk.LabelFrame(self, text=" Load Material into Active Simulation Slot ", padding=10)
        action_frame.pack(fill="x", padx=15, pady=8)

        tk.Label(action_frame, text="Select Target Slot:", font=("Segoe UI", 9, "bold")).pack(side="left", padx=5)
        self.slot_var = tk.IntVar(value=0)
        ttk.Radiobutton(action_frame, text="Slot 1", variable=self.slot_var, value=0).pack(side="left", padx=10)
        ttk.Radiobutton(action_frame, text="Slot 2", variable=self.slot_var, value=1).pack(side="left", padx=10)
        ttk.Radiobutton(action_frame, text="Slot 3", variable=self.slot_var, value=2).pack(side="left", padx=10)

        ttk.Button(action_frame, text="Apply Selected Material", command=self.apply_selection).pack(side="right", padx=5)

    def populate_table(self, data_list):
        """Clears and populates the Treeview table with all matching materials."""
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        for mat in data_list:
            self.tree.insert("", "end", values=(
                mat["name"], 
                mat.get("chemical", "N/A"),
                mat.get("spec", "N/A"),
                mat.get("university", "N/A"), 
                f"{float(mat['alpha']):.3f}", 
                f"{float(mat['epsilon']):.3f}", 
                mat.get("reference", "N/A")
            ))

    def filter_data(self, *args):
        """Applies text, institution, alpha range, and epsilon range filters simultaneously."""
        query = self.search_var.get().lower().strip()
        selected_univ = self.univ_var.get()

        try:
            alpha_min = float(self.alpha_min_var.get())
        except ValueError:
            alpha_min = 0.0

        try:
            alpha_max = float(self.alpha_max_var.get())
        except ValueError:
            alpha_max = 1.0

        try:
            eps_min = float(self.eps_min_var.get())
        except ValueError:
            eps_min = 0.0

        try:
            eps_max = float(self.eps_max_var.get())
        except ValueError:
            eps_max = 1.0

        filtered = []
        for mat in self.raw_materials:
            # Check University
            univ_match = (selected_univ == "All Institutions") or (mat.get("university", "N/A") == selected_univ)
            
            # Check Text Search (Material Name or Chemical Formula)
            text_searchable = f"{mat['name']} {mat.get('chemical', '')} {mat.get('reference', '')}".lower()
            text_match = (not query) or (query in text_searchable)

            # Check Numeric Ranges
            alp = float(mat['alpha'])
            eps = float(mat['epsilon'])
            alpha_match = (alpha_min <= alp <= alpha_max)
            eps_match = (eps_min <= eps <= eps_max)

            if univ_match and text_match and alpha_match and eps_match:
                filtered.append(mat)

        self.populate_table(filtered)

#a, this is just to merge branches dont mind it
    
    def reset_filters(self):
        """Resets all filter controls to show all 60 materials."""
        self.search_var.set("")
        self.univ_var.set("All Institutions")
        self.alpha_min_var.set("0.0")
        self.alpha_max_var.set("1.0")
        self.eps_min_var.set("0.0")
        self.eps_max_var.set("1.0")
        self.populate_table(self.raw_materials)

    def sort_column(self, col, reverse):
        """Sorts Treeview columns when header is clicked."""
        items = [(self.tree.set(k, col), k) for k in self.tree.get_children('')]
        
        try:
            items.sort(key=lambda x: float(x[0]), reverse=reverse)
        except ValueError:
            items.sort(reverse=reverse)

        for index, (val, k) in enumerate(items):
            self.tree.move(k, '', index)

        self.tree.heading(col, command=lambda: self.sort_column(col, not reverse))

    def apply_selection(self):
        """Passes selected material data back to the main GUI."""
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Selection Required", "Please select a material from the table first.")
            return

        values = self.tree.item(selected_item, "values")
        material_data = {
            "name": f"{values[0]} [{values[1]}]",
            "alpha": float(values[4]),
            "epsilon": float(values[5])
        }
        slot_index = self.slot_var.get()
        
        self.on_select_callback(slot_index, material_data)
        self.destroy()


class RadiativeCoolingGui:
    def __init__(self, rootWindow):
        self.rootWindow = rootWindow
        self.rootWindow.title("Saudi Arabia Radiative Cooling Simulator")
        self.rootWindow.geometry("1150x860")
        self.rootWindow.configure(bg="#f4f6f9")
        
        self.baseDir = os.path.dirname(os.path.abspath(__file__))
        
        self.cityProfiles = {
            "Dhahran (Eastern Province)": {
                "file": os.path.join(self.baseDir, "epw_files", "SAU_SH_Dhahran-Abdulaziz.AB.404160_TMYx.2011-2025.epw"),
                "specialty": "Arabian Gulf Coastal-Desert: Subject to extreme summer humidity spikes (often >90% RH) and intense solar loads."
            },
            "Jeddah (Red Sea Coast)": {
                "file": os.path.join(self.baseDir, "epw_files", "SAU_MK_Jeddah-Abdulaziz.Intl.AP.410240_TMYx.2011-2025.epw"),
                "specialty": "Red Sea Coastal Climate: High year-round mean relative humidity (~55–65%) with narrow diurnal temperature ranges."
            },
            "Tabuk / Wajh (Western Coast & Northern Region)": {
                "file": os.path.join(self.baseDir, "epw_files", "SAU_TB_Wajh.AP.404000_TMYx.2007-2021.epw"),
                "specialty": "Red Sea North-Western Coastline: Subject to persistent coastal winds and maritime air masses."
            },
            "Al Baha / Aqiq (High Elevation)": {
                "file": os.path.join(self.baseDir, "epw_files", "SAU_BA_Aqiq-Abdulaziz.AP.410550_TMYx.2011-2025.epw"),
                "specialty": "Sarawat Mountain Elevation (~1,700m+): Lower air density, reduced water vapor column, ideal for high sub-ambient cooling."
            }
        }
        
        self.weatherDataFrame = None
        self.climateScenarios = {}
        self.currentCity = "Dhahran (Eastern Province)"
        self.currentScenario = "Absolute Hottest Day"
        
        self.solarIrradianceGhi = self.ambientTempCelsius = self.windSpeedMs = self.relativeHumidityPct = self.ambientTempKelvin = 0.0
        self.stringVars = {varKey: tk.StringVar() for varKey in ["ghi", "tair", "wind", "rh", "specialty"]}
        
        self.materialsDatabase = self.loadMaterialDatabase()

        self.entryNamesCollection = []
        self.entryEmissivityCollection = []
        self.entryAbsorptivityCollection = []
        
        self.configureStyles()
        self.createWidgets()
        self.loadCityWeatherAndScenarios()

    def loadMaterialDatabase(self):
        """Loads all 60 materials from CSV or fallback internal dictionary."""
        csv_path = os.path.join(self.baseDir, "materials_database.csv")
        
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
            except Exception as e:
                print(f"Warning: Failed reading CSV ({e}). Using embedded database.")

        # Embedded Fallback
        return [
            {"name": "Purdue BaSO4 Paint", "chemical": "BaSO4", "spec": "400 µm", "university": "Purdue University", "alpha": 0.019, "epsilon": 0.950, "reference": "Li et al. (ACS Appl. Mater. 2021)"},
            {"name": "Purdue CaCO3 Paint", "chemical": "CaCO3", "spec": "400 µm", "university": "Purdue University", "alpha": 0.045, "epsilon": 0.955, "reference": "Li et al. (Cell Rep. Phys. Sci. 2020)"},
            {"name": "Stanford Photonic Emitter", "chemical": "HfO2 / SiO2 / Ag", "spec": "1.8 µm", "university": "Stanford University", "alpha": 0.030, "epsilon": 0.960, "reference": "Raman et al. (Nature 2014)"},
            {"name": "Columbia Porous Film", "chemical": "P(VdF-HFP)", "spec": "300 µm", "university": "Columbia University", "alpha": 0.040, "epsilon": 0.960, "reference": "Mandal et al. (Science 2018)"},
            {"name": "Maryland Cooling Wood", "chemical": "Cellulose / Wood", "spec": "Engineered", "university": "University of Maryland", "alpha": 0.040, "epsilon": 0.920, "reference": "Li et al. (Science 2019)"},
            {"name": "Standard TiO2 White Paint", "chemical": "TiO2", "spec": "150 µm", "university": "Commercial Benchmark", "alpha": 0.200, "epsilon": 0.880, "reference": "Commercial Control"}
        ]

    def configureStyles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(".", background="#f4f6f9", font=("Segoe UI", 9))
        style.configure("TLabelFrame", background="#ffffff", bordercolor="#dbe0e6", borderwidth=1, relief="solid")
        style.configure("TLabelFrame.Label", background="#ffffff", foreground="#1e293b", font=("Segoe UI", 10, "bold"))
        style.configure("TButton", background="#0f172a", foreground="#ffffff", font=("Segoe UI", 9, "bold"), padding=6, borderwidth=0)
        style.map("TButton", background=[("active", "#334155")])
        style.configure("TCombobox", fieldbackground="#ffffff", background="#e2e8f0")

    def calculateSkyEmissivity(self, tempCelsius, relativeHumidity):
        a, b = 17.27, 237.7
        alpha = ((a * tempCelsius) / (b + tempCelsius)) + np.log(max(relativeHumidity, 1e-3) / 100.0)
        tempDew = (b * alpha) / (a - alpha)
        
        skyEmissivity = 0.741 + (0.0062 * tempDew)
        return np.clip(skyEmissivity, 0.70, 0.95)

    def calculateConvectiveCoefficient(self, windSpeed, tempCelsius, relativeHumidity):
        naturalConvection = 2.5
        if windSpeed <= 0.05: 
            return naturalConvection
            
        tempKelvin = tempCelsius + 273.15
        satPressure = 610.78 * np.exp((17.27 * tempCelsius) / (tempCelsius + 237.3))
        humidityRatio = 0.622 * ((relativeHumidity / 100.0) * satPressure) / (101325.0 - ((relativeHumidity / 100.0) * satPressure))
        airDensity = 101325.0 / (287.05 * tempKelvin * (1.0 + 0.608 * humidityRatio))

        moistViscosity = (1.458e-6 * (tempKelvin**1.5) / (tempKelvin + 110.4)) * (1.0 + 0.23 * humidityRatio)
        moistConductivity = (2.495e-3 * (tempKelvin**1.5) / (tempKelvin + 194.4)) * (1.0 + 0.45 * humidityRatio)
        prandtlNum = ((1005 + 1820 * humidityRatio) * moistViscosity) / moistConductivity
        reynoldsNum = (airDensity * windSpeed * 1.0) / moistViscosity

        if reynoldsNum < 5e5:
            forcedNusselt = 0.664 * (reynoldsNum**0.5) * (prandtlNum**(1.0/3.0))
        else:
            forcedNusselt = 0.037 * (reynoldsNum**0.8) * (prandtlNum**(1.0/3.0))

        return naturalConvection + ((forcedNusselt * moistConductivity) / 1.0)

    def solveEquilibriumTemperature(self, tempAirK, ghiValue, windSpeed, relHumidity, emissivity, absorptivity):
        tempAirC = tempAirK - 273.15
        convectiveCoef = self.calculateConvectiveCoefficient(windSpeed, tempAirC, relHumidity)
        skyEmissivity = self.calculateSkyEmissivity(tempAirC, relHumidity)
        stefanBoltzmannConst = 5.67e-8

        heatBalance = lambda tempK: (emissivity * stefanBoltzmannConst * (tempK**4)) - (emissivity * skyEmissivity * stefanBoltzmannConst * (tempAirK**4)) - (absorptivity * ghiValue) + (convectiveCoef * (tempK - tempAirK))
        return fsolve(heatBalance, x0=tempAirK)[0] - 273.15

    def safeCloseWindow(self, targetWindow, figureObj):
        try:
            plt.close(figureObj)
        except Exception:
            pass
        targetWindow.destroy()

    def loadCityWeatherAndScenarios(self, eventObj=None):
        self.currentCity = self.cityComboBox.get()
        cityInfo = self.cityProfiles[self.currentCity]
        self.stringVars["specialty"].set(cityInfo["specialty"])
        
        try:
            if not os.path.exists(cityInfo["file"]):
                raise FileNotFoundError(f"EPW file not found at path: {cityInfo['file']}")

            self.weatherDataFrame = pd.read_csv(cityInfo["file"], skiprows=8, header=None)
            targetCols = [epwDryBulbTempCol, epwRelHumidityCol, epwGhiCol, epwWindSpeedCol]
            
            for colIdx in targetCols:
                self.weatherDataFrame[colIdx] = pd.to_numeric(self.weatherDataFrame[colIdx], errors='coerce')
            self.weatherDataFrame = self.weatherDataFrame.dropna(subset=targetCols)
            
            self.climateScenarios = {
                "Absolute Hottest Day": self.weatherDataFrame[epwDryBulbTempCol].idxmax(),
                "Maximum Solar Peak Day": self.weatherDataFrame[epwGhiCol].idxmax()
            }
            
            self.scenarioComboBox['values'] = list(self.climateScenarios.keys())
            self.scenarioComboBox.set("Absolute Hottest Day")
            self.updateActiveScenarioData()
            
        except Exception as err:
            messagebox.showerror("Data Load Error", f"Could not process EPW file:\n{err}")

    def updateActiveScenarioData(self, eventObj=None):
        if self.weatherDataFrame is None: 
            return
        
        self.currentScenario = self.scenarioComboBox.get()
        scenarioRow = self.weatherDataFrame.loc[self.climateScenarios[self.currentScenario]]
        
        self.ambientTempCelsius = float(scenarioRow[epwDryBulbTempCol])
        self.relativeHumidityPct = float(scenarioRow[epwRelHumidityCol])
        self.solarIrradianceGhi = float(scenarioRow[epwGhiCol])
        self.windSpeedMs = float(scenarioRow[epwWindSpeedCol])
        self.ambientTempKelvin = self.ambientTempCelsius + 273.15
        
        self.stringVars["ghi"].set(f"Solar Irradiance (GHI): {self.solarIrradianceGhi:.1f} W/m²")
        self.stringVars["tair"].set(f"Ambient Air Temp: {self.ambientTempCelsius:.1f}°C")
        self.stringVars["wind"].set(f"Wind Speed: {self.windSpeedMs:.1f} m/s")
        self.stringVars["rh"].set(f"Relative Humidity: {self.relativeHumidityPct:.1f}%")

    def openMaterialBrowser(self):
        MaterialLibraryDialog(self.rootWindow, self.materialsDatabase, self.assignMaterialFromBrowser)

    def assignMaterialFromBrowser(self, slotIndex, matData):
        self.entryNamesCollection[slotIndex].delete(0, tk.END)
        self.entryNamesCollection[slotIndex].insert(0, matData["name"])

        self.entryEmissivityCollection[slotIndex].delete(0, tk.END)
        self.entryEmissivityCollection[slotIndex].insert(0, str(matData["epsilon"]))

        self.entryAbsorptivityCollection[slotIndex].delete(0, tk.END)
        self.entryAbsorptivityCollection[slotIndex].insert(0, str(matData["alpha"]))

    def createWidgets(self):
        regionFrame = ttk.LabelFrame(self.rootWindow, text=" Dynamic Saudi Region Setup ", padding=(12, 8))
        regionFrame.pack(fill="x", padx=20, pady=6)
        
        tk.Label(regionFrame, text="Select Meteorological Region:", font=("Segoe UI", 10), bg="#ffffff", fg="#334155").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.cityComboBox = ttk.Combobox(regionFrame, values=list(self.cityProfiles.keys()), state="readonly", width=34)
        self.cityComboBox.set(self.currentCity)
        self.cityComboBox.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.cityComboBox.bind("<<ComboboxSelected>>", self.loadCityWeatherAndScenarios)

        tk.Label(regionFrame, textvariable=self.stringVars["specialty"], font=("Segoe UI", 9, "italic"), bg="#ffffff", fg="#64748b", wraplength=950, justify="left").grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="w")

        selectorFrame = ttk.LabelFrame(self.rootWindow, text=" Target Environmental Evaluation Scenario ", padding=(12, 8))
        selectorFrame.pack(fill="x", padx=20, pady=6)
        tk.Label(selectorFrame, text="Select Profile Case Study:", font=("Segoe UI", 10), bg="#ffffff", fg="#334155").pack(side="left", padx=5)
        self.scenarioComboBox = ttk.Combobox(selectorFrame, values=list(self.climateScenarios.keys()), state="readonly", width=34)
        self.scenarioComboBox.pack(side="left", padx=5, pady=5)
        self.scenarioComboBox.bind("<<ComboboxSelected>>", self.updateActiveScenarioData)

        weatherFrame = ttk.LabelFrame(self.rootWindow, text=" Peak Regional Climate Benchmarks ", padding=(12, 8))
        weatherFrame.pack(fill="x", padx=20, pady=6)
        
        varKeys = [["ghi", "tair"], ["wind", "rh"]]
        for rowIdx in range(2):
            for colIdx in range(2):
                tk.Label(weatherFrame, textvariable=self.stringVars[varKeys[rowIdx][colIdx]], font=("Segoe UI", 10, "bold"), bg="#ffffff", fg="#0f172a").grid(row=rowIdx, column=colIdx, sticky="w", padx=15, pady=2)

        inputFrame = ttk.LabelFrame(self.rootWindow, text=" Core Materials Matrix & Database ", padding=(12, 8))
        inputFrame.pack(fill="x", padx=20, pady=6)
        
        headerList = [("Material / Surface Label Selection", 0, "w"), ("Emissivity (ε)", 1, "center"), ("Absorptivity (α)", 2, "center")]
        for textStr, colIdx, alignmentStr in headerList:
            headerLabel = tk.Label(inputFrame, text=textStr, font=("Segoe UI", 9, "bold"), bg="#ffffff", fg="#334155")
            if alignmentStr == "w":
                headerLabel.grid(row=0, column=colIdx, padx=5, pady=2, sticky="w")
            else:
                headerLabel.grid(row=0, column=colIdx, padx=5, pady=2)

        default_seeds = [
            self.materialsDatabase[7] if len(self.materialsDatabase) > 7 else {"name": "Standard TiO2 Paint", "chemical": "TiO2", "epsilon": 0.88, "alpha": 0.20},
            self.materialsDatabase[0] if len(self.materialsDatabase) > 0 else {"name": "Purdue BaSO4 Paint", "chemical": "BaSO4", "epsilon": 0.95, "alpha": 0.019},
            self.materialsDatabase[21] if len(self.materialsDatabase) > 21 else {"name": "Stanford Photonic Emitter", "chemical": "HfO2/SiO2/Ag", "epsilon": 0.96, "alpha": 0.03}
        ]

        for itemIdx in range(3):
            nameEntry = ttk.Entry(inputFrame, width=42)
            emissivityEntry = ttk.Entry(inputFrame, width=12)
            absorptivityEntry = ttk.Entry(inputFrame, width=12)
            
            seed = default_seeds[itemIdx]
            label_text = f"{seed['name']} [{seed.get('chemical', 'N/A')}]"
            nameEntry.insert(0, label_text)
            emissivityEntry.insert(0, str(seed["epsilon"]))
            absorptivityEntry.insert(0, str(seed["alpha"]))

            nameEntry.grid(row=itemIdx+1, column=0, padx=5, pady=4, sticky="w")
            emissivityEntry.grid(row=itemIdx+1, column=1, padx=5, pady=4)
            absorptivityEntry.grid(row=itemIdx+1, column=2, padx=5, pady=4)

            self.entryNamesCollection.append(nameEntry)
            self.entryEmissivityCollection.append(emissivityEntry)
            self.entryAbsorptivityCollection.append(absorptivityEntry)

        ttk.Button(
            inputFrame, 
            text="🔍 Search & Filter 60+ Literature Material Database (University, Chemical, Alpha, Epsilon)", 
            command=self.openMaterialBrowser
        ).grid(row=4, column=0, columnspan=3, pady=8)

        resultsFrame = ttk.LabelFrame(self.rootWindow, text=" Computed Comparative Equilibrium Temperatures ", padding=(12, 8))
        resultsFrame.pack(fill="x", padx=20, pady=6)
        self.outputText = tk.Text(resultsFrame, height=7, font=("Consolas", 10), bg="#0f172a", fg="#38bdf8", relief="flat", bd=0, padx=10, pady=10)
        self.outputText.pack(fill="x", expand=True)
        self.outputText.insert("1.0", "Select a city, browse/set up your materials, and click 'Run Equilibrium Calculation'.")
        self.outputText.config(state="disabled")

        buttonFrame = tk.Frame(self.rootWindow, bg="#f4f6f9")
        buttonFrame.pack(fill="x", padx=20, pady=10)
        actionList = [("Run Equilibrium Calculation", self.calculatePeakMatrix), 
                      ("Plot Diurnal Performance Profile", self.plotComparativeProfile),
                      ("Plot Sensitivity Sweeps", self.plotSensitivityAnalysis), 
                      ("Plot Fixed-Context Wind Sweep", self.plotFixedWindSensitivity)]
        for buttonText, buttonCmd in actionList:
            ttk.Button(buttonFrame, text=buttonText, command=buttonCmd).pack(side="left", padx=4)

    def getMaterialMatrix(self):
        try:
            return [{"name": self.entryNamesCollection[itemIdx].get().strip(), 
                     "eps": float(self.entryEmissivityCollection[itemIdx].get()), 
                     "alp": float(self.entryAbsorptivityCollection[itemIdx].get())} for itemIdx in range(len(self.entryNamesCollection))]
        except ValueError as err:
            messagebox.showerror("Input Validation Error", f"Failed parsing inputs:\n{err}")
            return None

    def calculatePeakMatrix(self):
        materialMatrix = self.getMaterialMatrix()
        if not materialMatrix: 
            return
            
        self.outputText.config(state="normal")
        self.outputText.delete("1.0", tk.END)
        
        outputLines = [f"REGIONAL TRACE: {self.currentCity.upper()} - {self.currentScenario.upper()}", f"{'CONFIGURATION MATERIAL':<34} | {'EQUILIBRIUM TEMP':<18} | {'THERMAL DELTA VS AIR':<22}", "-" * 82]
        for materialItem in materialMatrix:
            eqCelsius = self.solveEquilibriumTemperature(self.ambientTempKelvin, self.solarIrradianceGhi, self.windSpeedMs, self.relativeHumidityPct, materialItem["eps"], materialItem["alp"])
            tempDelta = eqCelsius - self.ambientTempCelsius
            statusText = f"{tempDelta:+.2f}°C (Sub-ambient Cooling)" if tempDelta < 0 else f"{tempDelta:+.2f}°C (Heating Penalty)"
            outputLines.append(f"{materialItem['name']:<34} | {eqCelsius:>6.2f} °C        | {statusText}")
            
        self.outputText.insert("1.0", "\n".join(outputLines))
        self.outputText.config(state="disabled")

    def plotComparativeProfile(self):
        materialMatrix = self.getMaterialMatrix()
        if not materialMatrix: 
            return
        
        targetWindow = None
        try:
            totalRows = len(self.weatherDataFrame)
            startIndex = (self.climateScenarios[self.currentScenario] // 24) * 24
            if startIndex + 24 > totalRows:
                startIndex = max(0, totalRows - 24)
                
            dayDataFrame = self.weatherDataFrame.iloc[startIndex : startIndex + 24]
            actualRowCount = len(dayDataFrame)
            hourArray = np.arange(1, actualRowCount + 1)
            
            if actualRowCount == 0:
                raise ValueError("No weather data found for the selected slice range.")
            
            ambientTemps = dayDataFrame[epwDryBulbTempCol].values
            simulatedProfiles = {
                matIdx: [self.solveEquilibriumTemperature(
                        rowVal[epwDryBulbTempCol] + 273.15, 
                        rowVal[epwGhiCol], 
                        rowVal[epwWindSpeedCol], 
                        rowVal[epwRelHumidityCol], 
                        matData["eps"], 
                        matData["alp"]
                    ) for _, rowVal in dayDataFrame.iterrows()] 
                for matIdx, matData in enumerate(materialMatrix)
            }

            targetWindow = tk.Toplevel(self.rootWindow)
            targetWindow.title(f"Diurnal Performance: {self.currentCity}")
            targetWindow.geometry("1150x670")
            targetWindow.lift()
            targetWindow.focus_force()

            figureObj = Figure(figsize=(11.0, 5.2), dpi=100)
            plotAx1, plotAx2 = figureObj.subplots(1, 2)
            
            plotAx1.plot(hourArray, ambientTemps, 'k--', lw=2.5, label='Ambient Air Temp Baseline')
            colorPalette, markerStyles = ['#e63946', '#2a9d8f', '#457b9d'], ['-o', '-s', '-^']
            for matIdx, matData in enumerate(materialMatrix):
                plotAx1.plot(hourArray, simulatedProfiles[matIdx], markerStyles[matIdx%3], color=colorPalette[matIdx%3], lw=1.8, label=f"{matData['name']} (ε={matData['eps']}, α={matData['alp']})")

            plotAx1.set_xlabel('Hour of Day', fontweight='bold')
            plotAx1.set_ylabel('Steady State Temp (°C)', fontweight='bold')
            plotAx1.set_xticks(hourArray)
            plotAx1.grid(True, linestyle=':', alpha=0.6)
            plotAx1.legend(loc='upper left', fontsize='x-small') 
            plotAx1.set_title("Equilibrium Thermal Response", fontsize=10, fontweight='bold')

            plotAx2.plot(hourArray, dayDataFrame[epwGhiCol].values, color='#f39c12', lw=2, marker='o', label='Solar GHI (W/m²)')
            plotAx2.set_xlabel('Hour of Day', fontweight='bold')
            plotAx2.set_ylabel('Solar Irradiance (GHI) [W/m²]', color='#f39c12', fontweight='bold')
            plotAx2.tick_params(axis='y', labelcolor='#f39c12')
            plotAx2.set_xticks(hourArray)
            plotAx2.grid(True, linestyle=':', alpha=0.4)

            twinAx2 = plotAx2.twinx()
            twinAx2.plot(hourArray, ambientTemps, color='#e74c3c', lw=2, linestyle='-.', label='Ambient Temp (°C)')
            twinAx2.plot(hourArray, dayDataFrame[epwWindSpeedCol].values, color='#3498db', lw=2, linestyle=':', label='Wind Speed (m/s)')
            twinAx2.plot(hourArray, dayDataFrame[epwRelHumidityCol].values, color='#2ecc71', lw=1.8, label='Relative Humidity (%)')
            twinAx2.set_ylabel('Temp / Wind / Humidity', fontweight='bold')
            
            handles1, labels1 = plotAx2.get_legend_handles_labels()
            handles2, labels2 = twinAx2.get_legend_handles_labels()
            twinAx2.legend(handles1+handles2, labels1+labels2, loc='upper right', fontsize='x-small') 
            plotAx2.set_title("Active Meteorological Variables", fontsize=10, fontweight='bold')

            figureObj.suptitle(f"Diurnal Performance Profile - {self.currentCity}", fontsize=12, fontweight='bold')
            figureObj.tight_layout()

            canvasObj = FigureCanvasTkAgg(figureObj, master=targetWindow)
            canvasObj.draw()
            
            toolbar = NavigationToolbar2Tk(canvasObj, targetWindow)
            toolbar.update()
            canvasObj.get_tk_widget().pack(fill="both", expand=True)
            
            targetWindow.protocol("WM_DELETE_WINDOW", lambda: self.safeCloseWindow(targetWindow, figureObj))

        except Exception as err:
            if targetWindow is not None:
                try:
                    targetWindow.destroy()
                except:
                    pass
            messagebox.showerror("Plotting Error", f"An error occurred while generating the diurnal plot:\n{err}")

    def plotSensitivityAnalysis(self):
        materialMatrix = self.getMaterialMatrix()
        if not materialMatrix: 
            return

        ghiSweep = np.linspace(0, 1000, 50)
        windSweep = np.linspace(0.1, 12, 50)
        
        targetWindow = tk.Toplevel(self.rootWindow)
        targetWindow.title("Sensitivity Analysis Dashboard")
        targetWindow.geometry("1100x600")
        targetWindow.lift()
        targetWindow.focus_force()

        figureObj = Figure(figsize=(11.5, 5), dpi=100)
        subPlot1, subPlot2 = figureObj.subplots(1, 2)
        colorPalette, markerStyles = ['#e63946', '#2a9d8f', '#457b9d'], ['o', 's', '^']

        for matIdx, matData in enumerate(materialMatrix):
            subPlot1.plot(ghiSweep, [self.solveEquilibriumTemperature(self.ambientTempKelvin, ghiVal, self.windSpeedMs, self.relativeHumidityPct, matData["eps"], matData["alp"]) for ghiVal in ghiSweep], color=colorPalette[matIdx%3], marker=markerStyles[matIdx%3], markevery=5, lw=2, label=matData["name"])
            subPlot2.plot(windSweep, [self.solveEquilibriumTemperature(self.ambientTempKelvin, self.solarIrradianceGhi, windVal, self.relativeHumidityPct, matData["eps"], matData["alp"]) for windVal in windSweep], color=colorPalette[matIdx%3], marker=markerStyles[matIdx%3], markevery=5, lw=2, label=matData["name"])
            
        for subAx, xLabelStr, plotTitleStr in [(subPlot1, "Solar Radiation Load (GHI) [W/m²]", "Sensitivity vs. Solar Irradiance Load"), (subPlot2, "Convective Wind Speed [m/s]", "Sensitivity vs. Wind Convection")]:
            subAx.axhline(self.ambientTempCelsius, color='black', linestyle='--', alpha=0.7, label="Ambient Baseline")
            subAx.set_xlabel(xLabelStr, fontweight='bold')
            subAx.set_ylabel("Equilibrium Temperature (°C)", fontweight='bold')
            subAx.set_title(plotTitleStr, fontsize=10, fontweight='bold')
            subAx.grid(True, linestyle=':', alpha=0.6)
            subAx.legend(loc='lower left', fontsize='small')

        figureObj.suptitle(f"Parametric Sensitivity Dashboard ({self.ambientTempCelsius:.1f}°C Ambient)", fontsize=11, fontweight='bold')
        figureObj.tight_layout()
        
        canvasObj = FigureCanvasTkAgg(figureObj, master=targetWindow)
        canvasObj.draw()
        
        toolbar = NavigationToolbar2Tk(canvasObj, targetWindow)
        toolbar.update()
        canvasObj.get_tk_widget().pack(fill="both", expand=True)
        
        targetWindow.protocol("WM_DELETE_WINDOW", lambda: self.safeCloseWindow(targetWindow, figureObj))

    def plotFixedWindSensitivity(self):
        materialMatrix = self.getMaterialMatrix()
        if not materialMatrix: 
            return

        targetWindow = tk.Toplevel(self.rootWindow)
        targetWindow.title("Wind Speed Sensitivity")
        targetWindow.geometry("850x700")
        targetWindow.lift()
        targetWindow.focus_force()

        infoFrame = ttk.LabelFrame(targetWindow, text=" Fixed Environmental Boundary Conditions ", padding=(15, 10))
        infoFrame.pack(fill="x", padx=20, pady=10)
        tk.Label(infoFrame, text=f"• Solar Irradiance (GHI): {self.solarIrradianceGhi:.1f} W/m²\n• Ambient Air Temperature: {self.ambientTempCelsius:.1f}°C\n• Boundary Relative Humidity: {self.relativeHumidityPct:.1f}%", font=("Consolas", 10, "bold"), justify="left", fg="#0284c7", bg="#ffffff").pack(anchor="w")

        figureObj = Figure(figsize=(7.5, 4.5), dpi=100)
        subAx = figureObj.add_subplot(111)
        colorPalette, markerStyles = ['#e63946', '#2a9d8f', '#457b9d'], ['o', 's', '^']

        for matIdx, matData in enumerate(materialMatrix):
            subAx.plot(np.linspace(0.1, 12, 50), [self.solveEquilibriumTemperature(self.ambientTempKelvin, self.solarIrradianceGhi, windVal, self.relativeHumidityPct, matData["eps"], matData["alp"]) for windVal in np.linspace(0.1, 12, 50)], color=colorPalette[matIdx%3], marker=markerStyles[matIdx%3], markevery=5, lw=2, label=matData["name"])
            
        subAx.axhline(self.ambientTempCelsius, color='black', linestyle='--', alpha=0.7, label=f"Ambient Baseline ({self.ambientTempCelsius:.1f}°C)")
        subAx.set_xlabel("Convective Wind Speed [m/s]", fontweight='bold')
        subAx.set_ylabel("Equilibrium Temperature (°C)", fontweight='bold')
        subAx.grid(True, linestyle=':', alpha=0.6)
        subAx.legend(loc='lower left', fontsize='small')

        figureObj.tight_layout()
        
        canvasObj = FigureCanvasTkAgg(figureObj, master=targetWindow)
        canvasObj.draw()
        
        toolbar = NavigationToolbar2Tk(canvasObj, targetWindow)
        toolbar.update()
        canvasObj.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=5)
        
        targetWindow.protocol("WM_DELETE_WINDOW", lambda: self.safeCloseWindow(targetWindow, figureObj))


if __name__ == "__main__":
    rootWindow = tk.Tk()
    appObj = RadiativeCoolingGui(rootWindow)
    rootWindow.mainloop()
