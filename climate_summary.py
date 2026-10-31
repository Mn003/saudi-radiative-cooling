"""
climate_summary.py
Summarises the four TMYx weather files used by the project (run from the repo root):
    python climate_summary.py
Prints, per station: elevation, maximum dry-bulb temperature, hours >= 40 C,
annual and summer (Jun-Aug) mean relative humidity, and summer mean dew point.
EPW fields (0-indexed): 1 month, 3 hour, 6 dry-bulb C, 7 dew point C, 8 RH %.
"""
import glob, os, pandas as pd
here = os.path.dirname(os.path.abspath(__file__))
for f in sorted(glob.glob(os.path.join(here, "epw_files", "*.epw"))):
    loc = open(f, encoding="utf-8").readline().split(",")
    df = pd.read_csv(f, skiprows=8, header=None)
    jja = df[df[1].isin([6, 7, 8])]
    print(f"{loc[1]:26s} elev {float(loc[9]):7.1f} m | Tmax {df[6].max():4.1f} C | hours>=40C {(df[6] >= 40).sum():4d} | "
          f"RH annual {df[8].mean():4.1f}% | RH Jun-Aug {jja[8].mean():4.1f}% | dew point Jun-Aug {jja[7].mean():4.1f} C")
