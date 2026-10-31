"""make_figures.py - figures for the Ibdaa 2027 Qalib, drawn only from files in this repo.
Run after analysis_summer.py:  python make_figures.py --out figures

Figure 1  fig_climate_dewpoint.png      mean June-August dew point per station (from the EPW files)
Figure 2  fig_heat_balance.png          the energy balance the model solves (concept diagram)
Figure 3  fig_method_flowchart.png      method overview
Figure 4  fig_results_dT.png            median and IQR of surface-minus-air temperature (results/)
Figure 5  fig_reflectance_threshold.png solar reflectance needed per station (results/)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import PRC  # noqa: E402

# ---- shared style (serif to match the Times New Roman text of the form) ----------
INK, INK2, MUTED, GRID = "#1b1b1b", "#4a4946", "#8a8984", "#e6e5e1"
BLUE, BLUE_L, ORANGE, ORANGE_L, GREEN, SUN = "#2a78d6", "#cfe2f8", "#e0602b", "#fbdccd", "#178a5f", "#f2a900"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 10.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titleweight": "bold", "axes.titlesize": 11,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.06, "figure.facecolor": "white",
})
ORDER = ["Al Baha", "Dhahran", "Jeddah", "Al Wajh"]
SHORT = {"BaSO4-acrylic paint": "BaSO$_4$ paint", "BaSO4 nanoparticle film": "BaSO$_4$ film",
         "Hierarchically porous P(VdF-HFP) coating": "Porous P(VdF-HFP)", "CaCO3 paint": "CaCO$_3$ paint",
         "Commercial white paint (control)": "White paint (control)"}


def _clean(ax, grid_axis="x"):
    ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# ---- Figure 4 ---------------------------------------------------------------------
def fig_results(res, out):
    s = pd.read_csv(os.path.join(res, "summary_by_material_station.csv"))
    s = s[s.material.isin(SHORT)]
    env_path = os.path.join(os.path.dirname(res.rstrip("/")), "results_uncertainty", "envelope.csv")
    env = pd.read_csv(env_path) if os.path.exists(env_path) else None
    mats = list(SHORT)
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 3.4), sharey=True, sharex=True)
    xmin, xmax = -8.6, 4.4
    for ax, st in zip(axes, ORDER):
        ax.axvspan(xmin, 0, color=BLUE_L, alpha=0.35, lw=0)
        ax.axvspan(0, xmax, color=ORANGE_L, alpha=0.35, lw=0)
        d = s[s.station == st].set_index("material").loc[mats]
        y = np.arange(len(mats))[::-1]
        for yi, (m, r) in zip(y, d.iterrows()):
            c = ORANGE if "control" in m else BLUE
            if env is not None:
                e = env[(env.station == st) & (env.material == m)].iloc[0]
                ax.plot([e.low, e.high], [yi, yi], color=c, lw=1.1, alpha=0.9, zorder=1)
                for xe in (e.low, e.high):
                    ax.plot([xe, xe], [yi - 0.13, yi + 0.13], color=c, lw=1.1, alpha=0.9, zorder=1)
            ax.plot([r.p25_dT, r.p75_dT], [yi, yi], color=c, lw=6, alpha=0.35, solid_capstyle="round")
            ax.plot(r.median_dT, yi, "o", ms=7.5, color=c, mec="white", mew=1.4, zorder=3)
            lab = f"{r.median_dT:+.1f}".replace("-", "−")
            if abs(r.median_dT) < 0.8:
                ax.text(r.median_dT - 0.35, yi + 0.36, lab, ha="right", fontsize=8.5, color=INK2)
            else:
                ax.text(r.median_dT, yi + 0.36, lab, ha="center", fontsize=8.5, color=INK2)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_title(st, loc="left", pad=6)
        ax.set_xlim(xmin, xmax)
        ax.set_xticks([-6, -3, 0, 3])
        ax.set_xticklabels(["−6", "−3", "0", "+3"])
        _clean(ax)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(np.arange(len(mats))[::-1])
    axes[0].set_yticklabels([SHORT[m] for m in mats])
    axes[0].set_ylim(-0.6, len(mats) - 0.25)
    axes[0].text(-8.3, -0.52, "cooler than air", fontsize=8, color=BLUE, style="italic")
    axes[-1].text(4.2, -0.52, "warmer", fontsize=8, color=ORANGE, style="italic", ha="right")
    fig.supxlabel("Predicted surface minus air temperature, $\\Delta T$ (°C)\n• dot = median   • thick bar = middle 50% of hours"
                  "   • thin line = range over all tested uncertainties", fontsize=9, color=INK2, y=0.01)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(os.path.join(out, "fig_results_dT.png"))
    plt.close(fig)


# ---- Figure 5 ---------------------------------------------------------------------
def fig_threshold(res, out):
    t = pd.read_csv(os.path.join(res, "alpha_threshold.csv")).set_index("station").loc[ORDER]
    need = (1 - t.max_alpha_for_median_below_air) * 100
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    x = np.arange(len(ORDER))
    ax.bar(x, need - 80, bottom=80, width=0.56, color=BLUE, edgecolor="white", lw=1.5, zorder=2)
    rng_path = os.path.join(os.path.dirname(res.rstrip("/")), "results_uncertainty", "threshold_range.csv")
    if os.path.exists(rng_path):
        r = pd.read_csv(rng_path).groupby("station").max_alpha.agg(["min", "max"]).loc[ORDER]
        lo, hi = (1 - r["max"]) * 100, (1 - r["min"]) * 100
        for xi, a, b in zip(x, lo, hi):
            ax.plot([xi + 0.36, xi + 0.36], [a, b], color=INK2, lw=1.3, zorder=4)
            for v in (a, b):
                ax.plot([xi + 0.31, xi + 0.41], [v, v], color=INK2, lw=1.3, zorder=4)
    for xi, v in zip(x, need):
        ax.text(xi, v - 0.95, f"{v:.1f}%", ha="center", fontsize=10, color="white", fontweight="bold")
    refs = [(98.1, "BaSO$_4$ paint 98.1%", INK2), (95.5, "CaCO$_3$ paint 95.5%", INK2),
            (87.2, "White paint (control) 87.2%", ORANGE)]
    for v, lab, c in refs:
        ax.plot([-0.55, len(ORDER) - 0.45], [v, v], color=c, lw=1.3, ls=(0, (5, 3)), zorder=3)
        ax.text(len(ORDER) - 0.35, v, lab, ha="left", va="center", fontsize=9, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER)
    ax.set_xlim(-0.55, len(ORDER) + 1.5)
    ax.spines["bottom"].set_bounds(-0.55, len(ORDER) - 0.45)
    ax.set_ylim(80, 100)
    ax.set_ylabel("Solar reflectance needed (%)")
    _clean(ax, "y")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_reflectance_threshold.png"))
    plt.close(fig)


# ---- Figure 1 ---------------------------------------------------------------------
def fig_climate(out):
    elev = {"Al Baha": 1672, "Dhahran": 26, "Jeddah": 15, "Al Wajh": 20}
    rows = []
    for c in PRC.city_profiles:
        df = PRC.load_epw_weather(c)
        jja = df[df[1].isin([6, 7, 8])]
        st = [k for k in elev if c.startswith(k)][0]
        rows.append((st, jja[PRC.epwDewPointCol].mean()))
    d = pd.DataFrame(rows, columns=["st", "dew"]).sort_values("dew")
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    y = np.arange(len(d))
    shades = ["#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab"]  # light = drier, dark = more humid
    ax.barh(y, d.dew, height=0.58, color=shades[: len(d)], edgecolor="white", lw=1.5, zorder=2)
    for yi, v in zip(y, d.dew):
        ax.text(v + 0.4, yi, f"{v:.1f} °C", va="center", fontsize=10, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s}  ({elev[s]:,} m)" for s in d.st])
    ax.set_xlabel("Mean June–August dew point (°C)   ← drier air            more humid air →")
    ax.set_xlim(0, 27)
    _clean(ax)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_climate_dewpoint.png"))
    plt.close(fig)
    return d


# ---- helpers for diagrams ---------------------------------------------------------
def _arrow(ax, a, b, c=INK2, lw=2.0, ms=14, style="-|>", z=4):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=ms, color=c, lw=lw, zorder=z,
                                 shrinkA=0, shrinkB=0))


def _box(ax, x, y, w, h, txt, fc, ec, fs=9.5, bold_first=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2, zorder=2))
    if bold_first and "\n" in txt:
        first, rest = txt.split("\n", 1)
        n = rest.count("\n") + 1
        ax.text(x, y + 0.14 + 0.13 * (n - 1), first, ha="center", va="center", fontsize=fs, fontweight="bold",
                color=INK, zorder=3)
        ax.text(x, y - 0.16 - 0.04 * (n - 1), rest, ha="center", va="center", fontsize=fs - 0.8,
                color=INK2, zorder=3, linespacing=1.25)
    else:
        ax.text(x, y, txt, ha="center", va="center", fontsize=fs, color=INK, zorder=3, linespacing=1.3)


# ---- Figure 2 ---------------------------------------------------------------------
def fig_heat_balance(out):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    # sky gradient
    sky = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(sky, extent=(0, 10, 1.55, 6.4), origin="lower", aspect="auto", zorder=0,
              cmap=matplotlib.colors.LinearSegmentedColormap.from_list("sky", ["#f4f8fd", "#cfe0f5"]))
    ax.text(9.85, 6.2, "Sky / atmosphere", ha="right", va="top", fontsize=9, color=INK2, style="italic")
    # sun
    ax.add_patch(Circle((1.0, 5.6), 0.38, color=SUN, zorder=2))
    for ang in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        ax.plot([1.0 + 0.5 * np.cos(ang), 1.0 + 0.66 * np.cos(ang)],
                [5.6 + 0.5 * np.sin(ang), 5.6 + 0.66 * np.sin(ang)], color=SUN, lw=1.6, zorder=2)
    # roof: coating on top of an insulated roof slab
    ax.add_patch(Rectangle((0.6, 1.3), 8.8, 0.25, fc="#ffffff", ec=BLUE, lw=1.4, zorder=3))
    ax.add_patch(Rectangle((0.6, 0.75), 8.8, 0.55, fc="#efeee9", ec=MUTED, lw=1, hatch="////", zorder=3))
    ax.text(5.0, 1.425, "PDRC coating (α, ε), surface temperature $T_s$", ha="center", va="center",
            fontsize=9.5, zorder=4)
    ax.text(5.0, 0.45, "Roof below treated as insulated: no heat flow through the back (model assumption)",
            ha="center", va="center", fontsize=8.8, color=INK2)
    # 1 sunlight in, part reflected
    _arrow(ax, (1.45, 5.15), (2.35, 1.6), c=SUN, lw=3.0, ms=18)
    _arrow(ax, (2.55, 1.6), (3.25, 3.9), c="#f6cd6a", lw=2.2, ms=14)
    ax.text(0.15, 2.45, "Sunlight\nabsorbed\n$\\alpha \\cdot GHI$", fontsize=9.5, color="#a87400", va="center")
    ax.text(3.35, 3.95, "most sunlight\nreflected", fontsize=8.6, color="#a87400", va="bottom")
    # 2 emitted infrared, up through the window
    _arrow(ax, (4.75, 1.6), (4.75, 5.55), c=BLUE, lw=3.0, ms=18)
    ax.text(4.95, 5.25, "Infrared emitted\n$\\varepsilon\\sigma T_s^4$", fontsize=9.5, color=BLUE, va="center")
    ax.text(4.95, 4.45, "escapes to space mainly\nvia the 8–13 µm window", fontsize=8.4,
            color=BLUE, va="center", style="italic")
    # 3 sky infrared absorbed
    _arrow(ax, (7.45, 5.55), (7.45, 1.6), c=MUTED, lw=3.0, ms=18)
    ax.text(7.65, 5.25, "Sky infrared absorbed\n$\\varepsilon\\,\\varepsilon_{sky}\\sigma T_{air}^4$",
            fontsize=9.5, color=INK2, va="center")
    ax.text(7.65, 4.45, "larger when the air\nis more humid", fontsize=8.4, color=INK2, va="center", style="italic")
    # 4 convection with the air
    _arrow(ax, (7.9, 2.25), (9.6, 2.25), c=GREEN, lw=2.2, ms=14, style="<|-|>")
    ax.text(8.75, 2.75, "Convection with air\n$h_c\\,(T_s - T_{air})$", fontsize=9.5, color=GREEN,
            ha="center", va="center")
    # balance
    ax.add_patch(FancyBboxPatch((1.15, 5.95 - 0.02), 7.7, 0.0, boxstyle="square,pad=0", lw=0))
    fig.text(0.5, 0.005, "Steady state:   $\\varepsilon\\sigma T_s^4 \\;-\\; \\varepsilon\\,\\varepsilon_{sky}\\sigma T_{air}^4 "
             "\\;-\\; \\alpha\\,GHI \\;+\\; h_c\\,(T_s - T_{air}) \\;=\\; 0$",
             ha="center", va="bottom", fontsize=11)
    fig.subplots_adjust(bottom=0.09)
    fig.savefig(os.path.join(out, "fig_heat_balance.png"))
    plt.close(fig)


# ---- Figure 3 ---------------------------------------------------------------------
def fig_flowchart(out):
    fig, ax = plt.subplots(figsize=(7.2, 4.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    stages = [(6.15, "INPUTS"), (4.55, "MODEL"), (2.85, "ANALYSES"), (1.1, "CHECK")]
    for y, lab in stages:
        ax.text(0.05, y, lab, fontsize=8.5, color=MUTED, fontweight="bold", va="center", rotation=90)
    BL, BLF = BLUE, "#eaf2fc"
    _box(ax, 2.95, 6.15, 4.1, 1.0, "Weather\n4 TMYx station files [4]\nJun–Aug, GHI > 600 W/m², clear sky",
         BLF, BL, bold_first=True)
    _box(ax, 7.45, 6.15, 4.1, 1.0, "Materials\n4 PDRC materials + 1 control\nα and ε checked in papers [10, 21, 22]",
         BLF, BL, bold_first=True)
    _box(ax, 5.2, 4.55, 6.6, 1.0, "Energy balance, solved for every hour × material × station\n"
         "sky emissivity [23] • convection [24, 25] • Brent solver", "#dcebfb", BL, bold_first=True)
    _box(ax, 2.05, 2.85, 2.9, 1.0, "Ranking\nObjective 2", BLF, BL, bold_first=True)
    _box(ax, 5.2, 2.85, 2.9, 1.0, "Sensitivity ±0.02\nα threshold • Objective 3", BLF, BL, bold_first=True)
    _box(ax, 8.35, 2.85, 2.9, 1.0, "Uncertainty\nα, ε ±0.02 • L 0.1–10 m\n2 sky models [23, 26]", BLF, BL, bold_first=True)
    _box(ax, 5.2, 1.1, 8.0, 0.95, "Validation (Objective 4): planned\nrooftop test in Jeddah, model settings fixed "
         "in advance (Section 8)", "#f5f4f0", MUTED, bold_first=True)
    _arrow(ax, (2.95, 5.65), (4.0, 5.07), lw=1.6, ms=12)
    _arrow(ax, (7.45, 5.65), (6.4, 5.07), lw=1.6, ms=12)
    for x in (2.05, 5.2, 8.35):
        _arrow(ax, (5.2 + (x - 5.2) * 0.42, 4.03), (x, 3.37), lw=1.6, ms=12)
    _arrow(ax, (5.2, 2.33), (5.2, 1.6), c=MUTED, lw=1.4, ms=11)
    fig.savefig(os.path.join(out, "fig_method_flowchart.png"))
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--out", default=os.path.join(HERE, "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_results(a.results, a.out)
    fig_threshold(a.results, a.out)
    print(fig_climate(a.out))
    fig_flowchart(a.out)
    fig_heat_balance(a.out)
    print("Figures written to", a.out)
