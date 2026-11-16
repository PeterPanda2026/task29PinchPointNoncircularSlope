"""Slope in three soil layers known only at four boreholes: geometry, borehole logs and soil statistics.

Geometry
    Ground surface (20, 25) - (30, 25) - (50, 35) - (70, 35); model bottom y = 20, x = 20 to 70.
    The slope is made of three soil layers. Layer 1 is the top layer, layer 2 the middle layer and
    layer 3 the bottom layer. The layers always appear in this order from top to bottom.

Site investigation
    Four boreholes were drilled at x = 20, 30, 50 and 70 m, each from the ground surface down to the
    model bottom (y = 20). The borehole logs below are the only information about the soil layers.
    The soil profile between the boreholes (the location and shape of the layer interfaces) is NOT
    known. Layer 1 and layer 2 are not found at x = 20 and x = 30, but both are found at x = 50 and
    x = 70, so each of them pinches out somewhere on the left of x = 50.

Soil properties
    Unit weight of all layers is fixed. Cohesion of layers 1 and 2 is fixed.
    Random variables: friction angle of layer 1, friction angle of layer 2, and cohesion and
    friction angle of layer 3. All random variables are mutually independent.
    Layer 1 is a slightly cohesive sand, layer 2 a frictional, almost cohesionless soil and layer 3
    a cohesive, low-friction clay. No pore-water pressure.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

TITLE = "Three-layer slope known only at four boreholes"

# ---------------------------------------------------------------- geometry
GROUND_SURFACE = [(20, 25), (30, 25), (50, 35), (70, 35)]
MODEL_BOTTOM = 20.0
X_RANGE = (20.0, 70.0)
DOMAIN = GROUND_SURFACE + [(70, MODEL_BOTTOM), (20, MODEL_BOTTOM)]

# ---------------------------------------------------------------- borehole logs
# x (m): list of (layer, top elevation, bottom elevation) in m, from ground surface to model bottom
BOREHOLES = {
    20.0: [("Layer 3", 25.0, 20.0)],
    30.0: [("Layer 3", 25.0, 20.0)],
    50.0: [("Layer 1", 35.0, 29.0), ("Layer 2", 29.0, 24.5), ("Layer 3", 24.5, 20.0)],
    70.0: [("Layer 1", 35.0, 31.0), ("Layer 2", 31.0, 24.0), ("Layer 3", 24.0, 20.0)],
}

# ---------------------------------------------------------------- soil statistics
UNIT_WEIGHT = {"Layer 1": 19.5, "Layer 2": 19.5, "Layer 3": 19.5}          # kN/m3, deterministic
PARAMS = {   # (symbol, unit, distribution, mean, COV)
    "Layer 1": [("c", "kPa", "deterministic", 4.0, 0.0), ("phi", "deg", "lognormal", 27.0, 0.10)],
    "Layer 2": [("c", "kPa", "deterministic", 1.0, 0.0), ("phi", "deg", "lognormal", 25.0, 0.15)],
    "Layer 3": [("c", "kPa", "lognormal", 18.0, 0.30), ("phi", "deg", "lognormal", 12.0, 0.20)],
}

# ---------------------------------------------------------------- plotting set-up
COLORS = {"Layer 1": "#8fd98f", "Layer 2": "#9db4f0", "Layer 3": "#f0908f"}
BH_WIDTH = 1.2
XLIM, YLIM, FIGSIZE = (5, 74), (18.5, 39), (12.5, 5.6)

SUMMARY = ["Ground surface (20, 25) - (30, 25) - (50, 35) - (70, 35); model bottom y = 20, x = 20 to 70.",
           "Boreholes at x = 20, 30, 50 and 70 m; soil profile between the boreholes is unknown.",
           "Layer order (top to bottom): layer 1, layer 2, layer 3.",
           "All random variables independent; no pore-water pressure."]


def ground_elevation(x):
    gx, gy = zip(*GROUND_SURFACE)
    return float(np.interp(x, gx, gy))


def lognormal_params(mean, cov):
    xi = np.sqrt(np.log(1 + cov**2))
    lam = np.log(mean) - 0.5 * xi**2
    return lam, xi


def print_info():
    print(TITLE)
    for line in SUMMARY:
        print(line)
    print()
    print(f"{'Borehole x (m)':<16}{'ground (m)':>11}   log (layer: top - bottom elevation, m)")
    for x, log in BOREHOLES.items():
        txt = "; ".join(f"{name}: {top:g} - {bot:g}" for name, top, bot in log)
        print(f"{x:<16g}{ground_elevation(x):>11g}   {txt}")
    print()
    hdr = (f"{'Layer':<12}{'Variable':<20}{'Distribution':<14}{'mean':>9}{'COV':>7}{'std':>9}"
           f"{'lambda':>9}{'xi':>8}")
    print(hdr)
    print("-" * len(hdr))
    for layer, items in PARAMS.items():
        print(f"{layer:<12}{'unit weight':<20}{'deterministic':<14}{UNIT_WEIGHT[layer]:>9g}   (kN/m3)")
        for sym, unit, dist, mean, cov in items:
            if dist == "deterministic":
                print(f"{'':<12}{sym + ' (' + unit + ')':<20}{dist:<14}{mean:>9g}")
                continue
            std = mean * cov
            line = f"{'':<12}{sym + ' (' + unit + ')':<20}{dist:<14}{mean:>9g}{cov:>7.3g}{std:>9.4g}"
            if dist == "lognormal":
                lam, xi = lognormal_params(mean, cov)
                line += f"{lam:>9.4f}{xi:>8.4f}"
            print(line)
    print("\n(lambda, xi) = mean and std of the log variable (lognormal case).")


def _label(layer):
    lines = [layer, f"$\\gamma$ = {UNIT_WEIGHT[layer]:g} kN/m$^3$"]
    for sym, unit, dist, mean, cov in PARAMS[layer]:
        if dist == "deterministic":
            lines.append(f"{sym} = {mean:g} {unit} (fixed)")
        else:
            lines.append(f"{sym}: {dist.capitalize()}, $\\mu$={mean:g} {unit}, COV={cov:g}")
    return "\n".join(lines)


def plot_section(save_path):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    # slope domain: soil profile unknown
    ax.add_patch(Polygon(DOMAIN, closed=True, fc="#f2f2f2", ec="k", lw=1.0, zorder=0))
    for xt, yt in [(40, 23.5), (60, 27.5)]:
        ax.text(xt, yt, "soil profile\nunknown", fontsize=9, ha="center", va="center",
                color="0.4", style="italic")
    gx, gy = zip(*GROUND_SURFACE)
    ax.plot(gx, gy, "k-", lw=2)
    # borehole columns (kept inside the model domain at the two side boundaries)
    for x, log in BOREHOLES.items():
        x0 = min(max(x - BH_WIDTH / 2, X_RANGE[0]), X_RANGE[1] - BH_WIDTH)
        for name, top, bot in log:
            ax.add_patch(Rectangle((x0, bot), BH_WIDTH, top - bot, fc=COLORS[name], ec="k", lw=0.8))
            ax.text(x0 + BH_WIDTH + 0.3, top - 0.05, f"{top:g}", fontsize=7, va="center",
                    bbox=dict(fc="white", ec="none", pad=0.5))
        ax.text(x0 + BH_WIDTH + 0.3, MODEL_BOTTOM, f"{MODEL_BOTTOM:g}", fontsize=7, va="center",
                bbox=dict(fc="white", ec="none", pad=0.5))
        ax.text(x0 + BH_WIDTH / 2, log[0][1] + 1.6, f"BH\nx = {x:g}", fontsize=8, ha="center",
                fontweight="bold")
    # key points of the ground surface
    for x, y in GROUND_SURFACE:
        ax.plot(x, y, "ko", ms=3, zorder=5)
    ax.text(40, 31.0, "ground surface (20, 25) - (30, 25) - (50, 35) - (70, 35)", fontsize=7.5,
            rotation=26.6, rotation_mode="anchor", ha="center", va="bottom")
    # legend with soil statistics
    handles = [Rectangle((0, 0), 1, 1, fc=COLORS[k], ec="k") for k in PARAMS]
    ax.legend(handles, [_label(k).replace("\n", ", ", 1).replace("\n", "\n  ") for k in PARAMS],
              loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=7.5, framealpha=0.9, borderpad=0.8, labelspacing=1.0)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.set_xlabel("Horizontal distance (m)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title(TITLE + " (elevations of layer boundaries in m)", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    return fig


if __name__ == "__main__":
    import os
    print_info()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       os.path.splitext(os.path.basename(__file__))[0] + ".png")
    plot_section(out)
    print(f"\nFigure saved to {out}")
    plt.show()
