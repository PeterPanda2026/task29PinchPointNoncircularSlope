"""Geological uncertainty, case 1: one linear piece between boreholes, uncertain pinch-out of layer 2.

Borehole information (three_layer_slope_boreholes.BOREHOLES)
    Layers 1 and 2 are absent at x = 20 and 30 and present at x = 50 and 70, so both pinch out
    somewhere between x = 30 and 50. Interfaces: z1(x) = base of layer 1, z2(x) = base of layer 2
    (an absent layer has zero thickness, i.e. its base equals the base of the layer above).

Model
    * Between any two neighbouring boreholes each interface is ONE straight line through the
      borehole data (as in the conventional profile of spencer_slope.profile_from_boreholes).
    * z1 is therefore fixed: (20, 25) - (30, 25) - (50, 29) - (70, 31). Layer 1 thins to zero at
      the toe (30, 25).
    * Layer 2 pinches out at an uncertain point P = (x_p, z1(x_p)) on the base of layer 1. Its base
      z2 is one straight line from P to the borehole value at x = 50 (24.5); for x < x_p layer 2
      has zero thickness (z2 = z1). Between x = 50 and 70, z2 is the straight line 24.5 -> 24.
    * x_p is a random variable confined to the slope region: from the last borehole without layer 2
      to the first borehole with it, intersected with the slope face (toe x = 30 to crest x = 50).
      So the pinch-out never leaves the slope region, whatever distribution is used.

Distributions for x_p (--dist)
    Only bounded distributions are used, and every one has support EXACTLY [x_min, x_max] = [30, 50]
    (the slope region), so a pinch-out can fall anywhere on the slope face but never outside it.
    Unbounded distributions (e.g. the normal) are not used.
    uniform      U(x_min, x_max); mean = mid-interval, sd = (x_max - x_min)/sqrt(12) = 5.77 m
    triangular   on [x_min, x_max] with mode c = 3 mean - x_min - x_max (mean set by --mean);
                 symmetric for the default mean 40 m, sd = (x_max - x_min)/sqrt(24) = 4.08 m
    beta         x_min + (x_max - x_min) Beta(a, b), a and b from --mean and --sd (method of
                 moments); default sd = that of the symmetric triangular, 4.08 m
    The support is shared; the sd follows from the shape (only the beta sd is free).

Outputs (saved next to this script)
    geo_case1_distributions.png   pdf of x_p and the probability of layer 2 at each point for
                                  every distribution
    geo_case1_realisations.png    sampled profiles (chosen --dist) and layer probability maps
    geo_case1_realisations.csv    interface vertices of every realisation
    geo_case1_pinchout.csv        x_p of every realisation

Usage
    python geo_uncertainty_case1_pinchout.py                         # uniform, 1000 realisations
    python geo_uncertainty_case1_pinchout.py --dist triangular --mean 37
    python geo_uncertainty_case1_pinchout.py --dist beta --mean 38 --sd 3 --n 5000

    from geo_uncertainty_case1_pinchout import sample_profiles
    profiles, x_p = sample_profiles(100, "uniform", rng=np.random.default_rng(1))
    # each profile is a spencer_slope.SoilProfile, e.g. spencer_fs(surface, profiles[0], c, phi)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from scipy import stats

from spencer_slope import LAYER_NAMES, SoilProfile, ground, profile_from_boreholes
from three_layer_slope_boreholes import BOREHOLES, DOMAIN, GROUND_SURFACE, MODEL_BOTTOM, X_RANGE

HERE = os.path.dirname(os.path.abspath(__file__))
COLORS = ["#8fd98f", "#9db4f0", "#f0908f"]
DISTS = ["uniform", "triangular", "beta"]


# ================================================================ borehole data
def borehole_interfaces():
    """Interface elevations at the boreholes: dict x -> (z1, z2), plus layer presence."""
    conv = profile_from_boreholes()
    z = {x: (y1, y2) for x, y1, y2 in zip(conv.z1x, conv.z1y, conv.z2y)}
    present = {x: (ground(x) - z[x][0] > 1e-9, z[x][0] - z[x][1] > 1e-9) for x in z}
    return z, present


def slope_region():
    """x range of the slope face (between the flat toe and the flat crest of the ground)."""
    gx, gy = map(np.asarray, zip(*GROUND_SURFACE))
    sloping = np.where(np.abs(np.diff(gy)) > 1e-9)[0]
    return float(gx[sloping[0]]), float(gx[sloping[-1] + 1])


def pinch_interval(layer):
    """Where layer (0 = layer 1, 1 = layer 2) can pinch out: between the last borehole without it and
    the first borehole with it, intersected with the slope region."""
    z, present = borehole_interfaces()
    xs = sorted(z)
    for a, b in zip(xs[:-1], xs[1:]):
        if not present[a][layer] and present[b][layer]:
            toe, crest = slope_region()
            lo, hi = max(a, toe), min(b, crest)
            if lo >= hi:
                raise ValueError("pinch-out interval does not overlap the slope region")
            return lo, hi
    raise ValueError(f"no pinch-out of {LAYER_NAMES[layer]} found in the borehole data")


# ================================================================ pinch-out distribution
def pinch_distribution(dist, lo, hi, mean=None, sd=None):
    """scipy frozen distribution of x_p with support exactly [lo, hi].
    mean: triangular and beta (default mid-interval; the uniform mean is always mid-interval).
    sd:   beta only (default (hi - lo)/sqrt(24), the sd of the symmetric triangular)."""
    w = hi - lo
    mean = 0.5 * (lo + hi) if mean is None else mean
    if not lo < mean < hi:
        raise ValueError(f"mean must lie strictly inside [{lo:g}, {hi:g}]")
    if dist == "uniform":
        return stats.uniform(loc=lo, scale=w)
    if dist == "triangular":
        c = 3 * mean - lo - hi
        if not lo <= c <= hi:
            raise ValueError(f"a triangular on [{lo:g}, {hi:g}] needs a mean in "
                             f"[{(2 * lo + hi) / 3:.4g}, {(lo + 2 * hi) / 3:.4g}] m")
        return stats.triang((c - lo) / w, loc=lo, scale=w)
    if dist == "beta":
        sd = w / np.sqrt(24) if sd is None else sd
        m, v = (mean - lo) / w, (sd / w) ** 2
        if not 0 < v < m * (1 - m):
            raise ValueError(f"beta sd must be in (0, {w * np.sqrt(m * (1 - m)):.4g}) m for mean {mean:g} m")
        k = m * (1 - m) / v - 1
        return stats.beta(m * k, (1 - m) * k, loc=lo, scale=w)
    raise ValueError(f"unknown distribution '{dist}'; choose from {DISTS}")


# ================================================================ profiles
def profile_for_pinch(x_p):
    """SoilProfile with z1 through the boreholes and layer 2 pinching out at x_p."""
    z, present = borehole_interfaces()
    xs = sorted(z)
    z1_pts = [(x, z[x][0]) for x in xs]
    lo, hi = pinch_interval(1)
    z1_at = lambda x: float(np.interp(x, [p[0] for p in z1_pts], [p[1] for p in z1_pts]))
    z2_pts = []
    for x in xs:
        if x <= lo:                       # boreholes left of the pinch interval: layer 2 absent
            z2_pts.append((x, z[x][1]))
    z2_pts.append((x_p, z1_at(x_p)))      # pinch point on the base of layer 1
    z2_pts += [(x, z[x][1]) for x in xs if x >= hi]
    # z2 = z1 between the last absent borehole and x_p (zero thickness)
    z2_pts = sorted(set(z2_pts))
    return SoilProfile(z1_pts, z2_pts)


def sample_profiles(n, dist="uniform", rng=None, **kw):
    """n random profiles and their pinch-out x of layer 2."""
    rng = np.random.default_rng() if rng is None else rng
    lo, hi = pinch_interval(1)
    x_p = pinch_distribution(dist, lo, hi, **kw).rvs(size=n, random_state=rng)
    x_p = np.clip(x_p, lo, np.nextafter(hi, lo))          # never exactly at the borehole
    return [profile_for_pinch(x) for x in x_p], x_p


# ================================================================ checks and maps
def check_profile(p):
    """True if the profile honours every borehole log."""
    z, _ = borehole_interfaces()
    return all(abs(p.z1(x) - z1) < 1e-9 and abs(p.z2(x) - z2) < 1e-9 for x, (z1, z2) in z.items())


def layer_probability(profiles, nx=251, ny=91):
    """Fraction of realisations in which each grid point belongs to layer 1, 2, 3."""
    y_top = max(y for _, y in GROUND_SURFACE)
    xg = np.linspace(*X_RANGE, nx)
    yg = np.linspace(MODEL_BOTTOM, y_top, ny)
    X, Y = np.meshgrid(xg, yg)
    counts = np.zeros((3,) + X.shape)
    for p in profiles:
        idx = p.layer_index(X, Y)
        for k in range(3):
            counts[k] += idx == k
    prob = counts / len(profiles)
    prob[:, Y > ground(X) + 1e-9] = np.nan
    return X, Y, prob


def draw_frame(ax):
    ax.plot(*zip(*GROUND_SURFACE), "k-", lw=1.8)
    ax.plot([X_RANGE[0], X_RANGE[1]], [MODEL_BOTTOM, MODEL_BOTTOM], "k-", lw=0.8)
    for x, log in BOREHOLES.items():
        for name, top, bot in log:
            ax.add_patch(plt.Rectangle((x - 0.5, bot), 1.0, top - bot, fc=COLORS[LAYER_NAMES.index(name)],
                                       ec="k", lw=0.6, zorder=5))
    ax.set_xlim(X_RANGE[0] - 1, X_RANGE[1] + 1)
    ax.set_ylim(MODEL_BOTTOM - 0.5, max(y for _, y in GROUND_SURFACE) + 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Elevation (m)")


def plot_realisations(profiles, pinches, title, path, n_show=60, pinch_labels=("layer 2",)):
    """Overlay of realisations + probability maps of layers 1-3 (+ pinch-out histograms)."""
    X, Y, prob = layer_probability(profiles)
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, :2])
    for poly, col in zip(profiles[0].polygons(), COLORS):
        ax.add_patch(Polygon(poly, closed=True, fc=col, ec="none", alpha=0.35))
    x = np.linspace(*X_RANGE, 400)
    for p in profiles[:n_show]:
        ax.plot(x, p.z1(x), color="darkgreen", lw=0.5, alpha=0.5)
        ax.plot(x, p.z2(x), color="navy", lw=0.5, alpha=0.5)
    ax.plot([], [], color="darkgreen", label="base of layer 1")
    ax.plot([], [], color="navy", label="base of layer 2")
    draw_frame(ax)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"{min(n_show, len(profiles))} of {len(profiles)} realisations (fill: realisation 1)", fontsize=9)
    for k in range(3):
        ax = fig.add_subplot(gs[1, k])
        im = ax.pcolormesh(X, Y, prob[k], cmap="viridis", vmin=0, vmax=1, shading="auto")
        draw_frame(ax)
        fig.colorbar(im, ax=ax, shrink=0.8, label="probability")
        ax.set_title(f"P({LAYER_NAMES[k]})", fontsize=9)
    # pinch-out histograms in their own panel, so no realisation lines are hidden
    hx = fig.add_subplot(gs[0, 2])
    toe, crest = slope_region()
    for pv, lab in zip(pinches, pinch_labels):
        hx.hist(pv[np.isfinite(pv)], bins=30, range=(toe, crest), alpha=0.6, label=lab)
    hx.set_xlim(toe, crest)
    hx.set_xlabel("pinch-out x (m)")
    hx.set_ylabel("count")
    hx.set_title(f"Pinch-out location (slope region {toe:g} - {crest:g} m)", fontsize=9)
    hx.legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("  saved", os.path.basename(path))


def save_realisations(profiles, prefix, extra):
    rows = []
    for i, p in enumerate(profiles):
        for name, xs, ys in [("base_layer1", p.z1x, p.z1y), ("base_layer2", p.z2x, p.z2y)]:
            rows += [{"realisation": i, "interface": name, "x": float(a), "y": float(b)} for a, b in zip(xs, ys)]
    pd.DataFrame(rows).to_csv(os.path.join(HERE, f"{prefix}_realisations.csv"), index=False)
    pd.DataFrame({"realisation": range(len(profiles)), **extra}).to_csv(
        os.path.join(HERE, f"{prefix}_pinchout.csv"), index=False)
    print(f"  saved {prefix}_realisations.csv and {prefix}_pinchout.csv")


# ================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dist", default="uniform", choices=DISTS, help="distribution of the pinch-out x")
    ap.add_argument("--mean", type=float, help="mean of x_p for triangular and beta (default mid-interval)")
    ap.add_argument("--sd", type=float, help="sd of x_p for beta (default (x_max - x_min)/sqrt(24))")
    ap.add_argument("--n", type=int, default=1000, help="number of realisations")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    kw = dict(mean=args.mean, sd=args.sd)

    lo, hi = pinch_interval(1)
    toe, crest = slope_region()
    print(f"Slope region x = {toe:g} - {crest:g} m; layer 2 pinch-out allowed in x = {lo:g} - {hi:g} m")
    for dist in DISTS:
        d = pinch_distribution(dist, lo, hi, **kw)
        a, b = d.support()
        print(f"  {dist:<11}mean {d.mean():.3f} m, sd {d.std():.3f} m, support {a:.2f} - {b:.2f} m")
    rng = np.random.default_rng(args.seed)
    profiles, x_p = sample_profiles(args.n, args.dist, rng, **kw)
    bad = sum(not check_profile(p) for p in profiles)
    print(f"{args.n} realisations ({args.dist}): x_p mean {x_p.mean():.2f} m, sd {x_p.std():.2f} m, "
          f"range {x_p.min():.2f} - {x_p.max():.2f} m; profiles not honouring the boreholes: {bad}")
    save_realisations(profiles, "geo_case1", {"x_pinch_layer2": x_p})
    plot_realisations(profiles, [x_p], f"Case 1: one linear piece between boreholes, layer 2 pinch-out "
                      f"~ {args.dist} on [{lo:g}, {hi:g}] m", os.path.join(HERE, "geo_case1_realisations.png"))

    # compare the distributions
    fig, axes = plt.subplots(2, len(DISTS), figsize=(4.4 * len(DISTS), 6.5),
                             gridspec_kw={"height_ratios": [1, 1.1]})
    xs = np.linspace(lo, hi, 300)
    for j, dist in enumerate(DISTS):
        d = pinch_distribution(dist, lo, hi, **kw)
        axes[0, j].plot(xs, d.pdf(xs), "k-")
        axes[0, j].fill_between(xs, d.pdf(xs), alpha=0.2)
        axes[0, j].set_xlim(X_RANGE)
        axes[0, j].axvspan(X_RANGE[0], lo, color="grey", alpha=0.15)
        axes[0, j].axvspan(hi, X_RANGE[1], color="grey", alpha=0.15)
        axes[0, j].set_title(f"{dist}: mean {d.mean():.1f} m, sd {d.std():.2f} m", fontsize=9)
        axes[0, j].set_xlabel("pinch-out x of layer 2 (m)")
        prof, _ = sample_profiles(400, dist, np.random.default_rng(args.seed), **kw)
        X, Y, prob = layer_probability(prof)
        im = axes[1, j].pcolormesh(X, Y, prob[1], cmap="viridis", vmin=0, vmax=1, shading="auto")
        draw_frame(axes[1, j])
        axes[1, j].set_title("P(layer 2)", fontsize=9)
        if j:
            axes[1, j].set_ylabel("")
    axes[0, 0].set_ylabel("pdf")
    fig.colorbar(im, ax=axes[1, :].tolist(), shrink=0.7, label="probability")
    fig.suptitle(f"Case 1: distributions of the layer 2 pinch-out, all on [{lo:g}, {hi:g}] m "
                 f"(grey = outside the slope region)", fontsize=10)
    fig.savefig(os.path.join(HERE, "geo_case1_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved geo_case1_distributions.png")


if __name__ == "__main__":
    main()
