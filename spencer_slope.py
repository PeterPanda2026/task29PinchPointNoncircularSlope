"""Slope stability of the three-layer borehole slope by the Spencer method (limit equilibrium).

Contents
    SoilProfile       layered profile defined by two interface polylines (base of layer 1 and
                      base of layer 2). Any profile that honours the boreholes can be passed in,
                      so the same solver can be reused for randomly generated profiles.
    profile_from_boreholes   conventional profile: same layers in neighbouring boreholes joined by
                      straight lines (a layer absent in a borehole has zero thickness there).
    spencer_fs        Spencer factor of safety for any kinematically admissible slip surface
                      given as a polyline (circular or non-circular).
    search_critical_circle   global search (differential evolution + Nelder-Mead polish) for the
                      critical circular slip surface.

Spencer method
    Interslice forces are parallel (constant inclination theta). For each slice, force
    equilibrium in two directions gives the base normal force N and the change of the interslice
    force. F and theta are found from (1) zero interslice force at the last slice (overall force
    equilibrium) and (2) overall moment equilibrium of the sliding mass. Moment equilibrium is
    taken about a fixed point, so the method is valid for non-circular surfaces too.

The slope faces left: the sliding mass moves towards -x. No pore-water pressure.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.optimize import brentq, differential_evolution, minimize, root

from three_layer_slope_boreholes import (BOREHOLES, GROUND_SURFACE, MODEL_BOTTOM, PARAMS,
                                         UNIT_WEIGHT, X_RANGE)

LAYER_NAMES = ["Layer 1", "Layer 2", "Layer 3"]
GX, GY = map(np.asarray, zip(*GROUND_SURFACE))


def ground(x):
    return np.interp(x, GX, GY)


# ================================================================ soil profile
class SoilProfile:
    """Three layers separated by two interfaces z1(x) (base of layer 1) and z2(x) (base of layer 2).

    Each interface is a polyline [(x, y), ...] covering x = 20 to 70. It is clipped so that
    z2 <= z1 <= ground, i.e. a layer has zero thickness where the interfaces meet.
    """

    def __init__(self, z1_pts, z2_pts):
        self.z1x, self.z1y = map(np.asarray, zip(*sorted(z1_pts)))
        self.z2x, self.z2y = map(np.asarray, zip(*sorted(z2_pts)))

    def z1(self, x):
        return np.minimum(np.interp(x, self.z1x, self.z1y), ground(x))

    def z2(self, x):
        return np.minimum(np.interp(x, self.z2x, self.z2y), self.z1(x))

    def layer_index(self, x, y):
        """0, 1, 2 for layer 1, 2, 3 at points (x, y)."""
        return np.where(y > self.z1(x), 0, np.where(y > self.z2(x), 1, 2))

    def column_weight(self, x, y_base, gamma):
        """Weight per unit width of the soil column from y_base up to the ground surface."""
        zg, z1, z2 = ground(x), self.z1(x), self.z2(x)
        h1 = np.clip(zg - np.maximum(z1, y_base), 0, None)
        h2 = np.clip(np.minimum(z1, zg) - np.maximum(z2, y_base), 0, None)
        h3 = np.clip(np.minimum(z2, zg) - y_base, 0, None)
        return gamma[0] * h1 + gamma[1] * h2 + gamma[2] * h3

    def breakpoints(self):
        return np.unique(np.concatenate([GX, self.z1x, self.z2x]))

    def polygons(self, n=400):
        """Layer polygons for plotting."""
        x = np.linspace(*X_RANGE, n)
        zg, z1, z2 = ground(x), self.z1(x), self.z2(x)
        zb = np.full_like(x, MODEL_BOTTOM)
        poly = lambda top, bot: np.column_stack([np.r_[x, x[::-1]], np.r_[top, bot[::-1]]])
        return [poly(zg, z1), poly(z1, z2), poly(z2, zb)]


def profile_from_boreholes(boreholes=BOREHOLES):
    """Conventional profile: interfaces at each borehole joined by straight lines."""
    z1_pts, z2_pts = [], []
    for x, log in sorted(boreholes.items()):
        bottoms = {name: bot for name, top, bot in log}
        top_of_log = log[0][1]
        b1 = bottoms.get("Layer 1", top_of_log)       # absent layer 1 -> base at ground
        b2 = bottoms.get("Layer 2", b1)               # absent layer 2 -> base at base of layer 1
        z1_pts.append((x, b1))
        z2_pts.append((x, b2))
    return SoilProfile(z1_pts, z2_pts)


def mean_strength():
    """(c, phi) per layer at the mean values: arrays of length 3."""
    c = np.array([dict((s, m) for s, _, _, m, _ in PARAMS[k])["c"] for k in LAYER_NAMES])
    phi = np.array([dict((s, m) for s, _, _, m, _ in PARAMS[k])["phi"] for k in LAYER_NAMES])
    return c, phi


GAMMA = np.array([UNIT_WEIGHT[k] for k in LAYER_NAMES])


# ================================================================ slices
def make_slices(surface, profile, gamma=GAMMA, n_slices=60):
    """Slice data for a slip surface polyline [(x, y), ...] ordered by increasing x.

    The surface must start and end on the ground surface and lie below it in between.
    """
    sx, sy = map(np.asarray, zip(*surface))
    edges = np.linspace(sx[0], sx[-1], n_slices + 1)
    extra = np.concatenate([sx, profile.breakpoints()])
    extra = extra[(extra > sx[0] + 1e-6) & (extra < sx[-1] - 1e-6)]
    edges = np.unique(np.round(np.concatenate([edges, extra]), 9))
    edges = edges[np.r_[True, np.diff(edges) > 1e-4]]
    yb_edges = np.interp(edges, sx, sy)
    xm = 0.5 * (edges[:-1] + edges[1:])
    b = np.diff(edges)
    dy = np.diff(yb_edges)
    ybm = 0.5 * (yb_edges[:-1] + yb_edges[1:])
    length = np.hypot(b, dy)
    # weight: average of the column weights at both edges and the middle (Simpson rule)
    w_left = profile.column_weight(edges[:-1], yb_edges[:-1], gamma)
    w_mid = profile.column_weight(xm, ybm, gamma)
    w_right = profile.column_weight(edges[1:], yb_edges[1:], gamma)
    W = b * (w_left + 4 * w_mid + w_right) / 6.0
    lay = profile.layer_index(xm, ybm + 1e-9)
    return dict(xm=xm, ybm=ybm, b=b, dy=dy, l=length, W=W, layer=lay)


# ================================================================ Spencer method
def _slice_forces(F, theta, sl, c, tphi):
    """Base normal force N and interslice force change dZ for every slice.

    Slice equilibrium: W + N n - S s + dZ d = 0, with S = (c l + N tan phi) / F,
    s = sliding direction along the base (towards -x), n = base normal pointing into the slice,
    d = (cos theta, sin theta) = direction of the interslice force on the left face of a slice.
    """
    tx, ty = sl["b"] / sl["l"], sl["dy"] / sl["l"]          # base tangent towards +x
    sx_, sy_ = -tx, -ty                                       # sliding direction (towards -x)
    nx, ny = -ty, tx                                          # upward base normal
    k = tphi / F
    a11, a21 = nx - k * sx_, ny - k * sy_                     # coefficient of N
    a12, a22 = np.cos(theta), np.sin(theta)                   # coefficient of dZ
    cl = c * sl["l"] / F
    r1 = cl * sx_                                  # right-hand side: -W_vec + (c l / F) s, W_vec = (0, -W)
    r2 = sl["W"] + cl * sy_
    det = a11 * a22 - a12 * a21
    N = (r1 * a22 - a12 * r2) / det
    dZ = (a11 * r2 - r1 * a21) / det
    S = (c * sl["l"] + N * tphi) / F
    return N, dZ, S, det, (sx_, sy_, nx, ny)


def _residuals(F, theta, sl, c, tphi, x0, y0):
    N, dZ, S, det, (sx_, sy_, nx, ny) = _slice_forces(F, theta, sl, c, tphi)
    Wsum = sl["W"].sum()
    r_force = dZ.sum() / Wsum                                 # interslice force at the last slice
    # overall moment about (x0, y0): weights + base forces (interslice forces are internal)
    px, py = sl["xm"] - x0, sl["ybm"] - y0
    fx, fy = N * nx - S * sx_, N * ny - S * sy_
    moment = np.sum(px * (-sl["W"])) + np.sum(px * fy - py * fx)
    r_moment = moment / (Wsum * (np.ptp(sl["xm"]) + 1.0))
    return r_force, r_moment, det


def spencer_fs(surface, profile, c, phi, gamma=GAMMA, n_slices=60, return_details=False):
    """Spencer factor of safety for a slip surface polyline (ordered by increasing x).

    c (kPa) and phi (deg) are arrays with one value per layer. Returns np.nan if no valid
    solution is found.
    """
    sl = make_slices(surface, profile, gamma, n_slices)
    if sl["W"].sum() <= 0:
        return np.nan
    cs, tphi = np.asarray(c, float)[sl["layer"]], np.tan(np.radians(np.asarray(phi, float)))[sl["layer"]]
    x0, y0 = sl["xm"].mean(), sl["ybm"].max() + 10.0         # any fixed point works

    def fun(v):
        rf, rm, _ = _residuals(v[0], v[1], sl, cs, tphi, x0, y0)
        return [rf, rm]

    sol = None
    for F0, th0 in [(1.5, 0.2), (1.0, 0.0), (2.5, 0.4), (1.2, -0.2)]:
        r = root(fun, [F0, th0], method="hybr")
        if r.success and 0.05 < r.x[0] < 50 and abs(r.x[1]) < np.radians(80) \
                and np.max(np.abs(fun(r.x))) < 1e-8:
            sol = r.x
            break
    if sol is None:                                           # fallback: nested root finding
        sol = _spencer_nested(sl, cs, tphi, x0, y0)
        if sol is None:
            return np.nan
    F, theta = sol
    _, _, det = _residuals(F, theta, sl, cs, tphi, x0, y0)
    if np.min(np.abs(det)) < 0.1:                             # numerically unstable slice (m_alpha)
        return np.nan
    if not return_details:
        return F
    N, dZ, S, _, _ = _slice_forces(F, theta, sl, cs, tphi)
    Z = np.r_[0.0, -np.cumsum(dZ)]                            # interslice force at slice boundaries
    return dict(F=F, theta=np.degrees(theta), N=N, S=S, Z=Z, slices=sl)


def _spencer_nested(sl, cs, tphi, x0, y0):
    """For each theta solve F from force and from moment equilibrium; theta where they agree."""
    def f_of(theta, which):
        g = lambda F: _residuals(F, theta, sl, cs, tphi, x0, y0)[which]
        Fs = np.linspace(0.2, 10, 50)
        v = np.array([g(F) for F in Fs])
        idx = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
        return brentq(g, Fs[idx[0]], Fs[idx[0] + 1]) if len(idx) else np.nan

    ths = np.radians(np.linspace(-40, 60, 41))
    diff = np.array([f_of(t, 0) - f_of(t, 1) for t in ths])
    ok = np.isfinite(diff)
    idx = np.where(ok[:-1] & ok[1:] & (np.sign(diff[:-1]) != np.sign(diff[1:])))[0]
    if not len(idx):
        return None
    th = brentq(lambda t: f_of(t, 0) - f_of(t, 1), ths[idx[0]], ths[idx[0] + 1])
    return f_of(th, 0), th


# ================================================================ circular slip surfaces
def circle_surface(x_exit, x_entry, k, n=80):
    """Circular slip surface through ground points at x_exit (toe side) and x_entry (crest side).

    k >= 0.5 is the radius divided by the chord length (large k = shallow surface).
    Returns (polyline, centre, radius) or None if the surface is not admissible.
    """
    if not (X_RANGE[0] <= x_exit < x_entry <= X_RANGE[1]):
        return None
    A = np.array([x_exit, ground(x_exit)])
    B = np.array([x_entry, ground(x_entry)])
    chord = np.linalg.norm(B - A)
    R = k * chord
    if R <= chord / 2:
        return None
    mid, u = (A + B) / 2, (B - A) / chord
    nrm = np.array([-u[1], u[0]])                             # normal pointing up-left of chord
    centre = mid + nrm * np.sqrt(R**2 - (chord / 2) ** 2)
    xs = np.linspace(x_exit, x_entry, n)
    ys = centre[1] - np.sqrt(np.clip(R**2 - (xs - centre[0]) ** 2, 0, None))
    if np.any(np.abs(xs - centre[0]) > R) or ys.min() < MODEL_BOTTOM - 1e-9:
        return None
    if np.any(ys[1:-1] > ground(xs[1:-1]) + 1e-6):            # surface must stay below ground
        return None
    if np.max(ground(xs) - ys) < 0.5:                         # ignore trivially thin slides
        return None
    return list(zip(xs, ys)), centre, R


def search_critical_circle(profile, c, phi, gamma=GAMMA, seed=0, maxiter=60, verbose=False):
    """Minimum Spencer FS over circular slip surfaces. Returns (FS, surface, centre, radius)."""
    def obj(v):
        circ = circle_surface(*v)
        if circ is None:
            return 1e3
        F = spencer_fs(circ[0], profile, c, phi, gamma)
        return F if np.isfinite(F) else 1e3

    bounds = [(20.0, 49.0), (32.0, 70.0), (0.51, 4.0)]
    de = differential_evolution(obj, bounds, seed=seed, maxiter=maxiter, popsize=15, tol=1e-6,
                                polish=False, disp=verbose)
    best = minimize(obj, de.x, method="Nelder-Mead",
                    options=dict(xatol=1e-3, fatol=1e-6, maxiter=400))
    v = best.x if best.fun < de.fun else de.x
    surface, centre, R = circle_surface(*v)
    return min(best.fun, de.fun), surface, centre, R, v


# ================================================================ plotting
COLORS = ["#8fd98f", "#9db4f0", "#f0908f"]


def plot_result(profile, surface, F, centre, save_path, title):
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    for poly, col, name in zip(profile.polygons(), COLORS, LAYER_NAMES):
        ax.add_patch(Polygon(poly, closed=True, fc=col, ec="k", lw=0.6, label=name))
    ax.plot(GX, GY, "k-", lw=2)
    for x, log in BOREHOLES.items():
        ax.plot([x, x], [log[0][1], MODEL_BOTTOM], "k--", lw=0.8)
        ax.text(x, log[0][1] + 0.6, f"BH {x:g}", ha="center", fontsize=8)
    sx, sy = zip(*surface)
    ax.plot(sx, sy, "r-", lw=2.2, label=f"critical slip surface, FS = {F:.3f}")
    if centre is not None:
        title += f"\ncircle centre ({centre[0]:.2f}, {centre[1]:.2f})"
    ax.set_xlim(17, 73)
    ax.set_ylim(18.5, 39)
    ax.set_aspect("equal")
    ax.set_xlabel("Horizontal distance (m)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    return fig


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    prof = profile_from_boreholes()
    c, phi = mean_strength()
    print("Conventional profile (straight lines between boreholes)")
    print("  base of layer 1:", list(zip(prof.z1x.tolist(), prof.z1y.tolist())))
    print("  base of layer 2:", list(zip(prof.z2x.tolist(), prof.z2y.tolist())))
    print("Mean strength: c =", c, "kPa, phi =", phi, "deg")
    F, surf, centre, R, v = search_critical_circle(prof, c, phi)
    det = spencer_fs(surf, prof, c, phi, return_details=True)
    print(f"\nCritical circular slip surface (Spencer): FS = {F:.4f}, theta = {det['theta']:.2f} deg")
    print(f"  exit x = {v[0]:.2f} m, entry x = {v[1]:.2f} m, centre = ({centre[0]:.2f}, {centre[1]:.2f}),"
          f" R = {R:.2f} m, lowest point y = {min(y for _, y in surf):.2f} m")
    out = os.path.join(here, "spencer_conventional_profile.png")
    plot_result(prof, surf, F, centre, out,
                "Conventional profile (boreholes joined by straight lines), mean soil parameters")
    print(f"Figure saved to {out}")
