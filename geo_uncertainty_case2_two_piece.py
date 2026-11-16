"""Geological uncertainty, case 2: two linear pieces for each interface between neighbouring boreholes.

Borehole information (three_layer_slope_boreholes.BOREHOLES)
    Layers 1 and 2 are absent at x = 20 and 30 and present at x = 50 and 70. Interfaces:
    z1(x) = base of layer 1, z2(x) = base of layer 2 (an absent layer has zero thickness).

Model
    Between two neighbouring boreholes a and b each interface is a polyline of TWO straight pieces,
        (a, z(a)) -> (x_k, y_k) -> (b, z(b)),
    passing exactly through the borehole data, with a random kink point (x_k, y_k):
        x_k  in (a + margin, b - margin): uniform, or Gaussian centred mid-way (sd = (b - a)/6)
             truncated to that range                                          (--kink-x)
        y_k  = straight-line value at x_k + e, e ~ uniform(-dy, dy) or N(0, (dy/2)^2)   (--dy-dist)
    Each interface and interval gets its own independent kink.

    Rules (realisations that break them are rejected and redrawn):
    * A layer that is absent at both boreholes of an interval stays absent there (no lenses): the
      interface has no kink and coincides with the boundary above it.
    * A layer present at both boreholes keeps a positive thickness over the whole interval.
    * Where a layer is absent at one borehole and present at the other (x = 30 - 50 here, i.e. the
      slope region), it pinches out exactly once: it does not disappear and reappear. The pinch-out
      location follows from the kink points (layer 1 where z1 meets the ground surface, layer 2
      where z2 meets z1), so it always lies between those two boreholes, inside the slope region.
    * Layer 3 keeps a positive thickness (z2 above the model bottom).
    Interfaces are clipped as in spencer_slope.SoilProfile: z2 <= z1 <= ground.

Outputs (saved next to this script)
    geo_case2_realisations.png   sampled profiles, pinch-out histograms, layer probability maps
    geo_case2_realisations.csv   interface vertices of every realisation
    geo_case2_pinchout.csv       pinch-out x of layers 1 and 2 for every realisation

Usage
    python geo_uncertainty_case2_two_piece.py                 # dy = 3 m, uniform kinks, 1000 realisations
    python geo_uncertainty_case2_two_piece.py --dy 5 --dy-dist normal --kink-x normal
    python geo_uncertainty_case2_two_piece.py --min-thickness 0.5 --n 5000

    from geo_uncertainty_case2_two_piece import sample_profiles
    profiles, pinch1, pinch2 = sample_profiles(100, rng=np.random.default_rng(1))
"""
import argparse
import os

import numpy as np
from scipy import stats

from geo_uncertainty_case1_pinchout import (HERE, borehole_interfaces, check_profile, plot_realisations,
                                            save_realisations, slope_region)
from spencer_slope import SoilProfile, ground
from three_layer_slope_boreholes import GROUND_SURFACE, MODEL_BOTTOM

GX = [gx for gx, _ in GROUND_SURFACE]

GRID = 201          # points per interval used to check the rules


def _kink(a, b, za, zb, rng, kink_x, dy, dy_dist, margin):
    if kink_x == "uniform":
        xk = rng.uniform(a + margin, b - margin)
    else:
        mid, sd = 0.5 * (a + b), (b - a) / 6
        xk = stats.truncnorm((a + margin - mid) / sd, (b - margin - mid) / sd, loc=mid, scale=sd).rvs(random_state=rng)
    e = rng.uniform(-dy, dy) if dy_dist == "uniform" else rng.normal(0.0, dy / 2)
    return xk, za + (zb - za) * (xk - a) / (b - a) + e


def _interp(pts, x):
    px, py = zip(*pts)
    return np.interp(x, px, py)


def _single_pinch(thickness, present_left, tol=1e-9):
    """True if presence along the interval changes at most once (absent -> present or the reverse)."""
    on = thickness > tol
    changes = np.count_nonzero(on[1:] != on[:-1])
    if present_left:
        return changes <= 1 and on[0]
    return changes <= 1 and on[-1]


def _pinch_x(x, raw, present_left, tol=1e-9):
    """x where the layer thickness (raw, unclipped difference) changes sign; interpolated linearly
    between check points, which is exact for piecewise-linear interfaces."""
    on = raw > tol
    i = np.flatnonzero(on[1:] != on[:-1])
    if not len(i):
        return np.nan
    i = i[0]
    r0, r1 = raw[i], raw[i + 1]
    return float(x[i] + (x[i + 1] - x[i]) * (0 - r0) / (r1 - r0)) if r1 != r0 else float(x[i])


def _check_grid(a, b, kinks):
    """Regular points plus every vertex inside (a, b) (kinks, ground breakpoints): the extremes and
    sign changes of piecewise-linear lines can only occur at these vertices."""
    extra = [k for k in list(kinks) + list(GX) if a < k < b]
    return np.unique(np.r_[np.linspace(a, b, GRID), extra])


def _sample_z1(a, b, z, present, rng, kink_x, dy, dy_dist, margin, min_thickness, max_tries):
    """Admissible base of layer 1 in [a, b]: (segment after a, pinch-out x, tries)."""
    if not (present[a][0] or present[b][0]):
        return [(b, z[b][0])], np.nan, 0
    for t in range(1, max_tries + 1):
        seg = [(a, z[a][0]), _kink(a, b, z[a][0], z[b][0], rng, kink_x, dy, dy_dist, margin), (b, z[b][0])]
        xg = _check_grid(a, b, [seg[1][0]])
        z1 = _interp(seg, xg)
        raw1 = ground(xg) - z1                                       # thickness of layer 1 (unclipped)
        if z1.min() <= MODEL_BOTTOM:
            continue
        if present[a][0] and present[b][0]:
            if raw1[1:-1].min() <= min_thickness:
                continue
        elif not _single_pinch(np.clip(raw1, 0, None)[1:], present[a][0]):
            continue
        pinch = _pinch_x(xg, raw1, present[a][0]) if present[a][0] != present[b][0] else np.nan
        return seg[1:], pinch, t
    raise RuntimeError(f"no admissible base of layer 1 in [{a}, {b}] after {max_tries} tries")


def _sample_z2(a, b, z, present, z1_full, z1_seg, rng, kink_x, dy, dy_dist, margin, min_thickness, tries_max):
    """Admissible base of layer 2 in [a, b] given layer 1: (segment, pinch-out x, tries) or None."""
    if not (present[a][1] or present[b][1]):
        return [p for p in z1_seg], np.nan, 0                       # coincides with the base of layer 1
    z1_kinks = [p[0] for p in z1_seg[:-1]]
    for t in range(1, tries_max + 1):
        seg = [(a, z[a][1]), _kink(a, b, z[a][1], z[b][1], rng, kink_x, dy, dy_dist, margin), (b, z[b][1])]
        xg = _check_grid(a, b, z1_kinks + [seg[1][0]])
        z1c = np.minimum(_interp(z1_full, xg), ground(xg))           # clipped base of layer 1
        z2 = _interp(seg, xg)
        raw2 = z1c - z2                                              # thickness of layer 2 (unclipped)
        if z2.min() <= MODEL_BOTTOM:
            continue
        if present[a][1] and present[b][1]:
            if raw2[1:-1].min() <= min_thickness:
                continue
        elif not _single_pinch(np.clip(raw2, 0, None)[1:], present[a][1]):
            continue
        pinch = _pinch_x(xg, raw2, present[a][1]) if present[a][1] != present[b][1] else np.nan
        return seg[1:], pinch, t
    return None


def sample_profile(rng, kink_x="uniform", dy=3.0, dy_dist="uniform", margin=1.0, min_thickness=0.0,
                   max_tries=2000):
    """One random profile; returns (SoilProfile, pinch-out x of layer 1, of layer 2, tries used).
    If layer 2 cannot fit under a sampled base of layer 1, that interval's base of layer 1 is redrawn."""
    z, present = borehole_interfaces()
    xs = sorted(z)
    z1_pts, z2_pts = [(xs[0], z[xs[0]][0])], [(xs[0], z[xs[0]][1])]
    pinch = [np.nan, np.nan]
    tries = 0
    opts = (rng, kink_x, dy, dy_dist, margin, min_thickness)
    for a, b in zip(xs[:-1], xs[1:]):
        for _ in range(max_tries):
            z1_seg, p1, t1 = _sample_z1(a, b, z, present, *opts, max_tries)
            tries += t1
            res = _sample_z2(a, b, z, present, z1_pts + z1_seg, z1_seg, *opts, 50)
            if res is None:
                tries += 50
                continue
            z2_seg, p2, t2 = res
            tries += t2
            break
        else:
            raise RuntimeError(f"no admissible interfaces in [{a}, {b}] after {max_tries} attempts")
        if np.isfinite(p1):
            pinch[0] = p1
        if np.isfinite(p2):
            pinch[1] = p2
        z1_pts, z2_pts = z1_pts + z1_seg, z2_pts + z2_seg
    return SoilProfile(z1_pts, z2_pts), pinch[0], pinch[1], tries


def sample_profiles(n, rng=None, **kw):
    """n random profiles and the pinch-out x of layers 1 and 2."""
    rng = np.random.default_rng() if rng is None else rng
    out = [sample_profile(rng, **kw) for _ in range(n)]
    profiles, p1, p2, tries = zip(*out)
    sample_profiles.tries = int(sum(tries))
    return list(profiles), np.array(p1), np.array(p2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000, help="number of realisations")
    ap.add_argument("--kink-x", default="uniform", choices=["uniform", "normal"],
                    help="distribution of the kink x between the two boreholes")
    ap.add_argument("--dy", type=float, default=3.0, help="size of the kink offset from the straight line (m)")
    ap.add_argument("--dy-dist", default="uniform", choices=["uniform", "normal"],
                    help="uniform(-dy, dy) or normal with sd = dy/2")
    ap.add_argument("--margin", type=float, default=1.0, help="minimum distance of a kink from a borehole (m)")
    ap.add_argument("--min-thickness", type=float, default=0.0,
                    help="minimum thickness of a layer between two boreholes where it is present at both (m)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    profiles, p1, p2 = sample_profiles(args.n, rng, kink_x=args.kink_x, dy=args.dy, dy_dist=args.dy_dist,
                                       margin=args.margin, min_thickness=args.min_thickness)
    toe, crest = slope_region()
    bad = sum(not check_profile(p) for p in profiles)
    z, present = borehole_interfaces()
    xs = sorted(z)
    n_int = sum(present[a][k] or present[b][k] for a, b in zip(xs[:-1], xs[1:]) for k in (0, 1))  # kinked pieces
    print(f"{args.n} realisations: kink x {args.kink_x}, offset {args.dy_dist} with dy = {args.dy:g} m; "
          f"acceptance rate {n_int * args.n / sample_profiles.tries:.0%}")
    print(f"Profiles not honouring the boreholes: {bad}")
    for name, p in [("layer 1", p1), ("layer 2", p2)]:
        inside = np.mean((p >= toe) & (p <= crest))
        print(f"Pinch-out of {name}: mean {np.nanmean(p):.2f} m, sd {np.nanstd(p):.2f} m, "
              f"range {np.nanmin(p):.2f} - {np.nanmax(p):.2f} m; inside the slope region "
              f"[{toe:g}, {crest:g}]: {inside:.0%}")
    save_realisations(profiles, "geo_case2", {"x_pinch_layer1": p1, "x_pinch_layer2": p2})
    plot_realisations(profiles, [p1, p2],
                      f"Case 2: two linear pieces between neighbouring boreholes (kink x {args.kink_x}, "
                      f"offset {args.dy_dist} +/- {args.dy:g} m)",
                      os.path.join(HERE, "geo_case2_realisations.png"), pinch_labels=("layer 1", "layer 2"))


if __name__ == "__main__":
    main()
