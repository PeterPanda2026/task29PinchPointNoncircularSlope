"""Karhunen-Loeve (KL) expansion of Gaussian random fields with several autocorrelation functions.

A zero-mean, unit-variance stationary Gaussian field G on a domain D is written as

    G(x) = sum_k sqrt(lambda_k) phi_k(x) xi_k ,     xi_k ~ N(0, 1) independent,

where (lambda_k, phi_k) are the eigenpairs of the correlation kernel:

    integral_D rho(x, x') phi_k(x') dx' = lambda_k phi_k(x).

The series is truncated after M terms so that sum_{k<=M} lambda_k / |D| reaches a target share of
the variance (|D| = length or area of the domain = sum of all eigenvalues for unit variance).

Autocorrelation functions (all written with the scale of fluctuation delta = integral of rho over
the whole lag axis, so they can be compared at equal delta; tau = lag distance)
    markov              single exponential      exp(-2|tau|/delta)
    gaussian            squared exponential     exp(-pi (tau/delta)^2)
    second_order_markov                         (1 + 4|tau|/delta) exp(-4|tau|/delta)
    cosine_exponential                          exp(-|tau|/delta) cos(tau/delta)
    binary_noise        triangular              1 - |tau|/delta for |tau| <= delta, else 0
    whittle_matern      Matern, smoothness nu   2^(1-nu)/Gamma(nu) (|tau|/l)^nu K_nu(|tau|/l),
                                                l = delta Gamma(nu) / (2 sqrt(pi) Gamma(nu + 1/2))
                                                (nu = 0.5 is markov, nu = 1.5 is exactly
                                                second_order_markov, nu -> inf tends to
                                                gaussian; default nu = 2.5)

Numerical method
    1-D: Nystrom method with Gauss-Legendre quadrature (works for any of the kernels). The
         eigenfunctions are evaluated anywhere with the Nystrom interpolation formula
             phi_k(x) = 1/lambda_k * sum_j w_j rho(x - x_j) phi_k(x_j).
    2-D: separable (anisotropic) correlation rho(tau_x) rho(tau_y) on the bounding rectangle of the
         slope, with horizontal and vertical scales of fluctuation delta_h and delta_v. The 2-D
         eigenpairs are products of 1-D eigenpairs: lambda_ij = lambda_i^x lambda_j^y,
         phi_ij(x, y) = phi_i(x) phi_j(y); they are sorted and truncated together.

Truncation and variance
    A truncated expansion has point-wise variance sum_k lambda_k phi_k(x)^2 < 1 (about the kept
    share of the variance on average, less near the domain edges). With normalise=True
    (--normalise-variance) every point is divided by its standard deviation, so the field has unit
    variance again and the soil property keeps its target COV; the correlation shape is then
    slightly changed instead.

Soil properties
    to_marginal() turns the standard Gaussian field into a normal or lognormal field with a given
    mean and COV, e.g. for the layer statistics in three_layer_slope_boreholes.PARAMS.

Usage
    python kl_expansion.py                         # all ACFs, delta_h = 20 m, delta_v = 2 m, figures
    python kl_expansion.py --acf markov gaussian --delta-h 10 --delta-v 1 --energy 0.9
    python kl_expansion.py --verify                # check against exact solutions and Monte Carlo
    python kl_expansion.py --layer "Layer 1" --param phi

    from kl_expansion import KL1D, KL2D, to_marginal
    kl = KL2D("markov", delta_x=20, delta_y=2, x_range=(20, 70), y_range=(20, 35), energy=0.95)
    G = kl.simulate(x, y, n_real=100, rng=np.random.default_rng(1))     # (len(x), 100)
    c = to_marginal(G, "lognormal", mean=18.0, cov=0.3)

Figures (saved next to this script)
    kl_acf_eigenvalues.png     the ACFs, their eigenvalue decay and captured variance
    kl_eigenfunctions.png      first eigenfunctions for each ACF
    kl_1d_realisations.png     1-D realisations with the same random numbers xi for every ACF
    kl_2d_slope_fields.png     2-D realisations of a soil property over the slope for every ACF
    kl_verification.png        (--verify) simulated vs target correlation
"""
import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from scipy.optimize import brentq
from scipy.special import gamma as gamma_fn, kv

from three_layer_slope_boreholes import DOMAIN, GROUND_SURFACE, MODEL_BOTTOM, PARAMS, X_RANGE, lognormal_params

HERE = os.path.dirname(os.path.abspath(__file__))
ACF_NAMES = ["markov", "gaussian", "second_order_markov", "cosine_exponential", "binary_noise",
             "whittle_matern"]


# ================================================================ autocorrelation functions
def correlation(acf, tau, delta, nu=2.5):
    """Correlation rho(tau) of the named ACF with scale of fluctuation delta."""
    t = np.abs(np.asarray(tau, float))
    if acf == "markov":
        return np.exp(-2.0 * t / delta)
    if acf == "gaussian":
        return np.exp(-np.pi * (t / delta) ** 2)
    if acf == "second_order_markov":
        return (1.0 + 4.0 * t / delta) * np.exp(-4.0 * t / delta)
    if acf == "cosine_exponential":
        return np.exp(-t / delta) * np.cos(t / delta)
    if acf == "binary_noise":
        return np.clip(1.0 - t / delta, 0.0, None)
    if acf == "whittle_matern":
        ell = delta * gamma_fn(nu) / (2.0 * math.sqrt(math.pi) * gamma_fn(nu + 0.5))
        s = t / ell
        with np.errstate(invalid="ignore"):
            r = 2.0 ** (1.0 - nu) / gamma_fn(nu) * s ** nu * kv(nu, s)
        return np.where(s < 1e-10, 1.0, np.nan_to_num(r, nan=0.0))
    raise ValueError(f"unknown ACF '{acf}'; choose from {ACF_NAMES}")


# ================================================================ 1-D KL (Nystrom)
class KL1D:
    """KL expansion of a unit-variance stationary field on the interval domain = (a, b)."""

    def __init__(self, acf, delta, domain, energy=0.95, n_quad=None, nu=2.5, max_terms=None):
        self.acf, self.delta, self.nu = acf, float(delta), nu
        self.a, self.b = map(float, domain)
        L = self.b - self.a
        if n_quad is None:                                    # enough nodes per scale of fluctuation
            n_quad = int(min(2000, max(200, 30 * L / self.delta)))
        t, w = np.polynomial.legendre.leggauss(n_quad)
        self.nodes = self.a + (t + 1) * L / 2
        self.weights = w * L / 2
        C = self._rho(self.nodes[:, None] - self.nodes[None, :])
        sw = np.sqrt(self.weights)
        lam, psi = np.linalg.eigh(sw[:, None] * C * sw[None, :])
        order = np.argsort(lam)[::-1]
        self.eigenvalues = np.clip(lam[order], 0.0, None)     # all n_quad eigenvalues, descending
        self.phi_nodes = psi[:, order] / sw[:, None]          # orthonormal in L2(domain)
        self.total = L                                        # = trace of the kernel (unit variance)
        cum = np.cumsum(self.eigenvalues) / self.total
        self.n_terms = int(np.searchsorted(cum, energy) + 1)
        if max_terms:
            self.n_terms = min(self.n_terms, max_terms)
        self.n_terms = min(self.n_terms, n_quad)

    def _rho(self, tau):
        return correlation(self.acf, tau, self.delta, self.nu)

    def energy_ratio(self, m=None):
        m = self.n_terms if m is None else m
        return float(self.eigenvalues[:m].sum() / self.total)

    def eigenfunctions(self, x, m=None):
        """phi_k(x) for k = 1..m at the points x (Nystrom interpolation); shape (len(x), m)."""
        m = self.n_terms if m is None else m
        x = np.atleast_1d(np.asarray(x, float))
        K = self._rho(x[:, None] - self.nodes[None, :]) * self.weights[None, :]
        lam = np.where(self.eigenvalues[:m] > 1e-14, self.eigenvalues[:m], np.inf)
        return K @ self.phi_nodes[:, :m] / lam[None, :]

    def variance(self, x, m=None):
        """Point-wise variance of the truncated expansion (1 = exact)."""
        m = self.n_terms if m is None else m
        return (self.eigenfunctions(x, m) ** 2) @ self.eigenvalues[:m]

    def simulate(self, x, n_real=1, rng=None, xi=None, normalise=False):
        """Standard Gaussian realisations at x; shape (len(x), n_real).
        normalise=True rescales every point to unit variance."""
        rng = np.random.default_rng() if rng is None else rng
        xi = rng.standard_normal((self.n_terms, n_real)) if xi is None else xi[:self.n_terms]
        phi = self.eigenfunctions(x)
        g = phi @ (np.sqrt(self.eigenvalues[:self.n_terms])[:, None] * xi)
        if normalise:
            g /= np.sqrt((phi ** 2) @ self.eigenvalues[:self.n_terms])[:, None]
        return g


# ================================================================ 2-D separable KL
class KL2D:
    """KL expansion of a unit-variance field with separable correlation rho_x(tau_x) rho_y(tau_y)
    on the rectangle x_range x y_range (delta_x horizontal, delta_y vertical)."""

    def __init__(self, acf, delta_x, delta_y, x_range, y_range, energy=0.95, nu=2.5, acf_y=None):
        self.acf, self.acf_y = acf, acf_y or acf
        self.kx = KL1D(acf, delta_x, x_range, energy=0.999, nu=nu)
        self.ky = KL1D(self.acf_y, delta_y, y_range, energy=0.999, nu=nu)
        lx = self.kx.eigenvalues[:self.kx.n_terms]
        ly = self.ky.eigenvalues[:self.ky.n_terms]
        prod = np.outer(lx, ly)
        order = np.argsort(prod.ravel())[::-1]
        self.total = self.kx.total * self.ky.total            # area of the rectangle
        lam = prod.ravel()[order]
        cum = np.cumsum(lam) / self.total
        self.n_terms = int(min(np.searchsorted(cum, energy) + 1, len(lam)))
        self.eigenvalues = lam[:self.n_terms]
        self.ix, self.iy = np.unravel_index(order[:self.n_terms], prod.shape)

    def energy_ratio(self):
        return float(self.eigenvalues.sum() / self.total)

    def eigenfunctions(self, x, y):
        """phi_k(x, y) for the retained terms; shape (len(x), n_terms)."""
        x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
        ux, inv_x = np.unique(x, return_inverse=True)        # evaluate each 1-D factor once
        uy, inv_y = np.unique(y, return_inverse=True)
        fx = self.kx.eigenfunctions(ux, self.ix.max() + 1)[inv_x]
        fy = self.ky.eigenfunctions(uy, self.iy.max() + 1)[inv_y]
        return fx[:, self.ix] * fy[:, self.iy]

    def variance(self, x, y):
        return (self.eigenfunctions(x, y) ** 2) @ self.eigenvalues

    def simulate(self, x, y, n_real=1, rng=None, xi=None, normalise=False):
        """Standard Gaussian realisations at the points (x, y); shape (len(x), n_real).
        normalise=True rescales every point to unit variance."""
        rng = np.random.default_rng() if rng is None else rng
        xi = rng.standard_normal((self.n_terms, n_real)) if xi is None else xi[:self.n_terms]
        phi = self.eigenfunctions(x, y)
        g = phi @ (np.sqrt(self.eigenvalues)[:, None] * xi)
        if normalise:
            g /= np.sqrt((phi ** 2) @ self.eigenvalues)[:, None]
        return g


# ================================================================ marginal transformation
def to_marginal(g, dist, mean, cov):
    """Standard Gaussian field g -> field with the given marginal distribution, mean and COV."""
    if dist == "normal":
        return mean + mean * cov * g
    if dist == "lognormal":
        lam, xi = lognormal_params(mean, cov)
        return np.exp(lam + xi * g)
    if dist == "deterministic":
        return np.full_like(g, mean, dtype=float)
    raise ValueError(f"unknown distribution '{dist}'")


def layer_statistics(layer, param):
    for sym, unit, dist, mean, cov in PARAMS[layer]:
        if sym == param:
            return dist, mean, cov, unit
    raise ValueError(f"{layer} has no parameter '{param}'")


# ================================================================ exact solution (verification)
def exact_markov_eigenvalues(delta, half_length, n):
    """Exact eigenvalues for rho = exp(-|tau|/b), b = delta/2, on [-a, a] (Ghanem & Spanos 1991):
    lambda = 2c / (w^2 + c^2), c = 1/b, with w from c - w tan(w a) = 0 (even) and
    w + c tan(w a) = 0 (odd functions)."""
    c, a = 2.0 / delta, half_length
    lams = []
    for k in range(n):
        lo, hi = k * np.pi / a + 1e-10, (k + 0.5) * np.pi / a - 1e-10
        w = brentq(lambda w: c - w * np.tan(w * a), lo, hi)                    # even
        lams.append(2 * c / (w ** 2 + c ** 2))
        lo, hi = (k + 0.5) * np.pi / a + 1e-10, (k + 1) * np.pi / a - 1e-10
        w = brentq(lambda w: w + c * np.tan(w * a), lo, hi)                    # odd
        lams.append(2 * c / (w ** 2 + c ** 2))
    return np.sort(lams)[::-1][:n]


def verify(acfs, delta, L, nu, energy, seed):
    print("\n=== Verification")
    kl = KL1D("markov", delta, (0, L), n_quad=600)
    ex = exact_markov_eigenvalues(delta, L / 2, 15)
    rel = np.abs(kl.eigenvalues[:15] - ex) / ex
    print(f"1. Markov eigenvalues, Nystrom vs exact (delta={delta:g} m, L={L:g} m): "
          f"max relative error of the first 15 = {rel.max():.2e}")
    G = kl.phi_nodes[:, :10]
    gram = G.T @ (kl.weights[:, None] * G)
    print(f"2. Orthonormality of the first 10 eigenfunctions: max |Gram - I| = {np.abs(gram - np.eye(10)).max():.2e}")
    rng = np.random.default_rng(seed)
    x = np.linspace(0, L, 101)
    lags = x - x[0]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True, sharey=True)
    print(f"3. Monte Carlo check, 4000 realisations, truncation at {energy:.0%} of the variance:")
    for ax, acf in zip(axes.flat, acfs):
        k = KL1D(acf, delta, (0, L), energy=energy, nu=nu)
        g = k.simulate(x, 4000, rng)
        emp = np.array([np.mean(g[: len(x) - j] * g[j:]) for j in range(len(x))])
        # target for the truncated field: average of the truncated covariance at each lag
        Ct = k.eigenfunctions(x) @ (k.eigenvalues[:k.n_terms, None] * k.eigenfunctions(x).T)
        trunc = np.array([np.mean(np.diag(Ct, j)) for j in range(len(x))])
        target = correlation(acf, lags, delta, nu)
        print(f"   {acf:<20s} terms={k.n_terms:3d}  mean variance={np.mean(np.diag(Ct)):.3f}  "
              f"max |MC - truncated model| = {np.abs(emp - trunc).max():.3f}  "
              f"max |truncated - target| = {np.abs(trunc - target).max():.3f}")
        ax.plot(lags, target, "k-", lw=2, label="target ACF")
        ax.plot(lags, trunc, "b--", label=f"truncated KL ({k.n_terms} terms)")
        ax.plot(lags, emp, "r:", lw=2, label="Monte Carlo")
        ax.set_title(acf, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("lag (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("correlation")
    fig.suptitle(f"KL verification, delta = {delta:g} m on a {L:g} m line")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "kl_verification.png"), dpi=150)
    plt.close(fig)
    print("   saved kl_verification.png")


# ================================================================ figures
def plot_acf_and_spectra(acfs, delta, L, nu, energy):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    tau = np.linspace(0, 3 * delta, 400)
    for acf in acfs:
        axes[0].plot(tau, correlation(acf, tau, delta, nu), label=acf)
        k = KL1D(acf, delta, (0, L), energy=energy, nu=nu)
        m = np.arange(1, 101)
        axes[1].semilogy(m, k.eigenvalues[:100] / k.total, label=acf)
        axes[2].plot(m, np.cumsum(k.eigenvalues[:100]) / k.total, label=f"{acf} ({k.n_terms} terms)")
    axes[0].axhline(0, color="grey", lw=0.5)
    axes[0].set_xlabel("lag tau (m)")
    axes[0].set_ylabel("rho(tau)")
    axes[0].set_title(f"Autocorrelation functions, delta = {delta:g} m", fontsize=10)
    axes[1].set_xlabel("term k")
    axes[1].set_ylabel("lambda_k / L")
    axes[1].set_ylim(1e-6, 1)
    axes[1].set_title(f"Eigenvalue decay on a {L:g} m line", fontsize=10)
    axes[2].axhline(energy, color="k", ls=":", lw=1)
    axes[2].set_xlabel("number of terms M")
    axes[2].set_ylabel("captured variance share")
    axes[2].set_title(f"Captured variance (terms for {energy:.0%} in legend)", fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "kl_acf_eigenvalues.png"), dpi=150)
    plt.close(fig)
    print("  saved kl_acf_eigenvalues.png")


def plot_eigenfunctions(acfs, delta, L, nu):
    x = np.linspace(0, L, 400)
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.5), sharex=True)
    for ax, acf in zip(axes.flat, acfs):
        k = KL1D(acf, delta, (0, L), nu=nu)
        phi = k.eigenfunctions(x, 4)
        for i in range(4):
            ax.plot(x, phi[:, i], label=f"k={i + 1}, lambda={k.eigenvalues[i]:.2f}")
        ax.set_title(acf, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("x (m)")
    fig.suptitle(f"First four KL eigenfunctions, delta = {delta:g} m")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "kl_eigenfunctions.png"), dpi=150)
    plt.close(fig)
    print("  saved kl_eigenfunctions.png")


def plot_1d_realisations(acfs, delta, L, nu, energy, seed, normalise=False):
    x = np.linspace(0, L, 500)
    xi = np.random.default_rng(seed).standard_normal((2000, 3))   # same xi for every ACF
    fig, axes = plt.subplots(len(acfs), 1, figsize=(11, 2.0 * len(acfs)), sharex=True)
    for ax, acf in zip(np.atleast_1d(axes), acfs):
        k = KL1D(acf, delta, (0, L), energy=energy, nu=nu)
        ax.plot(x, k.simulate(x, xi=xi, normalise=normalise), lw=1)
        ax.set_ylabel("G(x)")
        ax.set_ylim(-3.5, 3.5)
        ax.set_title(f"{acf}: {k.n_terms} terms, {k.energy_ratio():.1%} of the variance", fontsize=9)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[-1].set_xlabel("x (m)")
    fig.suptitle(f"Standard Gaussian realisations, delta = {delta:g} m (same random numbers for each ACF)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "kl_1d_realisations.png"), dpi=150)
    plt.close(fig)
    print("  saved kl_1d_realisations.png")


def plot_2d_slope_fields(acfs, args):
    dist, mean, cov, unit = layer_statistics(args.layer, args.param)
    if dist == "deterministic":
        raise SystemExit(f"{args.layer} {args.param} is deterministic; pick a random parameter")
    y_top = max(y for _, y in GROUND_SURFACE)
    xg = np.linspace(X_RANGE[0], X_RANGE[1], 251)
    yg = np.linspace(MODEL_BOTTOM, y_top, 76)
    X, Y = np.meshgrid(xg, yg)
    inside = MplPath(DOMAIN).contains_points(np.column_stack([X.ravel(), Y.ravel()]), radius=1e-6)
    xi = np.random.default_rng(args.seed).standard_normal((20000, 1))     # same xi for every ACF
    n = len(acfs)
    ncol = 2 if n > 1 else 1
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.5 * ncol, 3.2 * nrow), squeeze=False)
    vmin, vmax = to_marginal(np.array([-2.5, 2.5]), dist, mean, cov)
    print(f"  2-D fields of {args.layer} {args.param} ({dist}, mean {mean:g} {unit}, COV {cov:g}), "
          f"delta_h = {args.delta_h:g} m, delta_v = {args.delta_v:g} m:")
    for ax, acf in zip(axes.flat, acfs):
        kl = KL2D(acf, args.delta_h, args.delta_v, X_RANGE, (MODEL_BOTTOM, y_top), energy=args.energy, nu=args.nu)
        g = kl.simulate(X.ravel()[inside], Y.ravel()[inside], xi=xi, normalise=args.normalise_variance)[:, 0]
        field = np.full(X.size, np.nan)
        field[inside] = to_marginal(g, dist, mean, cov)
        im = ax.pcolormesh(X, Y, field.reshape(X.shape), cmap="viridis", vmin=vmin, vmax=vmax, shading="auto")
        ax.plot(*zip(*GROUND_SURFACE), "k-", lw=1.5)
        ax.set_aspect("equal")
        ax.set_title(f"{acf}: {kl.n_terms} terms, {kl.energy_ratio():.1%} of the variance", fontsize=9)
        fig.colorbar(im, ax=ax, label=f"{args.param} ({unit})", shrink=0.8)
        print(f"    {acf:<20s} {kl.n_terms:4d} terms ({kl.kx.n_terms} x-modes, {kl.ky.n_terms} y-modes available)")
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(f"{args.layer} {args.param} random field over the slope, delta_h = {args.delta_h:g} m, "
                 f"delta_v = {args.delta_v:g} m (same random numbers for each ACF)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "kl_2d_slope_fields.png"), dpi=150)
    plt.close(fig)
    print("  saved kl_2d_slope_fields.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acf", nargs="+", default=ACF_NAMES, choices=ACF_NAMES)
    ap.add_argument("--nu", type=float, default=2.5, help="Matern smoothness (whittle_matern only)")
    ap.add_argument("--delta-h", type=float, default=20.0, help="horizontal scale of fluctuation (m)")
    ap.add_argument("--delta-v", type=float, default=2.0, help="vertical scale of fluctuation (m)")
    ap.add_argument("--energy", type=float, default=0.95, help="share of the variance kept by the truncation")
    ap.add_argument("--layer", default="Layer 3", choices=list(PARAMS))
    ap.add_argument("--param", default="c", help="soil parameter for the 2-D example (c or phi)")
    ap.add_argument("--normalise-variance", action="store_true",
                    help="rescale the truncated fields to unit variance at every point")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--verify", action="store_true", help="check against exact eigenvalues and Monte Carlo")
    args = ap.parse_args()

    L = X_RANGE[1] - X_RANGE[0]
    print(f"KL expansion: ACFs {', '.join(args.acf)}")
    print(f"1-D examples on a {L:g} m line with delta = delta_h = {args.delta_h:g} m")
    for acf in args.acf:
        k = KL1D(acf, args.delta_h, (0, L), energy=args.energy, nu=args.nu)
        print(f"  {acf:<20s} {k.n_terms:3d} terms for {args.energy:.0%} of the variance "
              f"(first eigenvalue {k.eigenvalues[0]:.2f} m)")
    plot_acf_and_spectra(args.acf, args.delta_h, L, args.nu, args.energy)
    plot_eigenfunctions(args.acf, args.delta_h, L, args.nu)
    plot_1d_realisations(args.acf, args.delta_h, L, args.nu, args.energy, args.seed, args.normalise_variance)
    plot_2d_slope_fields(args.acf, args)
    if args.verify:
        verify(args.acf, args.delta_h, L, args.nu, args.energy, args.seed)


if __name__ == "__main__":
    main()
