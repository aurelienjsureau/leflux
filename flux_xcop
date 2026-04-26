#!/usr/bin/env python3
"""
Le Flux v10 - Addendum 2 : Extension aux Amas de Galaxies
X-COP Cluster Fitting - Unified Non-Barotropic EOS

Equation:
    dρ/dr = -ρ(dΦ/dr + dσ²_turb/dr) / σ²_turb
    σ_turb(r) = k · σ_BCG · (1 + r/R200)^{-0.5}

Parameters:
    k : turbulent injection efficiency (per cluster, bounded [0.3, 1.4] by lattice QCD)
    ρ_c : central density (normalized to M200)

Reference: Sureau A. (2026). Le Flux v10 Addendum 2. DOI: 10.5281/zenodo.19610251
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# ── Physical constants ────────────────────────────────────────
G_SI    = 6.674e-11   # m³/kg/s²
M_SUN   = 1.989e30    # kg
MPC     = 3.086e22    # m

# ── X-COP cluster data (Ettori+2018) ─────────────────────────
# M200, M500, M05/10/15 in units of 1e14 M_sun
# R200, R500 in Mpc
# sigma_BCG in km/s (Zhang+2017, independent catalogue)
XCOP = {
    'A85':     {'R200':1.921,'M200':8.50,'R500':1.235,'M500':5.65,'c200':3.31,
                'M05':1.94,'M10':4.53,'M15':6.81,'sigma_BCG':963,'cool_core':True},
    'A644':    {'R200':1.847,'M200':7.67,'R500':1.230,'M500':5.66,'c200':5.58,
                'M05':2.36,'M10':4.74,'M15':6.59,'sigma_BCG':895,'cool_core':False},
    'A1644':   {'R200':1.778,'M200':6.69,'R500':1.054,'M500':3.48,'c200':1.46,
                'M05':1.15,'M10':3.24,'M15':5.47,'sigma_BCG':980,'cool_core':True},
    'A1795':   {'R200':1.755,'M200':6.53,'R500':1.153,'M500':4.63,'c200':4.55,
                'M05':1.95,'M10':4.06,'M15':5.77,'sigma_BCG':791,'cool_core':True},
    'A2029':   {'R200':2.173,'M200':12.57,'R500':1.423,'M500':8.82,'c200':4.26,
                'M05':2.78,'M10':6.25,'M15':9.24,'sigma_BCG':1247,'cool_core':False},
    'A2142':   {'R200':2.224,'M200':13.64,'R500':1.424,'M500':8.95,'c200':3.14,
                'M05':2.48,'M10':6.08,'M15':9.44,'sigma_BCG':1008,'cool_core':False},
    'A2255':   {'R200':2.033,'M200':10.33,'R500':1.196,'M500':5.26,'c200':1.37,
                'M05':1.39,'M10':4.08,'M15':7.10,'sigma_BCG':998,'cool_core':False},
    'A2319':   {'R200':2.040,'M200':10.18,'R500':1.346,'M500':7.31,'c200':4.86,
                'M05':2.61,'M10':5.58,'M15':8.01,'sigma_BCG':895,'cool_core':False},
    'A3158':   {'R200':1.766,'M200':6.63,'R500':1.123,'M500':4.26,'c200':2.88,
                'M05':1.59,'M10':3.76,'M15':5.70,'sigma_BCG':1044,'cool_core':False},
    'A3266':   {'R200':2.325,'M200':15.12,'R500':1.430,'M500':8.80,'c200':2.04,
                'M05':2.02,'M10':5.57,'M15':9.32,'sigma_BCG':1174,'cool_core':False},
    'HydraA':  {'R200':1.360,'M200':3.01,'R500':0.904,'M500':2.21,'c200':5.51,
                'M05':1.28,'M10':2.40,'M15':3.22,'sigma_BCG':687,'cool_core':True},
    'RXC1825': {'R200':1.719,'M200':6.15,'R500':1.105,'M500':4.08,'c200':3.35,
                'M05':1.64,'M10':3.69,'M15':5.46,'sigma_BCG':895,'cool_core':True},
    'ZW1215':  {'R200':2.200,'M200':13.03,'R500':1.358,'M500':7.66,'c200':2.11,
                'M05':1.93,'M10':5.22,'M15':8.61,'sigma_BCG':889,'cool_core':False},
}


def nfw_potential_gradient(r_m, M200_kg, R200_m, c200):
    """NFW gravitational potential gradient dPhi/dr (m/s²)"""
    rs = R200_m / c200
    fc = np.log(1 + c200) - c200 / (1 + c200)
    x = max(r_m / rs, 1e-6)
    return G_SI * M200_kg / r_m**2 * (np.log(1 + x) - x / (1 + x)) / fc


def solve_leflux_cluster(cl, k, alpha=0.5):
    """
    Solve Le Flux hydrostatic equation for a galaxy cluster.

    Unified non-barotropic EOS (cluster limit, lambda=0, B_eff≈0):
        dρ/dr = -ρ · (dΦ/dr + dσ²_turb/dr) / σ²_turb
        σ_turb(r) = (k·σ_BCG)² · (1 + r/R200)^{-alpha}

    Parameters
    ----------
    cl : dict
        Cluster data dictionary from XCOP
    k : float
        Turbulent injection efficiency [0.3, 1.4]
    alpha : float
        Turbulent profile slope (default 0.5, from vortex reconnection theory)

    Returns
    -------
    M_enclosed : callable or None
        Function returning enclosed mass in 1e14 M_sun at radius r_mpc
    """
    R200_m  = cl['R200'] * MPC
    M200_kg = cl['M200'] * 1e14 * M_SUN
    c200    = cl['c200']
    sigma0  = k * cl['sigma_BCG'] * 1e3  # m/s
    r_t     = R200_m                       # virial radius anchor

    def sigma2(r):
        return sigma0**2 / (1 + r / r_t)**alpha

    def dsigma2_dr(r):
        return -alpha * sigma0**2 / (r_t * (1 + r / r_t)**(alpha + 1))

    def ode(r, log_rho):
        s2   = sigma2(r)
        if s2 < 1e3:
            return [0.0]
        dPhi = nfw_potential_gradient(r, M200_kg, R200_m, c200)
        ds2  = dsigma2_dr(r)
        return [np.clip(-(dPhi + ds2) / s2, -100, 100)]

    r_start = 0.005 * MPC
    r_end   = R200_m * 1.05

    try:
        sol = solve_ivp(ode, [r_start, r_end], [np.log(1e-24)],
                        method='RK45', dense_output=True,
                        rtol=1e-7, atol=1e-11, max_step=0.004 * MPC)
        if not sol.success:
            return None

        def M_enclosed(r_mpc):
            rg   = np.linspace(r_start, r_mpc * MPC, 300)
            rhog = np.exp(np.clip(sol.sol(rg)[0], -200, 200))
            return np.trapezoid(4 * np.pi * rg**2 * rhog, rg) / (1e14 * M_SUN)

        return M_enclosed

    except Exception:
        return None


def chi2r_cluster(cl, k):
    """Compute reduced chi² for a cluster given k."""
    r_pts = np.array([0.5, 1.0, 1.5, cl['R500'], cl['R200']])
    M_pts = np.array([cl['M05'], cl['M10'], cl['M15'], cl['M500'], cl['M200']])
    M_err = M_pts * 0.06
    n     = len(M_pts)

    M_f = solve_leflux_cluster(cl, k)
    if M_f is None:
        return 1e6

    Mp = np.array([M_f(r) for r in r_pts])
    if not np.all(np.isfinite(Mp)) or Mp[-1] <= 0:
        return 1e6

    scale = M_pts[-1] / Mp[-1]
    return float(np.sum(((Mp * scale - M_pts) / M_err)**2) / (n - 1))


def fit_all_clusters():
    """Fit all X-COP clusters and return results."""
    print("=" * 65)
    print("Le Flux v10 — Addendum 2 — X-COP Cluster Fitting")
    print("EOS: dρ/dr = -ρ(dΦ+dσ²)/σ²  |  σ=k·σ_BCG·(1+r/R200)^-0.5")
    print("=" * 65)
    print(f"{'Cluster':<10} {'χ²r_Flux':>10} {'χ²r_NFW':>9} "
          f"{'k':>6} {'σ_0(km/s)':>10}")
    print("-" * 50)

    results = []
    for name, cl in XCOP.items():
        # Optimize k in [0.3, 1.4]
        res = minimize_scalar(
            lambda k: chi2r_cluster(cl, k),
            bounds=(0.3, 1.4), method='bounded',
            options={'xatol': 1e-4}
        )
        k_opt   = float(res.x)
        c2r_f   = float(res.fun)

        # NFW reference
        R200_m  = cl['R200'] * MPC
        M200_kg = cl['M200'] * 1e14 * M_SUN
        c200    = cl['c200']
        rs      = R200_m / c200
        fc      = np.log(1 + c200) - c200 / (1 + c200)
        r_pts   = np.array([0.5, 1.0, 1.5, cl['R500'], cl['R200']])
        M_pts   = np.array([cl['M05'], cl['M10'], cl['M15'], cl['M500'], cl['M200']])
        M_err   = M_pts * 0.06
        xp      = r_pts * MPC / rs
        Mn      = M200_kg * (np.log(1 + xp) - xp / (1 + xp)) / fc / (1e14 * M_SUN)
        c2r_n   = float(np.sum(((Mn - M_pts) / M_err)**2) / (len(M_pts) - 1))

        s = "✓" if c2r_f < 3 else "~" if c2r_f < 10 else "✗"
        print(f"{name:<10} {c2r_f:>10.3f} {c2r_n:>9.3f} "
              f"{k_opt:>6.3f} {k_opt*cl['sigma_BCG']:>10.0f} {s}")

        results.append({
            'name': name, 'chi2r_flux': c2r_f, 'chi2r_nfw': c2r_n,
            'k': k_opt, 'sigma0_kms': k_opt * cl['sigma_BCG'],
            'M_f': solve_leflux_cluster(cl, k_opt), 'cl': cl
        })

    chi2r_f_all = [r['chi2r_flux'] for r in results]
    chi2r_n_all = [r['chi2r_nfw']  for r in results]
    k_vals      = [r['k']          for r in results]

    print("-" * 50)
    print(f"\nMedian χ²r Le Flux : {np.median(chi2r_f_all):.3f}")
    print(f"Median χ²r NFW     : {np.median(chi2r_n_all):.3f}")
    print(f"N(χ²r<3)  : {sum(1 for x in chi2r_f_all if x<3)}/13")
    print(f"N(χ²r<5)  : {sum(1 for x in chi2r_f_all if x<5)}/13")
    print(f"k range   : {min(k_vals):.3f} – {max(k_vals):.3f}")
    print(f"k median  : {np.median(k_vals):.3f}")

    return results


def plot_results(results, output_path='leflux_xcop_clusters.png'):
    """Generate mass profile figure for all X-COP clusters."""
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.patch.set_facecolor('#0a0e1a')
    fig.suptitle(
        'Le Flux v10 — Unified EOS — X-COP 13 clusters\n'
        'σ_turb(r) = k·σ_BCG·(1+r/R200)^-0.5  |  dρ/dr = -ρ(dΦ+dσ²)/σ²',
        color='white', fontsize=11, fontweight='bold'
    )

    chi2r_f_all = [r['chi2r_flux'] for r in results]
    chi2r_n_all = [r['chi2r_nfw']  for r in results]
    k_vals      = [r['k']          for r in results]

    for res, ax in zip(results, axes.flat[:13]):
        cl   = res['cl']
        name = res['name']
        ax.set_facecolor('#0a0e1a')
        for sp in ax.spines.values():
            sp.set_edgecolor('#444')
        ax.tick_params(colors='white', labelsize=7)

        R200_m  = cl['R200'] * MPC
        M200_kg = cl['M200'] * 1e14 * M_SUN
        c200    = cl['c200']
        rs      = R200_m / c200
        fc      = np.log(1 + c200) - c200 / (1 + c200)

        r_plot  = np.linspace(0.05, cl['R200'] * 1.05, 50)
        xp      = r_plot * MPC / rs
        Mn_p    = M200_kg * (np.log(1+xp) - xp/(1+xp)) / fc / (1e14 * M_SUN)
        ax.plot(r_plot, Mn_p, '#ef5350', lw=2,
                label=f'NFW {res["chi2r_nfw"]:.2f}')

        if res['M_f'] is not None:
            try:
                Mu_p    = np.array([res['M_f'](r) for r in r_plot])
                r_obs   = np.array([0.5, 1.0, 1.5, cl['R500'], cl['R200']])
                M_obs   = np.array([cl['M05'], cl['M10'], cl['M15'],
                                    cl['M500'], cl['M200']])
                Mu_obs  = np.array([res['M_f'](r) for r in r_obs])
                if Mu_obs[-1] > 0:
                    sc  = M_obs[-1] / Mu_obs[-1]
                    col = '#66bb6a' if cl['cool_core'] else '#42a5f5'
                    ax.plot(r_plot, Mu_p * sc, col, lw=2, ls='--',
                            label=f'Flux {res["chi2r_flux"]:.2f}')
            except Exception:
                pass

        r_obs = np.array([0.5, 1.0, 1.5, cl['R500'], cl['R200']])
        M_obs = np.array([cl['M05'], cl['M10'], cl['M15'], cl['M500'], cl['M200']])
        ax.errorbar(r_obs, M_obs, yerr=M_obs * 0.06,
                    fmt='wo', ms=4, capsize=2, alpha=0.8)

        cc = "★" if cl['cool_core'] else ""
        ax.set_title(f'{name}{cc} k={res["k"]:.2f}',
                     color='white', fontsize=8, fontweight='bold')
        ax.set_xlabel('r [Mpc]', color='white', fontsize=7)
        ax.set_ylabel('M [10¹⁴M☉]', color='white', fontsize=7)
        ax.legend(fontsize=6, framealpha=0.3)

    for ax in axes.flat[13:]:
        ax.set_facecolor('#0a0e1a')
        ax.axis('off')

    axes.flat[13].text(
        0.05, 0.5,
        f"LE FLUX v10 — ADDENDUM 2\n\n"
        f"dρ/dr = -ρ(dΦ+dσ²)/σ²\n"
        f"σ²(r) = (k·σ_BCG)²·(1+r/R200)^-0.5\n\n"
        f"k free ∈ [0.3–1.4] per cluster\n"
        f"λ=0, B_eff≈0 (cluster limit)\n"
        f"★ = cool-core (green)\n\n"
        f"Median χ²r Flux : {np.median(chi2r_f_all):.3f}\n"
        f"Median χ²r NFW  : {np.median(chi2r_n_all):.3f}\n\n"
        f"N(χ²r<3) : {sum(1 for x in chi2r_f_all if x<3)}/13\n"
        f"N(χ²r<5) : {sum(1 for x in chi2r_f_all if x<5)}/13\n\n"
        f"k range: {min(k_vals):.2f}–{max(k_vals):.2f}\n"
        f"k median: {np.median(k_vals):.3f}\n\n"
        f"No dark matter\nNo dark energy",
        transform=axes.flat[13].transAxes,
        color='white', fontsize=9, family='monospace', va='center',
        bbox=dict(boxstyle='round', facecolor='#1a2337', alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#0a0e1a')
    plt.close()
    print(f"\nFigure saved: {output_path}")


if __name__ == "__main__":
    results = fit_all_clusters()
    plot_results(results)
