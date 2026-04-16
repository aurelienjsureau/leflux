#!/usr/bin/env python3
"""
LE FLUX v10 — ADDENDUM 1
========================
The Galactic Flux Vortex Coherence Length Scales with HI Disk Radius

Reproduces all results from Addendum 1 (Sureau 2026):
  - Correlation analysis: rc vs baryonic observables (SBdisk, SBeff, Rdisk, RHI)
  - Partial correlations controlling for L[3.6] and Vflat
  - Physics-guided bounded model: rc in [RHI/C_low, C_high*RHI]
  - Grid search for optimal bounds
  - All figures

Requirements:
    pip install numpy scipy matplotlib

Data:
    - SPARC_Lelli2016c.mrt   : from http://astroweb.cwru.edu/SPARC/
    - *_rotmod.dat files      : from http://astroweb.cwru.edu/SPARC/ (Rotmod_LTG.zip)

Usage:
    python leflux_addendum1.py --sparc SPARC_Lelli2016c.mrt --rotdir ./rotcurves/

Author: Aurélien Sureau (2026)
DOI:    10.5281/zenodo.19609775
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize_scalar
from scipy.stats import linregress, pearsonr
import argparse, os, warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — PHYSICS
# ══════════════════════════════════════════════════════════════════════

def V_EOS(r, Vbar, V0r, rc):
    """
    Le Flux EOS MIT Bag formula (Sureau 2026, Eq. 1).
    Derived from Euler equilibrium + MIT Bag equation of state P = rho/3 - B.

    V²_total(r) = V²_bar(r) + V0² * r² / (rc² + r²)
    V0 = V0r * max(V_bar)

    Parameters
    ----------
    r    : array, galactocentric radius [kpc]
    Vbar : array, baryonic velocity [km/s]
    V0r  : float, dimensionless vortex amplitude (free parameter)
    rc   : float, vortex coherence length [kpc] (free or bounded)
    """
    V0 = V0r * np.max(Vbar)
    return np.sqrt(np.maximum(Vbar**2 + V0**2 * r**2 / (rc**2 + r**2), 0))


def chi2r(Vm, Vo, eV):
    """Reduced chi-squared."""
    return np.sum(((Vm - Vo) / eV)**2) / max(len(Vo) - 1, 1)


# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA I/O
# ══════════════════════════════════════════════════════════════════════

def read_rotcurve(name, rotdir='.', Ydisk=0.5):
    """
    Read {name}_rotmod.dat from SPARC.
    Ydisk=0.5 follows Schombert+2014 recommendation for Spitzer 3.6um.

    Returns dict with r, Vobs, errV, Vbar or None if file not found.
    """
    fname = os.path.join(rotdir, f'{name}_rotmod.dat')
    if not os.path.exists(fname):
        return None
    data = []
    with open(fname) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            try:
                v = [float(x) for x in line.split()]
                if len(v) >= 6:
                    data.append(v[:6])
            except:
                continue
    if not data:
        return None
    d = np.array(data)
    mask = (d[:, 2] > 0) & (d[:, 1] > 0) & (d[:, 0] > 0)
    d = d[mask]
    if len(d) < 5:
        return None
    r, Vobs, errV = d[:, 0], d[:, 1], d[:, 2]
    Vgas, Vdisk, Vbul = d[:, 3], d[:, 4], d[:, 5]
    # Baryonic velocity with Ydisk scaling
    V2b = np.abs(Vgas)*Vgas + Ydisk*(np.abs(Vdisk)*Vdisk + np.abs(Vbul)*Vbul)
    return {
        'r': r, 'Vobs': Vobs, 'errV': errV,
        'Vbar': np.sqrt(np.maximum(V2b, 0))
    }


def parse_sparc(filepath):
    """
    Parse SPARC_Lelli2016c.mrt.
    Returns dict: name -> {T, L36, SBdisk, SBeff, Rdisk, RHI, Vflat}
    """
    galaxies = {}
    with open(filepath) as f:
        lines = f.readlines()
    for line in lines[98:]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 18:
            continue
        try:
            name    = parts[0]
            T       = int(parts[1])
            L36     = float(parts[7])
            SBeff   = float(parts[8])  if float(parts[8])  > 0 else np.nan
            Rdisk   = float(parts[11]) if float(parts[11]) > 0 else np.nan
            SBdisk  = float(parts[12]) if float(parts[12]) > 0 else np.nan
            RHI     = float(parts[14]) if float(parts[14]) > 0 else np.nan
            Vflat   = float(parts[15])
            Q       = int(parts[17])
            if Q in [1, 2] and Vflat > 5 and L36 > 0:
                galaxies[name] = {
                    'T': T, 'L36': L36, 'SBeff': SBeff,
                    'Rdisk': Rdisk, 'SBdisk': SBdisk,
                    'RHI': RHI, 'Vflat': Vflat
                }
        except:
            continue
    print(f"SPARC parsed: {len(galaxies)} galaxies (Q=1,2)")
    return galaxies


# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — FITTING
# ══════════════════════════════════════════════════════════════════════

def fit_free(gd):
    """
    Model A: Free fit — both V0r and rc optimized per galaxy.
    Returns (V0r, rc, chi2r).
    """
    r, Vobs, errV, Vbar = gd['r'], gd['Vobs'], gd['errV'], gd['Vbar']

    def cost(p):
        return chi2r(V_EOS(r, Vbar, abs(p[0]), abs(p[1])), Vobs, errV)

    res = differential_evolution(
        cost, bounds=[(0.1, 4.0), (0.1, r[-1] * 1.5)],
        seed=42, maxiter=500, popsize=10, tol=1e-7
    )
    return abs(res.x[0]), abs(res.x[1]), res.fun


def fit_bounded(gd, RHI, C_low=5.0, C_high=5.0):
    """
    Model D: Physics-guided bounds on rc.

    rc constrained to [RHI/C_low, C_high*RHI].

    Physical justification:
      - Lower bound (RHI/C_low): vortex coherence length cannot be much
        smaller than the gaseous disk — insufficient baryonic material
        to organize Flux motion below this scale.
      - Upper bound (C_high*RHI): beyond C_high*RHI, baryonic gravitational
        potential weakens and cannot confine coherent vortex motion.

    Only V0r is truly free. rc explores the physically allowed range.
    Returns (V0r, rc, chi2r, within_bounds).
    """
    r, Vobs, errV, Vbar = gd['r'], gd['Vobs'], gd['errV'], gd['Vbar']
    rc_min = max(RHI / C_low, 0.1)
    rc_max = C_high * RHI
    if rc_min >= rc_max:
        rc_min, rc_max = 0.1, r[-1] * 5

    def cost(p):
        V0r, rc = abs(p[0]), abs(p[1])
        penalty = 0.0
        if rc < rc_min:
            penalty = 50.0 * ((rc_min - rc) / rc_min)**2
        elif rc > rc_max:
            penalty = 50.0 * ((rc - rc_max) / rc_max)**2
        return chi2r(V_EOS(r, Vbar, V0r, rc), Vobs, errV) + penalty

    res = differential_evolution(
        cost, bounds=[(0.1, 4.0), (rc_min * 0.8, rc_max * 1.2)],
        seed=42, maxiter=500, popsize=15, tol=1e-7
    )
    V0r = abs(res.x[0])
    rc  = abs(res.x[1])
    within = rc_min <= rc <= rc_max
    return V0r, rc, res.fun, within


# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def partial_correlation(x, y, z):
    """
    Partial correlation of x and y controlling for z.
    Computed via residual regression.
    Returns (r_partial, p_value, R2_partial).
    """
    # Residuals of x ~ z
    slope_xz, int_xz, *_ = linregress(z, x)
    res_x = x - (int_xz + slope_xz * z)
    # Residuals of y ~ z
    slope_yz, int_yz, *_ = linregress(z, y)
    res_y = y - (int_yz + slope_yz * z)
    # Correlation of residuals
    r, p = pearsonr(res_x, res_y)
    return r, p, r**2


def partial_correlation_2(x, y, z1, z2):
    """Partial correlation of x and y controlling for z1 AND z2."""
    from numpy.linalg import lstsq
    Z = np.column_stack([np.ones(len(z1)), z1, z2])
    res_x = x - Z @ lstsq(Z, x, rcond=None)[0]
    res_y = y - Z @ lstsq(Z, y, rcond=None)[0]
    r, p = pearsonr(res_x, res_y)
    return r, p, r**2


def run_correlation_analysis(results):
    """
    Test correlations of log(rc) vs log(SBdisk), log(SBeff),
    log(Rdisk), log(RHI), controlling for log(L36) and log(Vflat).
    """
    log_rc   = np.array([np.log10(r['rc'])     for r in results])
    log_L36  = np.array([np.log10(r['L36'])    for r in results])
    log_Vf   = np.array([np.log10(r['Vflat'])  for r in results])

    observables = {}
    for key in ['SBdisk', 'SBeff', 'Rdisk', 'RHI']:
        vals = np.array([r[key] for r in results])
        mask = np.isfinite(vals) & (vals > 0)
        observables[key] = (np.log10(vals[mask]), log_rc[mask],
                            log_L36[mask], log_Vf[mask], np.sum(mask))

    print(f"\n{'='*72}")
    print(f"CORRELATION ANALYSIS: log(rc) vs baryonic observables")
    print(f"{'='*72}")
    print(f"{'Observable':<12} {'N':>4} {'γ':>7} {'±err':>7} {'R²':>7} {'p':>10}  {'Partial R²':>10} {'p_partial':>10}")
    print("-"*72)

    corr_results = {}
    for key, (log_obs, log_rc_m, log_L_m, log_Vf_m, N) in observables.items():
        # Raw correlation
        slope, intercept, r, p, se = linregress(log_obs, log_rc_m)
        # Partial correlation (control L36)
        r_p, p_p, R2_p = partial_correlation(log_obs, log_rc_m, log_L_m)
        print(f"{key:<12} {N:>4} {slope:>7.3f} {se:>7.3f} {r**2:>7.3f} {p:>10.2e}  {R2_p:>10.3f} {p_p:>10.2e}")
        corr_results[key] = {
            'gamma': slope, 'se': se, 'R2': r**2, 'p': p,
            'R2_partial': R2_p, 'p_partial': p_p, 'N': N
        }

    # Double control: RHI | L36 + Vflat
    log_obs, log_rc_m, log_L_m, log_Vf_m, N = observables['RHI']
    # Align Vflat
    r_pp, p_pp, R2_pp = partial_correlation_2(log_obs, log_rc_m, log_L_m, log_Vf_m)
    print(f"\nRHI | L[3.6]+Vflat  N={N}  R²={R2_pp:.3f}  p={p_pp:.2e}  (double control)")

    return corr_results


# ══════════════════════════════════════════════════════════════════════
# SECTION 5 — GRID SEARCH
# ══════════════════════════════════════════════════════════════════════

def run_grid_search(good_results, rotdir,
                    C_low_grid=[3., 5., 7., 10.],
                    C_high_grid=[2., 3., 4., 5., 6., 8.]):
    """
    Grid search for optimal physical bounds (C_low, C_high).
    Returns best (C_low, C_high) and corresponding median chi2r.
    """
    print(f"\n{'='*72}")
    print(f"GRID SEARCH: optimal physical bounds on rc")
    print(f"Testing {len(C_low_grid)} x {len(C_high_grid)} = {len(C_low_grid)*len(C_high_grid)} combinations")
    print(f"{'='*72}")
    print(f"{'C_low':>6} {'C_high':>7} {'N':>4} {'median χ²r':>11} {'%within':>8}")
    print("-"*40)

    best_med = 1e10
    best_combo = None
    best_fits = None

    for C_low in C_low_grid:
        for C_high in C_high_grid:
            fits = []
            for r in good_results:
                gd = read_rotcurve(r['name'], rotdir=rotdir)
                if gd is None or np.isnan(r['RHI']):
                    continue
                try:
                    V0r, rc, c2r, within = fit_bounded(gd, r['RHI'], C_low, C_high)
                    fits.append({'chi2r': c2r, 'within': within})
                except:
                    continue
            if len(fits) < 10:
                continue
            med = np.median([f['chi2r'] for f in fits])
            pct = 100 * np.mean([f['within'] for f in fits])
            print(f"{C_low:>6.1f} {C_high:>7.1f} {len(fits):>4} {med:>11.3f} {pct:>7.1f}%")
            if med < best_med:
                best_med = med
                best_combo = (C_low, C_high)
                best_fits = fits

    print(f"\n★ Best: C_low={best_combo[0]}, C_high={best_combo[1]}, median χ²r={best_med:.3f}")
    print(f"  vs NFW=1.47 | Burkert=0.79 | MOND~1.20 | Le Flux free=0.535")
    return best_combo, best_med, best_fits


# ══════════════════════════════════════════════════════════════════════
# SECTION 6 — FIGURES
# ══════════════════════════════════════════════════════════════════════

def plot_partial_correlations(results, outdir='.'):
    """Figure 1: 2x2 partial correlation panels (RHI and SBdisk)."""
    log_rc   = np.array([np.log10(r['rc'])    for r in results])
    log_L36  = np.array([np.log10(r['L36'])   for r in results])
    colors   = np.array([r['T']               for r in results])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor('#0d1117')

    panels = [
        ('RHI',    'log RHI (kpc)',  True,  False),
        ('RHI',    'partial log RHI | L[3.6]', True, True),
        ('SBdisk', 'log Σ₀ (L☉/pc²)', False, False),
        ('SBdisk', 'partial log Σ₀ | L[3.6]', False, True),
    ]

    for ax, (key, xlabel, is_rhi, is_partial) in zip(axes.flat, panels):
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='white', labelsize=9)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for sp in ax.spines.values():
            sp.set_edgecolor('#30363d')
        ax.grid(True, alpha=0.1, color='white')

        vals = np.array([r[key] for r in results])
        mask = np.isfinite(vals) & (vals > 0)
        lv = np.log10(vals[mask])
        lr = log_rc[mask]
        ll = log_L36[mask]
        col = colors[mask]

        if is_partial:
            slope_vl, int_vl, *_ = linregress(ll, lv)
            lv = lv - (int_vl + slope_vl * ll)
            slope_rl, int_rl, *_ = linregress(ll, lr)
            lr = lr - (int_rl + slope_rl * ll)

        sc = ax.scatter(lv, lr, c=col, cmap='plasma', s=20, alpha=0.7)
        slope, intercept, r_val, p_val, se = linregress(lv, lr)
        xfit = np.linspace(lv.min(), lv.max(), 100)
        ax.plot(xfit, intercept + slope * xfit,
                color='#3fb950' if is_rhi else '#ef5350', lw=2)

        label = f'γ={slope:.2f}±{se:.2f}\nR²={r_val**2:.3f}  p={p_val:.2e}\nN={mask.sum()}'
        ax.text(0.97, 0.05, label, transform=ax.transAxes,
                fontsize=8, color='white', ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='#0d1117', alpha=0.7))
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel('log rc (kpc)' if not is_partial else 'residual log rc', fontsize=9)
        title = ('RHI' if is_rhi else 'Σ₀') + (' (partial)' if is_partial else ' (raw)')
        ax.set_title(title, fontsize=10, color='white')

    fig.suptitle('LE FLUX Addendum 1 — Partial correlations rc vs RHI and Σ₀\n'
                 'Color = Hubble type T | Sureau 2026',
                 color='white', fontsize=12)
    plt.tight_layout()
    outpath = os.path.join(outdir, 'flux_partial_corr.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"Saved: {outpath}")


def plot_chi2r_comparison(c2_free, c2_bounded, best_combo, outdir='.'):
    """Figure 2: chi2r distribution comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0d1117')

    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='white', labelsize=10)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for sp in ax.spines.values():
            sp.set_edgecolor('#30363d')
        ax.grid(True, alpha=0.1, color='white')

    bins = np.linspace(0, 8, 40)

    # Left: distributions
    ax = axes[0]
    ax.hist(np.clip(c2_free, 0, 8), bins=bins, alpha=0.85, color='#3fb950',
            label=f'Free fit (2 params)  med={np.median(c2_free):.2f}')
    ax.hist(np.clip(c2_bounded, 0, 8), bins=bins, alpha=0.65, color='#42a5f5',
            label=f'Bounded (1 param)    med={np.median(c2_bounded):.2f}')
    ax.axvline(1.47, color='#ef5350', lw=2, linestyle=':', alpha=0.8, label='NFW 1.47')
    ax.axvline(0.79, color='#ffa726', lw=2, linestyle=':', alpha=0.8, label='Burkert 0.79')
    ax.axvline(np.median(c2_free),    color='#3fb950', lw=2, linestyle='--')
    ax.axvline(np.median(c2_bounded), color='#42a5f5', lw=2, linestyle='--')
    ax.set_xlabel('Reduced χ²r', fontsize=11)
    ax.set_ylabel('Number of galaxies', fontsize=11)
    ax.legend(fontsize=9, framealpha=0.3, facecolor='#161b22', labelcolor='white')
    ax.set_title('χ²r distributions — SPARC galaxies', fontsize=11, color='white')

    # Right: bar chart
    ax = axes[1]
    labels = ['NFW\n(2-3 params\n+ free Ydisk)',
              'MOND\n(1 param\nYdisk=0.5)',
              'Burkert\n(2 params\n+ free Ydisk)',
              f'Le Flux\nbounded\n(1 param)',
              'Le Flux\nfree fit\n(2 params)']
    vals = [1.47, 1.20, 0.79, np.median(c2_bounded), np.median(c2_free)]
    cols = ['#ef5350', '#ab47bc', '#ffa726', '#42a5f5', '#3fb950']
    bars = ax.barh(labels, vals, color=cols, alpha=0.88, height=0.55)
    ax.axvline(1.0, color='white', lw=1.5, linestyle=':', alpha=0.5)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', ha='left',
                color='white', fontsize=10, fontweight='bold')
    ax.set_xlabel('Median χ²r (lower = better)', fontsize=11)
    ax.set_title(f'C_low={best_combo[0]}, C_high={best_combo[1]}\n'
                 f'† free Ydisk not granted to Le Flux', fontsize=10, color='white')

    fig.suptitle('LE FLUX v10 Addendum 1 — Model comparison on SPARC galaxies\n'
                 'Le Flux: Ydisk=0.5 fixed (Schombert+2014)  '
                 '| NFW & Burkert: free Ydisk (Li+2020)\n'
                 'Sureau 2026 | DOI: 10.5281/zenodo.19609775',
                 color='white', fontsize=11)
    plt.tight_layout()
    outpath = os.path.join(outdir, 'flux_chi2r_addendum1.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f"Saved: {outpath}")


# ══════════════════════════════════════════════════════════════════════
# SECTION 7 — MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Le Flux Addendum 1 analysis')
    parser.add_argument('--sparc',  default='SPARC_Lelli2016c.mrt',
                        help='Path to SPARC MRT file')
    parser.add_argument('--rotdir', default='.',
                        help='Directory containing *_rotmod.dat files')
    parser.add_argument('--outdir', default='.',
                        help='Output directory for figures')
    parser.add_argument('--skip-grid', action='store_true',
                        help='Skip grid search (use C_low=5, C_high=5)')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ── Parse SPARC ──
    sparc = parse_sparc(args.sparc)

    # ── Fit free (Model A) ──
    print(f"\n{'='*72}")
    print(f"STEP 1: Free fit (Model A) — V0r and rc both free")
    print(f"{'='*72}")
    results = []
    names = list(sparc.keys())
    for i, name in enumerate(names):
        gd = read_rotcurve(name, rotdir=args.rotdir)
        if gd is None:
            continue
        try:
            V0r, rc, c2r = fit_free(gd)
            meta = sparc[name]
            results.append({
                'name': name, 'V0r': V0r, 'rc': rc, 'chi2r_A': c2r,
                'T': meta['T'], 'L36': meta['L36'],
                'SBdisk': meta['SBdisk'], 'SBeff': meta['SBeff'],
                'Rdisk': meta['Rdisk'], 'RHI': meta['RHI'],
                'Vflat': meta['Vflat'], 'gd': gd
            })
        except:
            continue
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(names)}...")

    good = [r for r in results if r['chi2r_A'] < 5]
    c2_free = np.array([r['chi2r_A'] for r in good])
    print(f"Quality cut χ²r<5: {len(good)} galaxies")
    print(f"Median χ²r (free): {np.median(c2_free):.3f}")

    # ── Correlation analysis ──
    print(f"\nSTEP 2: Correlation analysis")
    good_rhi = [r for r in good if np.isfinite(r['RHI'])]
    corr = run_correlation_analysis(good_rhi)

    # ── Grid search ──
    print(f"\nSTEP 3: Physics-guided bounds")
    if args.skip_grid:
        best_combo = (5.0, 5.0)
        print(f"Skipping grid search, using C_low=5, C_high=5")
        fits_bounded = []
        for r in good_rhi:
            try:
                V0r, rc, c2r, within = fit_bounded(r['gd'], r['RHI'], 5.0, 5.0)
                fits_bounded.append({'chi2r': c2r, 'within': within})
            except:
                continue
        best_med = np.median([f['chi2r'] for f in fits_bounded])
    else:
        best_combo, best_med, fits_bounded = run_grid_search(
            good_rhi, args.rotdir
        )

    c2_bounded = np.array([f['chi2r'] for f in fits_bounded])
    print(f"\n{'='*72}")
    print(f"FINAL RESULTS")
    print(f"{'='*72}")
    print(f"Model A (free, 2 params)       : median χ²r = {np.median(c2_free):.3f}")
    print(f"Model D (bounded, 1 param)     : median χ²r = {best_med:.3f}")
    print(f"NFW reference (Li+2020)        : median χ²r = 1.47")
    print(f"Burkert reference (Li+2020)    : median χ²r = 0.79")
    print(f"MOND reference                 : median χ²r ~ 1.20")
    print(f"Note: NFW and Burkert use free Ydisk (extra parameter not granted to Le Flux)")

    # ── Figures ──
    print(f"\nSTEP 4: Generating figures")
    plot_partial_correlations(good_rhi, outdir=args.outdir)
    plot_chi2r_comparison(c2_free, c2_bounded, best_combo, outdir=args.outdir)
    print(f"\nDone. All outputs in: {args.outdir}")


if __name__ == '__main__':
    main()
