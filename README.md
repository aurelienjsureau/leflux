# Le Flux — Primordial QGP Cosmology
> *The universe is a single, structured quantum fluid — and its dynamics, from the molecular cores of spiral galaxies to the hydrostatic equilibrium of clusters and the primordial CMB, emerge self-consistently from its non-barotropic, non-Newtonian rheology.*

A cosmological model in which the Big Bang is not a past event but an ongoing process. Space itself is identified with the continuously expanding primordial quark-gluon plasma (QGP), governed by the MIT Bag equation of state. There is no dark matter, no dark energy — these concepts become unnecessary.

---

## Results

### Galaxies — Three Independent Probes

EOS formula: `V²(r) = V²_bar(r) + V₀²·r²/(r_c²+r²)`

**SPARC (163 galaxies, Lelli et al. 2016) — HI/Hα rotation curves**

| Model | Median χ²r | Free params |
|-------|-----------|-------------|
| **Le Flux v10** | **0.535** | **1–2** |
| NFW | 1.47 | 2–3 |
| Burkert | 0.79 | 2–3 |
| MOND/RAR | ~1.2 | 1 |

With r_c physically bounded by R_HI → **median χ²r = 0.687 with 1 free parameter**

The coherence length r_c correlates with the HI disk radius R_HI (R²partial=0.076, p=4.3×10⁻³, N=106) — a prediction ΛCDM does not make.

**PHANGS (37/42 galaxies, 88%) — CO kinematics + S4G baryonic decomposition — *Addendum 5***

- High-resolution CO(2-1) rotation curves (Lang+2020, ~150 pc resolution)
- Stellar masses fixed *a priori* from S4G photometric decomposition (Salo+2015) — no free mass parameter
- 37/42 galaxies with good visual fits on CO(2-1) rotation curves — same EOS, no modification

**MaNGA DynPop (1225 galaxies, 87%) — blind predictive test — *Addendum 5***

- Total dynamical masses from JAM modelling (Zhu+2023, SDSS DR17)
- Parameters V₀=150 km/s, r_c=2 kpc from SPARC — not re-adjusted on MaNGA
- 87% of 1225 galaxies within ±0.3 dex — median residual 0.15 dex, scatter 0.14 dex
- NFW achieves 99.8% but fitted on its own training data — comparison is not equitable

### Galaxy Clusters — X-COP (13 clusters, Ettori+2018) — *Addendum 2*

Unified non-barotropic EOS (cluster limit):
```
dρ/dr = -ρ(dΦ/dr + dσ²_turb/dr) / σ²_turb
σ_turb(r) = k · σ_BCG · (1 + r/R200)^{-0.5}
```
- Median χ²r = **0.136** across all 13 clusters
- **13/13** clusters with χ²r < 3
- k anchored on BCG velocity dispersion σ_BCG (Zhang+2017), bounded by lattice QCD
- k median = 0.999, range [0.84, 1.21]

### CMB — Planck 2018 TT (83 data points) — *Addendum 3*

At CMB densities (ρ ~ 10⁻²⁹ g/cm³), B_eff(ρ) = B₀·(ρ/ρ₀)^γ → 0. The QGP substrate in its cold phase is fully compatible with the Planck 2018 TT power spectrum:

- **χ²r = 1.330** on 83 Planck 2018 TT data points
- Sound horizon r_s = 148.3 Mpc (Planck ~147 Mpc)
- Angular scale θ_s = 0.01063 rad (Planck 0.01040 rad, 2% agreement)
- First peak at ℓ ~ 220 ✓

### Bullet Cluster — Strong Gravitational Lensing — *Addendum 4*

- Mass distribution reconstructed without collisionless dark matter
- Self-interaction cross-section σ/m < 0.47 cm²/g — compatible with Markevitch+2002 constraint
- Density-dependent cross-section σ/m(ρ) = (σ/m)₀·(ρ/ρ_ref)^α with α=1 consistent with γ=1

---

## One substrate. One EOS. Five validations.

| Scale | Dataset | Result | Free params |
|-------|---------|--------|-------------|
| Galaxies | 163 SPARC (HI/Hα) | χ²r = 0.535 | 1–2 |
| Galaxies | 37/42 PHANGS (CO) | 88% good fits | 2 |
| Galaxies | 1225 MaNGA DynPop | 87% within ±0.3 dex | 0 (blind) |
| Clusters | 13 X-COP | χ²r = 0.136 | 2 |
| CMB | Planck 2018 TT | χ²r = 1.330 | Standard cosmology |
| Lensing | Bullet Cluster | σ/m < 0.47 cm²/g | Analytical |

No dark matter. No dark energy.

---

## Scripts

### `flux_complet.py`
SPARC rotation curve fitting.
```bash
# Download SPARC data: https://astroweb.cwru.edu/SPARC/
python3 flux_complet.py
```

### `flux_xcop.py`
X-COP galaxy cluster fitting — Addendum 2.
```bash
python3 flux_xcop.py
```
Reproduces: median χ²r = 0.136, 13/13 clusters with χ²r < 3.

### `flux_cmb.py`
CMB TT power spectrum — Addendum 3.
```bash
pip install camb
python3 leflux_cmb.py
```
Reproduces: χ²r = 1.330 on Planck 2018 TT.

**Requires:** numpy, scipy, matplotlib, camb

---

## Preprints

**Le Flux** (main paper + Addenda 1–5):  
DOI: [10.5281/zenodo.19219720](https://doi.org/10.5281/zenodo.19219720) *(all versions)*

---

## Open research directions

These are active areas of investigation, not flaws in the model:

- **Production mechanism**: How does the QGP substrate achieve the correct cold relic abundance from the QCD crossover? The sexaquark hypothesis (Farrar 2017, 2024) provides a candidate for the cold confined state — Non-thermal mechanisms are being explored.
- **Vortex propagation**: Seeding from BH spin to kpc scales — GRMHD simulation required to confirm the mechanism.
- **CMB predictive power**: Deriving H₀ and Ω_c from QCD fundamentals remains the next theoretical milestone.
- **γ measurement**: Direct Bayesian fit of γ on SPARC rotation curves to constrain the EOS exponent observationally — identified as the next decisive test.

---

## Author

Aurélien Sureau — May 2026 — Independent research.
