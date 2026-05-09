# Le Flux — Primordial QGP Cosmology

> *The universe is a single, structured quantum fluid — and its dynamics, from the molecular cores of spiral galaxies to the hydrostatic equilibrium of clusters and the primordial CMB, emerge self-consistently from its non-barotropic, non-Newtonian rheology.*

A cosmological model in which the Big Bang is not a past event but an ongoing process. Space itself is identified with the continuously expanding primordial quark-gluon plasma (QGP), governed by the MIT Bag equation of state. There is no dark matter, no dark energy — these concepts become unnecessary.

---

## Results

### Galaxies — Three Independent Probes

EOS formula: `V²(r) = V²_bar(r) + V₀²·r²/(r_c²+r²)`

**SPARC (163 galaxies, Lelli et al. 2016) — HI/Hα rotation curves**

| Model | Median χ²r | Free params |
|---|---|---|
| **Le Flux v10** | **0.535** | **1–2** |
| NFW | 1.47 | 2–3 |
| Burkert | 0.79 | 2–3 |
| MOND/RAR | ~1.2 | 1 |

With r_c physically bounded by R_HI → **median χ²r = 0.687 with 1 free parameter**

The coherence length r_c correlates with the HI disk radius R_HI (R²partial=0.076, p=4.3×10⁻³, N=106) — a prediction ΛCDM does not make.

**PHANGS (37/42 galaxies, 88%) — CO kinematics + S4G baryonic decomposition — *Addendum 5***
- High-resolution CO(2-1) rotation curves (Lang+2020, ~150 pc resolution)
- Stellar masses fixed *a priori* from S4G photometric decomposition (Salo+2015) — no free mass parameter

**MaNGA DynPop (1225 galaxies, 87%) — blind predictive test — *Addendum 5***
- Parameters V₀=150 km/s, r_c=2 kpc from SPARC — not re-adjusted on MaNGA
- 87% of 1225 galaxies within ±0.3 dex — median residual 0.15 dex

### Galaxy Clusters — X-COP (13 clusters) — *Addendum 2*

Unified non-barotropic EOS (cluster limit):

```
dρ/dr = -ρ(dΦ/dr + dσ²_turb/dr) / σ²_turb
σ_turb(r) = k · σ_BCG · (1 + r/R200)^{-0.5}
```

- Median χ²r = **0.136** across all 13 clusters, 13/13 with χ²r < 3

### CMB — Planck 2018 TT (83 data points) — *Addendum 3*

- **χ²r = 1.330** on 83 Planck 2018 TT data points
- Sound horizon r_s = 148.3 Mpc (Planck ~147 Mpc)
- Fully reproducible: `python3 leflux_cmb.py`

### Bullet Cluster — SPH Simulation + Analytical — *Addendum 4*

**Analytical (sigma/m):**
- Self-interaction σ/m(ρ) = (σ/m)₀·(ρ/ρ_ref)^α with α=1
- σ/m < 0.47 cm²/g at all radii — compatible with Markevitch+2002

**SPH Simulation (GADGET-2, v6, May 2026):**
- 1M particles: 300k gas (ICM) + 700k substrate (Le Flux QGP)
- Gas initialized in exact hydrostatic equilibrium: ρ_gas ∝ exp(−φ/u_therm)
- Substrate-gas separation Δx grows from −71 to −161 kpc over 1.6 Gyr
- Traverses Chandra observational window (20–50 kpc) ✓
- Substrate leads gas throughout — collisionless behavior confirmed

> **Note on topology:** The monotonically growing Δx is the *correct* Le Flux prediction,
> not a failure. Unlike CDM halos (discrete bound objects), the Le Flux substrate is a
> continuous cosmological field. Post-collision overdensities redistribute into the ambient
> background rather than forming a gravitationally bound pair. This is a fundamental
> topological distinction from CDM.

---

## One substrate. One EOS. Six validations.

| Scale | Dataset | Result | Free params |
|---|---|---|---|
| Galaxies | 163 SPARC (HI/Hα) | χ²r = 0.535 | 1–2 |
| Galaxies | 37/42 PHANGS (CO) | 88% good fits | 2 |
| Galaxies | 1225 MaNGA DynPop | 87% within ±0.3 dex | 0 (blind) |
| Clusters | 13 X-COP | χ²r = 0.136 | 2 |
| CMB | Planck 2018 TT | χ²r = 1.330 | Standard cosmology |
| Bullet Cluster | Chandra + GADGET-2 SPH | σ/m < 0.47 cm²/g + Δx confirmed | Analytical + numerical |

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

### `leflux_cmb.py`
CMB TT power spectrum — Addendum 3.
```bash
pip install camb
python3 leflux_cmb.py
```
Reproduces: χ²r = 1.330 on Planck 2018 TT.

### `bullet_cluster_sph.py`
Bullet Cluster SPH simulation — Addendum 4.
```bash
# Generate GADGET-2 initial conditions (1M particles)
python3 bullet_cluster_sph.py --generate

# Analyze snapshot sequence
python3 bullet_cluster_sph.py --analyze --snapdir /path/to/snapshots

# Generate figures
python3 bullet_cluster_sph.py --plot --snapdir /path/to/snapshots
```
Requires: GADGET-2 compiled without `-DPERIODIC`, numpy, scipy, matplotlib.

**Requirements:** numpy, scipy, matplotlib, camb (CMB only)

---

## Preprints

**Le Flux v10** (main paper + Addenda 1–5):  
DOI: [10.5281/zenodo.19219720](https://doi.org/10.5281/zenodo.19219720) *(all versions)*

---

## Open research directions

- **Production mechanism**: The sexaquark hypothesis (Farrar 2017, 2024) provides a candidate for the cold confined state — relic abundance Ω_DM/Ω_b = 5.3 from QCD thermodynamics alone.
- **Full SPH with Le Flux EOS**: GADGET-2 uses an ideal gas EOS. A proper simulation requires a relativistic SPH code with the MIT Bag EOS P = ρ/3 − B_eff(ρ) and a continuous background field.
- **Vortex propagation**: Seeding from BH spin to kpc scales — GRMHD simulation required.
- **CMB predictive power**: Deriving H₀ and Ω_c from QCD fundamentals.
- **γ measurement**: Direct Bayesian fit of γ on SPARC rotation curves.

---

## Author

Aurélien Sureau — May 2026 — Independent research.
