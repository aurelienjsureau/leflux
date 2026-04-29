# Le Flux — Primordial QGP Cosmology

> *The universe is a single, structured quantum fluid — and its dynamics, from galactic rotation curves to cluster hydrostatic equilibrium to the cosmic microwave background, emerge self-consistently from its non-barotropic, non-Newtonian rheology.*

A cosmological model in which the Big Bang is not a past event but an ongoing process. Space itself is identified with the continuously expanding primordial quark-gluon plasma (QGP), governed by the MIT Bag equation of state. There is no dark matter, no dark energy — these concepts become unnecessary.

---

## Results

### Galaxies — SPARC (163 galaxies, Lelli et al. 2016)

EOS formula: `V²(r) = V²_bar(r) + V₀²·r²/(r_c²+r²)`

| Model | Median χ²r | Free params |
|-------|-----------|-------------|
| **Le Flux v10** | **0.535** | **1–2** |
| NFW | 1.47 | 2–3 |
| Burkert | 0.79 | 2–3 |
| MOND/RAR | ~1.2 | 1 |

With r_c physically bounded by R_HI → **median χ²r = 0.687 with 1 free parameter**

The coherence length r_c correlates with the HI disk radius R_HI (R²partial=0.076, p=4.3×10⁻³, N=106) — a prediction ΛCDM does not make.

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

---

## One substrate. One EOS. Three scales.

| Scale | Dataset | χ²r | Free params |
|-------|---------|-----|-------------|
| Galaxies | 163 SPARC | 0.535 | 1–2 |
| Clusters | 13 X-COP | 0.136 | 2 |
| CMB | Planck 2018 TT | 1.330 | Standard cosmology |

No dark matter. No dark energy.

---

## Scripts

### `flux_complet.py`
SPARC rotation curve fitting.
```bash
# Download SPARC data: https://astroweb.cwru.edu/SPARC/
python3 flux_complet.py
```

### `leflux_xcop_addendum2.py`
X-COP galaxy cluster fitting — Addendum 2.
```bash
python3 leflux_xcop_addendum2.py
```
Reproduces: median χ²r = 0.136, 13/13 clusters with χ²r < 3.

### `flux_cmb.py`
CMB TT power spectrum — Addendum 3.
```bash
pip install camb
python3 leflux_cmb_addendum3.py
```
Reproduces: χ²r = 1.330 on Planck 2018 TT.

**Requires:** numpy, scipy, matplotlib, camb

---

## Preprints

**Le Flux v10** (main paper + all Addenda):
DOI: [10.5281/zenodo.19785792](https://doi.org/10.5281/zenodo.19785792)

---

## Open challenges

- **Production mechanism**: How does the QGP substrate achieve the correct cold relic abundance from the QCD crossover?
- **Self-interaction cross-section**: Cold confined states have σ/m ~ 1 cm²/g — above the Bullet Cluster constraint of 0.47 cm²/g.
- **Vortex propagation**: Seeding from BH spin to kpc scales requires GRMHD simulation.
- **CMB predictive power**: Deriving H₀ and Ω_c from QCD fundamentals remains the next theoretical step.

---

## Author

Aurélien Sureau — April 2026 — Independent research.
