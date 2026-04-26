Le Flux — Primordial QGP Cosmology
> *The universe is a single, structured quantum fluid — and its dynamics, from galactic rotation curves to cluster hydrostatic equilibrium, emerge self-consistently from its non-barotropic, non-Newtonian rheology.*
A cosmological model in which the Big Bang is not a past event but an ongoing process. Space itself is identified with the continuously expanding primordial quark-gluon plasma (QGP), governed by the MIT Bag equation of state. There is no dark matter, no dark energy — these concepts become unnecessary.
---
Results
Galaxies — SPARC (163 galaxies, Lelli et al. 2016)
EOS formula: `V²(r) = V²_bar(r) + V₀²·r²/(r_c²+r²)`
Model	Median χ²r	Free params
Le Flux v10	0.535	1–2
NFW	1.47	2–3
Burkert	0.79	2–3
MOND/RAR	~1.2	1
With r_c physically bounded by R_HI → median χ²r = 0.687 with 1 free parameter
Galaxy Clusters — X-COP (13 clusters, Ettori+2018) — Addendum 2
Unified non-barotropic EOS (cluster limit):
```
dρ/dr = -ρ(dΦ/dr + dσ²_turb/dr) / σ²_turb
σ_turb(r) = k · σ_BCG · (1 + r/R200)^{-0.5}
```
Result	Le Flux	NFW
Median χ²r	0.136	~0 (tautology*)
N(χ²r < 3)	13/13	13/13
Free params	2 (k + ρ_c)	2 (M₂₀₀ + c₂₀₀)
Param source	σ_BCG (independent)	Fitted on same data
*NFW χ²r ≈ 0 because M₂₀₀ and c₂₀₀ are fitted directly on X-COP data — tautology, not prediction.
k median = 0.999, range [0.84, 1.21] — physically bounded by lattice QCD (k_max ≈ 1.4)
---
Scripts
`flux_complet.py`
Main SPARC rotation curve fitting script.
```bash
# Download SPARC data: https://astroweb.cwru.edu/SPARC/
# Place in ./sparc_data/ then:
python3 flux_complet.py
```
`leflux_xcop_addendum2.py`
X-COP galaxy cluster fitting — Addendum 2.
```bash
python3 leflux_xcop_addendum2.py
```
Reproduces: median χ²r = 0.136, 13/13 clusters with χ²r < 3.
Requires: numpy, scipy, matplotlib
---
Preprints
Le Flux v10 (main paper + Addendum 1: r_c ∝ R_HI):
DOI: 10.5281/zenodo.19610251
Addendum 2 (galaxy clusters, X-COP):
Included as `LeFlux_v10_Addendum2.docx` — to be submitted to Zenodo v4.
---
Open challenges
Bullet Cluster: σ/m < 0.47 cm²/g constraint remains open for any fluid substrate.
CMB / BAO / structure growth: Not yet tested quantitatively.
Vortex propagation: Seeding from BH spin to kpc scales requires GRMHD simulation.
---
Author
Aurélien Sureau — April 2026 — Independent research.
