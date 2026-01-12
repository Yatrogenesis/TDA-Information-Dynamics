# Release Notes

## v0.2.0 (January 12, 2026)

### Major Features

**Exact Persistent Homology**
- New module `persistence_exact.rs` implementing standard algorithm
- Boundary matrix reduction with column operations (Z/2Z)
- True birth/death pairs matching Python/Ripser output
- Required for entropy-based crystallization detection

**5 Dynamical Systems Validated**
- Brusselator (Hopf bifurcation)
- FitzHugh-Nagumo (excitable neuron network)
- XY Model (BKT transition)
- Kuramoto (synchronization)
- Lennard-Jones (crystallization)

**7 Executable Binaries**
- `brusselator_tda_cusum` - Chemical oscillator
- `fhn_tda_cusum` - Neuron network
- `xy_tda_cusum` - Planar spin model
- `kuramoto_tda` - Coupled oscillators
- `lj_tda_cusum` - Crystallization (Euler)
- `lj_exact_tda_cusum` - Crystallization (Exact)
- `lj_exact_vs_euler` - Method comparison

### Key Results

All systems demonstrate topological precursor detection:

| System | CUSUM | Traditional | Early Warning |
|--------|-------|-------------|---------------|
| Brusselator | B=1.36 | B=3.08 | +1.72 |
| FitzHugh-Nagumo | I=0.15 | I=0.56 | +0.41 |
| XY Model | T=0.73 | T=0.94 | +0.21 |
| Kuramoto | K=0.60 | K=1.10 | +0.50 |
| Lennard-Jones | 73% rate | - | Variable |

### Technical Improvements

- `sigma_min` parameter in CUSUM prevents infinite sensitivity
- Euler approximation retained for real-time monitoring
- 35 unit tests passing
- Comprehensive documentation

---

## v0.1.0 (January 11, 2026)

### Initial Release

- TDA-CUSUM framework core implementation
- Vietoris-Rips filtration construction
- Euler-approximated persistent homology
- Topological entropy computation
- CUSUM detector with calibration
- Lennard-Jones and Kuramoto systems
- Basic documentation

---

## Author

Francisco Molina-Burgos
Avermex Research Division
Merida, Yucatan, Mexico
