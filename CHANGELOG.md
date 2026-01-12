# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-01-12

### Added
- **Exact persistent homology** (`persistence_exact.rs`)
  - Standard algorithm with boundary matrix reduction
  - True birth/death pairs for persistence intervals
  - `ExactPersistenceDiagram` and `ExactInterval` types
  - `compute_exact_persistence()` and `compute_exact_persistence_simple()` functions
  - Matches Python/Ripser output for validation
- **New dynamical systems**
  - Brusselator chemical oscillator (`brusselator.rs`)
  - FitzHugh-Nagumo excitable neuron network (`fitzhugh_nagumo.rs`)
  - XY Model planar spin system (`xy_model.rs`)
- **New binaries**
  - `brusselator_tda_cusum` - Hopf bifurcation detection
  - `fhn_tda_cusum` - Network excitability transition
  - `xy_tda_cusum` - BKT transition detection
  - `lj_exact_tda_cusum` - Crystallization with exact persistence
  - `lj_exact_vs_euler` - Method comparison tool
- **System traits** (`traits.rs`)
  - `DynamicalSystem` - common interface
  - `Controllable` - parameter control
  - `Bifurcating` - bifurcation analysis
  - `TDAConfig` - topological analysis configuration
- **CUSUM enhancements**
  - `sigma_min` parameter to prevent infinite sensitivity
  - `with_sigma_min()` constructor
- **4 new unit tests** for exact persistence

### Changed
- Version bump from 0.1.0 to 0.2.0
- Updated author email to fmolina@avermex.com
- Comprehensive README with all 5 systems
- Module structure documentation

### Fixed
- CUSUM calibration sign handling for Kuramoto system
- Borrow checker issue in exact persistence tests

## [0.1.0] - 2026-01-11

### Added
- Initial TDA-CUSUM framework implementation
- Vietoris-Rips filtration construction (`vietoris_rips.rs`)
- Euler-approximated persistent homology (`persistence.rs`)
- Betti number computation (`betti.rs`)
- Topological entropy calculation (`entropy.rs`)
- Persistence landscapes (`landscape.rs`)
- CUSUM detector with calibration (`detector.rs`)
- Lennard-Jones fluid simulation (`lennard_jones.rs`)
- Kuramoto oscillator system (`kuramoto.rs`)
- Binary executables: `lj_tda_cusum`, `kuramoto_tda`
- 31 unit tests
- MIT License

---

## Links

- [Repository](https://github.com/Yatrogenesis/TDA-Information-Dynamics)
- [Related: TDA-Phase-Transitions (Python)](https://github.com/Yatrogenesis/TDA-Phase-Transitions)
- [Related: TDA-Complex-Systems (Rust)](https://github.com/Yatrogenesis/TDA-Complex-Systems)
