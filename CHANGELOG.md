# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `LocalPreferences.toml` file to ensure `TensorOperations` properly precompiles on testing
  infrastructure

### Fixed

- Dynamic tolerances yielded `NaN` during the initialization stage due to `1 / sqrt(iter)`
  where `iter = 0`.
- `InfiniteMPOHamiltonian` environments with low bond dimension and high Krylov dimension now are properly
  clamped.

### Changed

- The `changebonds(state, ::RandExpand)` algorithm now no longer has to perform a
  truncated SVD to obtain the desired spaces, and instead sample the space directly
  and then generates a random isometry. This should be slightly more performant, but
  otherwise equivalent.

### Deprecated

### Removed

[unreleased]: https://github.com/quantumkithub/pepskit.jl/compare/v0.13.8...HEAD
