# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Guidelines for updating this changelog

When making changes to this project, please update the "Unreleased" section with your changes under the appropriate category:

- **Added** for new features.
- **Changed** for changes in existing functionality.
- **Deprecated** for soon-to-be removed features.
- **Removed** for now removed features.
- **Fixed** for any bug fixes.

When releasing a new version, move the "Unreleased" changes to a new version section with the release date.

## [Unreleased](https://github.com/QuantumKitHub/MPSKit.jl/compare/v0.13.9...HEAD)

### Added

- `expectation_value` for local MPO tensors
  ([#327](https://github.com/QuantumKitHub/MPSKit.jl/pull/327))
- `Base.copy` for MPS types now performs a deep copy
  ([#387](https://github.com/QuantumKitHub/MPSKit.jl/pull/387))

### Changed

- `entropy` can now also be called directly on a spectrum (singular value vector)
  ([#377](https://github.com/QuantumKitHub/MPSKit.jl/pull/377))
- Updated compat bounds to remove broken package versions

### Deprecated

### Removed

### Fixed

- Fixed `Adapt` extension for GPU support
  ([#389](https://github.com/QuantumKitHub/MPSKit.jl/pull/389))

## [0.13.9](https://github.com/QuantumKitHub/MPSKit.jl/compare/v0.13.8...v0.13.9) - 2026-02-03

### Added

- `LocalPreferences.toml` file to ensure `TensorOperations` properly precompiles on testing
  infrastructure
- `GeometryStyle` and `OperatorStyle` traits for dispatching on finite/infinite geometry and
  operator types ([#352](https://github.com/QuantumKitHub/MPSKit.jl/pull/352), [#354](https://github.com/QuantumKitHub/MPSKit.jl/pull/354))
- `Base.isfinite` methods for MPS types ([#347](https://github.com/QuantumKitHub/MPSKit.jl/pull/347))
- Bose-Hubbard example ([#342](https://github.com/QuantumKitHub/MPSKit.jl/pull/342))
- WindowMPS example update ([#350](https://github.com/QuantumKitHub/MPSKit.jl/pull/350))
- Multifusion category compatibility ([#297](https://github.com/QuantumKitHub/MPSKit.jl/pull/297))

### Fixed

- Dynamic tolerances yielded `NaN` during the initialization stage due to `1 / sqrt(iter)`
  where `iter = 0` ([#335](https://github.com/QuantumKitHub/MPSKit.jl/pull/335))
- `InfiniteMPOHamiltonian` environments with low bond dimension and high Krylov dimension now are properly
  clamped ([#335](https://github.com/QuantumKitHub/MPSKit.jl/pull/335))
- Logical operator precedence in `getproperty` function ([#346](https://github.com/QuantumKitHub/MPSKit.jl/pull/346))
- Typo in `VUMPSSvdCut` ([#361](https://github.com/QuantumKitHub/MPSKit.jl/pull/361))
- Typo in time formatting for logs ([#336](https://github.com/QuantumKitHub/MPSKit.jl/pull/336))
- Domain/codomain of `MPODerivativeOperator` ([#370](https://github.com/QuantumKitHub/MPSKit.jl/pull/370))
- In-place operations handled more carefully ([#337](https://github.com/QuantumKitHub/MPSKit.jl/pull/337))
- Orthogonalization algorithms now use correct methods ([#373](https://github.com/QuantumKitHub/MPSKit.jl/pull/373))

### Changed

- The `changebonds(state, ::RandExpand)` algorithm now no longer has to perform a
  truncated SVD to obtain the desired spaces, and instead sample the space directly
  and then generates a random isometry. This should be slightly more performant, but
  otherwise equivalent ([#335](https://github.com/QuantumKitHub/MPSKit.jl/pull/335))
- `IDMRG` refactored to follow the `IterativeSolver` interface and share code between
  `IDMRG` and `IDMRG2` ([#348](https://github.com/QuantumKitHub/MPSKit.jl/pull/348))
- Bumped compatibility for TensorKit 0.16 and MatrixAlgebraKit 0.6 ([#365](https://github.com/QuantumKitHub/MPSKit.jl/pull/365))
- Removed `_left_orth` and `_right_orth` workarounds in favor of new orthogonalization methods
- Reduced allocation while computing Galerkin error ([#366](https://github.com/QuantumKitHub/MPSKit.jl/pull/366))
- Updated `show` methods to reflect new TensorKit printing ([#341](https://github.com/QuantumKitHub/MPSKit.jl/pull/341))
- More informative errors for finite MPS ([#367](https://github.com/QuantumKitHub/MPSKit.jl/pull/367))
- Minor documentation and docstring improvements ([#363](https://github.com/QuantumKitHub/MPSKit.jl/pull/363), [#372](https://github.com/QuantumKitHub/MPSKit.jl/pull/372), [#371](https://github.com/QuantumKitHub/MPSKit.jl/pull/371))

### Deprecated

### Removed

## [0.13.8](https://github.com/QuantumKitHub/MPSKit.jl/releases/tag/v0.13.8) - 2024-10-31

See full history and previous releases on [GitHub](https://github.com/QuantumKitHub/MPSKit.jl/releases).
