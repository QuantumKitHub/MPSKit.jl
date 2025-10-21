# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Fixed

### Changed

- The `changebonds(state, ::RandExpand)` algorithm now no longer projects onto the nullspace
  on both sides. This ensures that the expanded symmetry sectors can be selected beyond what
  is allowed by two-site updates, which can be relevant for certain systems that have
  symmetry-related obstructions.

### Deprecated

### Removed

[unreleased]: https://github.com/quantumkithub/pepskit.jl/compare/v0.13.8...HEAD
