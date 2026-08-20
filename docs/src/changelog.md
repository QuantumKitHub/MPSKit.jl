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

## [Unreleased](https://github.com/QuantumKitHub/MPSKit.jl/compare/v0.13.11...HEAD)

### Added

- Addition of `FiniteMPS`/`FiniteMPO` with different scalar types, through a new
  `Base.similar(ψ, ::Type{S})` for `S <: Number` on `FiniteMPS`. ([#484](https://github.com/QuantumKitHub/MPSKit.jl/pull/484))
- `Zipup`, an algorithm for `approximate`/`approximate!` that compresses a finite MPO-MPS product in
  a single sweep, optionally followed by a sweep in the opposite direction that imposes the final
  truncation. The sweep direction is selected by the `left_to_right` keyword. Both
  `approximate((O, ϕ), alg)` and `approximate!(ψ, (O, ϕ), alg)` are supported, where the destination
  `ψ` is a write target rather than an initial guess and may alias `ϕ`; they return `(ψ, ϵ)`.
- `BUG` time-evolution algorithm: a Basis-Update & Galerkin integrator for finite MPS.
  Unlike `TDVP` it has no backward-in-time substep (stable for imaginary-time evolution),
  and passing a truncating `trunc` enables rank-adaptivity (the bond dimension grows and shrinks
  automatically to track entanglement).
- A `backend` setting on every algorithm, for its tensor contractions and index manipulations,
  defaulting to `MPSKit.Defaults.backend()`. ([#467](https://github.com/QuantumKitHub/MPSKit.jl/pull/467))
- Local updates now serve their intermediate tensors from a dedicated allocator, selected internally
  by `MPSKit.default_allocator`, instead of leaving them to the garbage collector
  (two-site DMRG: -64% allocations, -57% GC time, -23% wall time).
  Disable with `MPSKit.Defaults.set_buffering!(false)`. ([#467](https://github.com/QuantumKitHub/MPSKit.jl/pull/467))
- Custom `show`/`summary` for `MultilineMPS`/`MultilineMPO`. Each row is now rendered via each row's own display, and row shifting is shown explicitly for `MultilineMPO`.
- `*(::MultilineMPO, ::InfiniteMPS)`, which pushes the boundary MPS through every row of the
  network in turn, advancing it by one full period.
- `dominant_eigenvalue(ψ, O, [environments])`, the eigenvalue of the transfer operator `O` for the
  boundary MPS `ψ`. `expectation_value(::InfiniteMPS, ::InfiniteMPO)` forwards here.
### Changed

- Renormalization during time evolution is now controlled by an explicit `normalize` keyword on
  `timestep`/`time_evolve` (default `false`), decoupled from `imaginary_evolution`. By default the
  norm is preserved, so it retains useful information (the accumulated truncation error in real time,
  or the decaying weight in imaginary time). Previously imaginary-time evolution always renormalized
  every step; **to recover that behavior, pass `normalize = true`** (e.g. for ground-state or
  thermal-state search via imaginary-time evolution).
- `environments` now follows a single positional contract for every state and operator kind:
  `environments(below, operator, above, alg)`, where `alg` is the environment algorithm
  (slot 4). The operator form requires an explicit `above`. Auxiliary inputs are keyword-only:
  `leftstart`/`rightstart` for finite and window environments, and `lenvs`/`renvs` for window
  and quasiparticle environments.
  The two-argument form `environments(below, above)` (with two states) is reserved for the
  operator-free overlap environments. There is no two-argument `environments(below, operator)`
  shorthand: a two-argument call is always the overlap, since the second argument cannot be
  disambiguated between a ket and an operator (undecidable for density matrices, where states
  and operators share a representation). ([#436](https://github.com/QuantumKitHub/MPSKit.jl/pull/436))
- `transfer_spectrum` now computes the spectrum for all sectors of the transfer space at once,
  returning a `TensorKit.SectorVector` that can be indexed per sector.
  The `sector` keyword is removed; a specific selection of sectors (with per-sector counts) can be
  requested by passing `howmany` as an `AbstractDict`/iterable of `sector => count` pairs.
  The `below` state and the eigensolver algorithm are optional positional arguments, and the
  algorithm can be resolved per sector. The Krylov dimension adapts to the number of values requested
  in each sector, controlled by the new `oversampling` and `oversampling_factor` keywords.
  Accordingly, `marek_gap` and `correlation_length` now return a `TensorKit.SectorDict` of
  per-sector results by default; pass `sector = ...` to obtain a single sector's result as before.
- All `trscheme` keyword arguments are renamed to `trunc` ([#482](https://github.com/QuantumKitHub/MPSKit.jl/pull/482)).
- `correlator` now throws an `ArgumentError` when the sites are not ordered as `i < j`.
  Previously such a call only logged an `@error` and then continued into a contraction that is
  not the requested correlator. ([#489](https://github.com/QuantumKitHub/MPSKit.jl/pull/489))
- `Multiline` (and therefore `MultilineMPS`/`MultilineMPO`) now consistently treats
  `length`/`eltype`/`iterate`/`m[i]` as referring to the individual lines it stores
  (`length(m) == nrows`), while `size`/`axes`/`eachindex` refer to the `(nrows, ncols)` lattice
  shape.
- `MultilineMPO` and `MultilineMPS` lines are now restricted by the type to
  `Union{InfiniteMPO, FiniteMPO}` and `Union{InfiniteMPS, FiniteMPS}` respectively, rather than to
  any `AbstractMPO`/`InfiniteMPS`. Hamiltonian lines are excluded outright. Finite lines are
  accepted by both the type and the constructors so that finite multiline networks can be built
  and inspected. No algorithm supports them yet, so they fail further down. The `AbstractMatrix`
  constructor that silently built finite-line `MultilineMPO`s was removed.

### Deprecated

### Removed

- `expectation_value(::MultilineMPS, ::MultilineMPO, envs...)` fallback method, which silently
  computed a meaningless value (`prod` instead of `sum`, no row shift, `envs` ignored) for any
  `MultilineMPO` line type not covered by the guarded method. Most notably this prevents a fallback for `InfiniteMPOHamiltonian`, a legal but never-meaningful `Multiline` line type.
- `expectation_value` for a `MultilineMPS`/`MultilineMPO` pair entirely, replaced by
  `dominant_eigenvalue`.
- `*(::MultilineMPO, ::MultilineMPS)` and `*(::MultilineMPO, ::MultilineMPO)`, as these
  were not meaningful operations. Neither method had ever been callable previously.

### Fixed

- `isfinite(::WindowMPOHamiltonian)` was undefined. ([#489](https://github.com/QuantumKitHub/MPSKit.jl/pull/489))
- `checkbounds` on the `AL`/`AR`/`AC`/`C` views of a `Multiline` now delegates to the
  matching view, and dispatches on `Multiline{<:InfiniteMPS}` versus `Multiline{<:AbstractFiniteMPS}`.
  The row index remains unchecked in both cases due to periodicity.
- `size`/`axes` for a `CView` over a `Multiline` with finite lines were missing.
- `excitations(::InfiniteMPO, ::QuasiparticleAnsatz, ::InfiniteQP, lenvs, renvs)` referenced `H_eff`  before assigning. ([#489](https://github.com/QuantumKitHub/MPSKit.jl/pull/489))
- `Base.:+`/`-` on `FiniteMPS` returned a wrong state for near-parallel operands carried by
  different tensor networks, e.g. `norm(E₀ * gs - H * gs)` coming out as `2 * norm(gs) * E₀`
  instead of ~0. The lazy gauge sweep in `CView.getindex` re-derived `AL`/`C` entries that were
  already cached, and since different code paths install different factorizations (a truncated SVD
  from DMRG vs. a positive QR from the sweep) the replacement differed by a bond unitary, so `+`
  combined tensors belonging to two different gauges. The same staleness was latent in every
  consumer that reads several gauge tensors across a center move, `dot` included.
  ([#473](https://github.com/QuantumKitHub/MPSKit.jl/issues/473), [#484](https://github.com/QuantumKitHub/MPSKit.jl/pull/484))
- Addition of single-site operands. `FiniteMPS + FiniteMPS` asserted `length > 1`, and
  `FiniteMPO + FiniteMPO` threw a space error, because both split the chain into a left and a
  right block and fuse them at the seam — of which there is none at length 1. Such an operand has
  no internal bond to fuse, so the sum is now simply the sum of the two tensors.
  ([#484](https://github.com/QuantumKitHub/MPSKit.jl/pull/484))
- `convert(TensorMap, ::FiniteMPO)` on a single-site MPO stripped the left virtual leg of `mpo[1]`
  and the right virtual leg of `mpo[end]` and contracted the two — which at length 1 is the *same*
  tensor, so it returned `O * O` on twice the physical space instead of `O`.
  ([#484](https://github.com/QuantumKitHub/MPSKit.jl/pull/484))
- `isfinite(::MultilineMPO)` threw (`isfinite(typeof(m))` had no matching type-level method for
  `Multiline`).
- `changebonds(::MultilineMPO, ::SvdCut)` threw (`convert(MultilineMPS, ::MultilineMPO)` has no
  method).
- `axes(m::Multiline, i)` threw for `i > 2`, but now returns `Base.OneTo(1)`, matching Base's own
  out-of-range convention (already the case for `size(m, i)`).
- `spacetype`/`sectortype`/`storagetype` on a `Multiline` instance were undefined. Only the type-level methods existed.

### Performance

## [0.13.11](https://github.com/QuantumKitHub/MPSKit.jl/compare/v0.13.10...v0.13.11) - 2026-05-04

### Added

- `MultilineMPO` space getters ([#407](https://github.com/QuantumKitHub/MPSKit.jl/pull/407))

### Changed

- Refactored time-evolution MPO construction ([#422](https://github.com/QuantumKitHub/MPSKit.jl/pull/422))
- Updated `TensorKitManifolds` compat to 0.8 ([#421](https://github.com/QuantumKitHub/MPSKit.jl/pull/421))
- Updated MatrixAlgebraKit algorithm specification ([#418](https://github.com/QuantumKitHub/MPSKit.jl/pull/418))
- Preparations for GPU / non-CPU array support ([#375](https://github.com/QuantumKitHub/MPSKit.jl/pull/375), [#392](https://github.com/QuantumKitHub/MPSKit.jl/pull/392))
- Generalized `calc_galerkin` to `AbstractMPS` ([#395](https://github.com/QuantumKitHub/MPSKit.jl/pull/395))
- Removed explicit call to `InfiniteMPS` in VUMPS ([#396](https://github.com/QuantumKitHub/MPSKit.jl/pull/396))
- Generalized `regauge!` to `AbstractVector` ([#393](https://github.com/QuantumKitHub/MPSKit.jl/pull/393))

### Fixed

- Various fixes for compatibility with latest TensorKit versions ([#416](https://github.com/QuantumKitHub/MPSKit.jl/pull/416))
- `changebonds` inconsistencies ([#415](https://github.com/QuantumKitHub/MPSKit.jl/pull/415))
- Small fixes for density operators ([#409](https://github.com/QuantumKitHub/MPSKit.jl/pull/409))
- Tolerance on positivity test ([#398](https://github.com/QuantumKitHub/MPSKit.jl/pull/398))

### Performance

- Benchmarks and AC/AC2 contraction improvements ([#345](https://github.com/QuantumKitHub/MPSKit.jl/pull/345))

## [0.13.10](https://github.com/QuantumKitHub/MPSKit.jl/compare/v0.13.9...v0.13.10) - 2026-02-26

### Added

- `expectation_value` for local MPO tensors
  ([#327](https://github.com/QuantumKitHub/MPSKit.jl/pull/327))
- `Base.copy` for MPS types now performs a deep copy
  ([#387](https://github.com/QuantumKitHub/MPSKit.jl/pull/387))

### Changed

- `entropy` can now also be called directly on a spectrum (singular value vector)
  ([#377](https://github.com/QuantumKitHub/MPSKit.jl/pull/377))
- Updated compat bounds to remove broken package versions

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
