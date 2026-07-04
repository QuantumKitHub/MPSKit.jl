# [Controlling bond dimension](@id howto_bond_dimension)

Bond dimension is the key knob in every MPS calculation: too small and the ansatz cannot represent the state, too large and computation slows to a crawl.
This page gives concrete recipes for inspecting, growing, and shrinking bond dimension in MPSKit.jl.
All examples share a single namespace:

```@example bond_dim
using MPSKit, TensorKit
using TensorKitTensors.SpinOperators: σˣ, σᶻ
```

---

## 1. Inspecting the current bond dimension

MPSKit exposes the virtual spaces through `left_virtualspace` and `right_virtualspace`.
This returns the raw vector spaces, which carry the information about the different sectors, but we can obtain a single number using `dim`:

```@example bond_dim
L = 10
ψ = FiniteMPS(L, ℂ^2, ℂ^8)   # finite MPS, max bond dim 8

# Bond dimension between sites i and i+1 equals dim(left_virtualspace(ψ, i+1))
# or equivalently dim(right_virtualspace(ψ, i)).
dim(left_virtualspace(ψ, 5))   # bond to the left of site 5
```

```@example bond_dim
# All bond dimensions in one go
[dim(left_virtualspace(ψ, i)) for i in 1:L]
```

!!! note
    For a `FiniteMPS` the leftmost and rightmost virtual spaces are typically one-dimensional (the trivial boundary space),
    so `left_virtualspace(ψ, 1)` and `left_virtualspace(ψ, L+1)` have dimension 1.

For an `InfiniteMPS` the same call works per unit-cell site:

```@example bond_dim
ψ_inf = InfiniteMPS(ℂ^2, ℂ^8)
dim(left_virtualspace(ψ_inf, 1))
```

---

## 2. Growing bond dimension

### 2a. Random expansion (no Hamiltonian required)

[`RandExpand`](@ref) pads the MPS with orthogonal random vectors drawn from the two-site null space.
It does **not** need the Hamiltonian, so it is cheap and works for any MPS type.

`trscheme` is **mandatory** and controls how many new directions are added.
Use `truncrank(n)` from MatrixAlgebraKit (re-exported by MPSKit) to add at most `n` extra singular values:

```@example bond_dim
ψ_small = FiniteMPS(L, ℂ^2, ℂ^4)       # start with D = 4
dim(left_virtualspace(ψ_small, 5))
```

```@example bond_dim
ψ_grown = changebonds(ψ_small, RandExpand(; trscheme = truncrank(8)))
dim(left_virtualspace(ψ_grown, 5))      # expanded, but ≤ 4 + 8 = 12
```

The new vectors are orthogonal to the original state, so the state it represents is unchanged (its overlap with the original is 1) while the variational manifold grows.

For an `InfiniteMPS` the call is identical:

```@example bond_dim
ψ_inf_small = InfiniteMPS(ℂ^2, ℂ^4)
ψ_inf_grown = changebonds(ψ_inf_small, RandExpand(; trscheme = truncrank(8)))
dim(left_virtualspace(ψ_inf_grown, 1))
```

### 2b. Optimal expansion (requires Hamiltonian)

[`OptimalExpand`](@ref) selects the dominant contributions of the two-site-updated MPS tensor that are orthogonal to the current state, as described by [Zauner-Stauber et al., Phys. Rev. B 97, 045145 (2018)](https://doi.org/10.1103/PhysRevB.97.045145).
It needs both the state and the Hamiltonian:

```@example bond_dim
# Build a finite TFIM Hamiltonian manually
J = 1.0; g = 0.5
lattice = fill(ℂ^2, L)
X = σˣ()
Z = σᶻ()
H = FiniteMPOHamiltonian(lattice, (i, i + 1) => -J * X ⊗ X for i in 1:(L - 1)) +
    FiniteMPOHamiltonian(lattice, (i,) => -g * Z for i in 1:L)

ψ_opt, envs_opt = changebonds(ψ_small, H, OptimalExpand(; trscheme = truncrank(8)))
dim(left_virtualspace(ψ_opt, 5))
```

`OptimalExpand` also works on `InfiniteMPS` with an `InfiniteMPOHamiltonian`.
The environment argument is optional and defaults to a freshly computed set:

```@example bond_dim
lattice_inf = PeriodicVector([ℂ^2])
H_inf = InfiniteMPOHamiltonian(lattice_inf, (1, 2) => -J * X ⊗ X, (1,) => -g * Z)

ψ_inf_opt, _ = changebonds(ψ_inf_small, H_inf, OptimalExpand(; trscheme = truncrank(8)))
dim(left_virtualspace(ψ_inf_opt, 1))
```

!!! note
    `OptimalExpand` and `VUMPSSvdCut` (see [§5](#5-growing-during-infinite-mps-optimization))
    both require the Hamiltonian.
    Pass environments as the optional fourth argument to avoid recomputing them if you
    already have them from a previous `find_groundstate` call.

---

## 3. Reducing bond dimension

[`SvdCut`](@ref) truncates the bond dimension by an SVD sweep.
It does **not** need the Hamiltonian and is the standard tool for compression.

```@example bond_dim
# compress ψ_grown (D up to 12) back to at most 6 singular values per bond
ψ_cut = changebonds(ψ_grown, SvdCut(; trscheme = truncrank(6)))
dim(left_virtualspace(ψ_cut, 5))
```

An in-place variant, `changebonds!`, exists for `FiniteMPS` and avoids allocating a copy.
It also accepts a `normalize` keyword (default `true`):

```@example bond_dim
ψ_inplace = FiniteMPS(L, ℂ^2, ℂ^12)
changebonds!(ψ_inplace, SvdCut(; trscheme = truncrank(6)); normalize = true)
dim(left_virtualspace(ψ_inplace, 5))
```

`SvdCut` also works on `InfiniteMPS` (2-arg form only; no in-place variant):

```@example bond_dim
ψ_inf_cut = changebonds(ψ_inf_grown, SvdCut(; trscheme = truncrank(6)))
dim(left_virtualspace(ψ_inf_cut, 1))
```

---

## 4. Truncation schemes

Every bond-change algorithm takes a mandatory `trscheme` keyword drawn from **MatrixAlgebraKit** (re-exported by MPSKit).
The main schemes are:

| Scheme | Meaning |
|:-------|:--------|
| `truncrank(n)` | Keep at most `n` singular values |
| `truncbelow(x)` | Drop singular values below the absolute value `x` |
| `trunctol(; atol)` | Drop singular values below `atol` times the largest |
| `notrunc()` | Keep all singular values (no truncation) |
| `truncspace(V)` | Keep only singular values whose index fits in the given space `V` |

Schemes compose with `&` to apply multiple criteria simultaneously.
For example, to keep at most 16 singular values **and** also drop anything below `1e-8`:

```@example bond_dim
trscheme_combined = trunctol(; atol = 1.0e-8) & truncrank(16)
ψ_combined = changebonds(ψ_grown, SvdCut(; trscheme = trscheme_combined))
dim(left_virtualspace(ψ_combined, 5))
```

!!! warning
    `trscheme` is **required** on every algorithm; there is no default.
    Omitting it will throw a `MethodError` at construction time.

---

## 5. Growing during finite MPS optimization

The two-site DMRG variant, [`DMRG2`](@ref), performs a bond expansion at every sweep step by keeping both sites together in the update.
Pass `trscheme` to control which singular values are retained:

```@example bond_dim
ψ_dmrg2_start = FiniteMPS(L, ℂ^2, ℂ^2)   # start small

ψ_dmrg2, envs_dmrg2, _ = find_groundstate(
    ψ_dmrg2_start, H,
    DMRG2(; trscheme = truncrank(16), maxiter = 5)
)
dim(left_virtualspace(ψ_dmrg2, 5))
```

A common pattern is to warm up with `DMRG2` to grow the bond dimension, then refine with single-site `DMRG` for efficiency.
The algorithm chaining operator `&` makes this easy (see [§7](#7-chaining-algorithms)):

```@example bond_dim
warmup_then_refine = DMRG2(; trscheme = truncrank(16), maxiter = 3) &
    DMRG(; maxiter = 20)

ψ_dmrg2, envs_dmrg2, _ = find_groundstate(ψ_dmrg2_start, H, warmup_then_refine)
dim(left_virtualspace(ψ_dmrg2, 5))
```

The `find_groundstate` convenience function also accepts a `trscheme` keyword that triggers the same warm-up automatically:

```@example bond_dim
ψ_conv, envs_conv, _ = find_groundstate(
    ψ_dmrg2_start, H;
    trscheme = truncrank(16), maxiter = 20
)
dim(left_virtualspace(ψ_conv, 5))
```

The `trscheme` keyword makes `find_groundstate` prepend a `DMRG2` pass before switching to the default `DMRG`.

TDVP2 also supports `trscheme` for two-site real- or imaginary-time evolution, but that is covered in the time-evolution documentation rather than here.

---

## 6. Growing during infinite MPS optimization

### IDMRG2 (two-site infinite DMRG)

[`IDMRG2`](@ref) is the infinite analogue of `DMRG2`.

!!! warning
    `IDMRG2` requires a unit cell of **at least 2 sites**.
    Passing a single-site `InfiniteMPS` will throw an `ArgumentError`.

```@example bond_dim
# 2-site unit cell: lattice, Hamiltonian, and initial state
lattice_2 = PeriodicVector([ℂ^2, ℂ^2])
H_inf_2 = InfiniteMPOHamiltonian(
    lattice_2,
    (1, 2) => -J * X ⊗ X,
    (2, 3) => -J * X ⊗ X,
    (1,) => -g * Z,
    (2,) => -g * Z,
)

ψ_idmrg2_start = InfiniteMPS([ℂ^2, ℂ^2], [ℂ^2, ℂ^2])

ψ_idmrg2, _, _ = find_groundstate(
    ψ_idmrg2_start, H_inf_2,
    IDMRG2(; trscheme = truncrank(16), maxiter = 5)
)
dim(left_virtualspace(ψ_idmrg2, 1))
```

### VUMPSSvdCut

[`VUMPSSvdCut`](@ref) grows the bond dimension of an `InfiniteMPS` by performing a two-site VUMPS update followed by an SVD truncation.
It requires the Hamiltonian and returns a new state with updated environments:

```@example bond_dim
ψ_vs, _ = changebonds(ψ_inf_small, H_inf, VUMPSSvdCut(; trscheme = truncrank(16)))
dim(left_virtualspace(ψ_vs, 1))
```

The typical workflow for infinite systems is to grow the bond dimension first (with `VUMPSSvdCut` or `IDMRG2`), then converge with [`VUMPS`](@ref) as a separate step, reusing the expanded state `ψ_vs` from above:

```@example bond_dim
ψ_vc, = find_groundstate(ψ_vs, H_inf, VUMPS(; maxiter = 10))
dim(left_virtualspace(ψ_vc, 1))
```

!!! note
    Bond-changing algorithms such as `VUMPSSvdCut` are applied through
    [`changebonds`](@ref), not `find_groundstate`. Grow the state first, then pass
    the result to a ground-state algorithm.

---

## 7. Chaining algorithms

The `&` operator chains any two algorithms that share the same interface, applying them in sequence.
This works for both ground-state algorithms and `changebonds` algorithms:

```@example bond_dim
# Expand with random vectors, then compress to a target rank
grow_and_cut = RandExpand(; trscheme = truncrank(12)) &
    SvdCut(; trscheme = truncrank(6))

ψ_final = changebonds(ψ_small, grow_and_cut)
dim(left_virtualspace(ψ_final, 5))
```

```@example bond_dim
# Alternatively: combine changebonds with a ground-state algorithm
ψ_expanded, envs_expanded = changebonds(
    ψ_small, H, OptimalExpand(; trscheme = truncrank(8))
)
ψ_gs, _, _ = find_groundstate(ψ_expanded, H, DMRG(; maxiter = 10), envs_expanded)
dim(left_virtualspace(ψ_gs, 5))
```

For background on when each algorithm is appropriate and how convergence is assessed, see [Ground-state algorithms](@ref lib_groundstate).
For constructing MPS objects from scratch, see [Constructing states](@ref howto_states).

