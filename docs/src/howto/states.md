# [Constructing states](@id howto_states)

This page collects recipes for building [`FiniteMPS`](@ref), [`InfiniteMPS`](@ref), [`WindowMPS`](@ref), and [`MultilineMPS`](@ref) objects.
All constructors live in the `MPSKit` namespace; the examples below assume

```@example howto_states
using MPSKit, TensorKit
```

For background on what these types represent and how gauging works, see the [States](@ref lib_states) reference page.

---

## 1. A finite MPS from length, physical space, and maximum bond dimension

The most common starting point: give the chain length `N`, the local physical `VectorSpace`, and the maximum allowed virtual space.
The constructor fills the tensors with random `ComplexF64` entries and trims the actual bond dimensions to full rank, so passing an over-large `maxVspace` is safe.

```@example howto_states
L = 10
d = ℂ^2       # spin-1/2 physical space (dim 2)
D = ℂ^16      # maximum bond dimension

ψ = FiniteMPS(L, d, D)
```

Inspect the resulting virtual spaces with `left_virtualspace` and `right_virtualspace`.
To get the numeric bond dimension at bond `i` use `dim`:

```@example howto_states
dim(left_virtualspace(ψ, 3))   # bond dimension between sites 2 and 3
```

```@example howto_states
physicalspace(ψ, 1)             # local Hilbert space at site 1
```

---

## 2. Choosing the initializer and element type

Pass an initializer function (`rand` or `randn`) and an element type as the first two arguments:

```@example howto_states
ψ_rand = FiniteMPS(rand, ComplexF64, L, d, D)   # default — same as FiniteMPS(L, d, D)
ψ_randn = FiniteMPS(randn, ComplexF64, L, d, D)   # normally distributed entries
```

The element type sets the scalar type of the tensors, e.g. `ComplexF64` (the default) or `Float64` for a real-valued state.

---

## 3. Per-site physical and virtual spaces

When the physical space varies from site to site — or you want fine control over which bond gets which maximum dimension — pass vectors instead of scalars.
The `maxVspaces` vector must have length `N - 1` (one entry per bond):

```@example howto_states
Pspaces = [ℂ^2, ℂ^3, ℂ^2, ℂ^3, ℂ^2]   # alternating physical spaces
maxVspaces = [ℂ^8, ℂ^8, ℂ^8, ℂ^8]        # one per bond (length N-1)

ψ_het = FiniteMPS(rand, ComplexF64, Pspaces, maxVspaces)
```

```@example howto_states
physicalspace(ψ_het, 2)   # ℂ^3
```

---

## 4. A product state (trivial virtual space)

A product (bond-dimension-1) state has no entanglement: each site carries its own single-site state, independent of the others (with `rand`, a random such state per site).
Achieve this by passing `oneunit(d)` — the one-dimensional unit space of the same symmetry sector — as the maximum virtual space:

```@example howto_states
ψ_prod = FiniteMPS(rand, ComplexF64, L, ℂ^2, oneunit(ℂ^2))
dim(left_virtualspace(ψ_prod, 5))   # should be 1
```

!!! note
    `oneunit(V)` returns the one-dimensional trivial space matching the symmetry type of `V`.
    For plain complex spaces, `oneunit(ℂ^2) == ℂ^1`.

---

## 5. From your own site tensors

If you already have a vector of `TensorMap` objects with the correct index structure (virtual ⊗ physical ← virtual), pass them directly.
The constructor performs a left-to-right QR sweep to bring the state into a canonical form:

```@example howto_states
# build three-site rank-1 tensors by hand
site_tensors = [rand(ComplexF64, ℂ^1 ⊗ ℂ^2 ← ℂ^1) for _ in 1:L]
ψ_from_tensors = FiniteMPS(site_tensors)
```

Set `normalize = true` to also normalize the state during construction (the default is `false` when passing raw tensors):

```@example howto_states
ψ_normed = FiniteMPS(site_tensors; normalize = true)
```

---

## 6. An infinite MPS

### Scalar convenience form

Provide `d` and `D` as integers or spaces; the constructor builds a single-site unit cell:

```@example howto_states
ψ_inf = InfiniteMPS(2, 20)        # integers → plain ComplexSpace dimensions
```

```@example howto_states
ψ_inf2 = InfiniteMPS(ℂ^2, ℂ^20)  # same, spelled out as spaces
```

### Multi-site unit cell

Pass vectors of physical and virtual spaces.
The virtual spaces are those to the *right* of the corresponding sites:

```@example howto_states
ψ_2site = InfiniteMPS([ℂ^2, ℂ^2], [ℂ^20, ℂ^20])
```

```@example howto_states
physicalspace(ψ_2site, 1)
```

```@example howto_states
right_virtualspace(ψ_2site, 1)   # virtual space to the right of site 1
```

### Choosing element type and initializer

```@example howto_states
ψ_inf_r = InfiniteMPS(rand, Float64, [ℂ^2], [ℂ^10])
```

### From site tensors

Tensors must form a valid periodic chain (virtual spaces must match across the unit-cell boundary):

```@example howto_states
inf_tensors = [rand(ComplexF64, ℂ^4 ⊗ ℂ^2 ← ℂ^4)]
ψ_inf_t = InfiniteMPS(inf_tensors)
```

---

## 7. A window MPS

A [`WindowMPS`](@ref) embeds a mutable finite window inside two infinite environments.

### Slice an existing InfiniteMPS

The simplest route: pick a region of length `L` from an `InfiniteMPS`.
Both environments are set to the same object (the original infinite state):

```@example howto_states
ψ_bulk = InfiniteMPS(ℂ^2, ℂ^8)
ψ_win = WindowMPS(ψ_bulk, 6)     # window of 6 sites
```

```@example howto_states
length(ψ_win)                     # 6
```

### From space specifications

Provide the window dimensions together with the infinite environments.
The boundary virtual spaces are taken automatically from `ψₗ`/`ψᵣ`:

```@example howto_states
ψ_win2 = WindowMPS(rand, ComplexF64, 6, ℂ^2, ℂ^8, ψ_bulk)
```

### From a FiniteMPS and two environments

Build a `FiniteMPS` with matching boundary virtual spaces first, then wrap:

```@example howto_states
finite_part = FiniteMPS(6, ℂ^2, ℂ^8; left = ℂ^8, right = ℂ^8)
ψ_win3 = WindowMPS(ψ_bulk, finite_part, ψ_bulk)
```

!!! warning
    When `ψᵣ` is omitted in the outer constructors, the right environment is
    **the same object** as the left environment (no copy is made).
    If you later evolve the two environments independently, pass `copy(ψ_bulk)`
    explicitly as the right argument to avoid aliasing:

    ```julia
    ψ_win_safe = WindowMPS(rand, ComplexF64, 6, ℂ^2, ℂ^8, ψ_bulk, copy(ψ_bulk))
    ```

---

## 8. A multiline MPS

[`MultilineMPS`](@ref) stacks several [`InfiniteMPS`](@ref) rows and is used in boundary-MPS methods for 2D classical partition functions.

### From a vector of InfiniteMPS rows

```@example howto_states
row1 = InfiniteMPS(ℂ^2, ℂ^8)
row2 = InfiniteMPS(ℂ^2, ℂ^8)
ψ_ml = MultilineMPS([row1, row2])
```

Access tensors with Cartesian `[row, col]` indexing:

```@example howto_states
ψ_ml.AL[1, 1]   # left-gauged tensor of row 1, unit-cell site 1
```

### From space matrices

Pass matrices whose rows correspond to MPS rows and columns to unit-cell sites:

```@example howto_states
pspaces = fill(ℂ^2, 2, 2)    # 2 rows × 2-site unit cell
Dspaces = fill(ℂ^8, 2, 2)
ψ_ml2 = MultilineMPS(pspaces, Dspaces)
```

---

## 9. States with symmetries

All constructors accept TensorKit graded spaces.
Pass a `Rep[G]` physical space and a `Rep[G]` maximum virtual space; the constructor automatically selects the consistent fusion channels.

### Finite MPS with U(1) symmetry

```@example howto_states
# U(1) spin-1/2: physical space = spin up (charge +1/2) + spin down (charge -1/2)
d_u1 = Rep[U₁](1 // 2 => 1, -1 // 2 => 1)   # dim 2 total
# the virtual space must span both charge parities (integer and half-integer):
# with only ±1/2 on each site, the total charge alternates parity bond to bond,
# so a purely half-integer virtual space would starve every even bond
D_u1 = Rep[U₁](0 => 2, 1 // 2 => 2, -1 // 2 => 2, 1 => 1, -1 => 1)

ψ_u1 = FiniteMPS(rand, ComplexF64, L, d_u1, D_u1)
physicalspace(ψ_u1, 1)
```

```@example howto_states
dim(left_virtualspace(ψ_u1, 5))   # actual trimmed bond dimension ≤ dim(D_u1)
```

!!! note
    The boundary virtual spaces default to `oneunit(spacetype(d_u1))`, i.e. the charge-0 sector.
    Use the `left` and `right` keywords to target a different total charge:

    ```julia
    # state in total charge-sector +1 (one more up-spin than down-spin)
    ψ_charged = FiniteMPS(rand, ComplexF64, L, d_u1, D_u1;
                          right = Rep[U₁](1 => 1))
    ```

### Infinite MPS with U(1) symmetry

```@example howto_states
ψ_inf_u1 = InfiniteMPS(d_u1, D_u1)
physicalspace(ψ_inf_u1, 1)
```
