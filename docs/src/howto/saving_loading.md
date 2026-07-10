# [Saving and loading](@id howto_saving_loading)

The examples on this page use MPSKit.jl and TensorKit.jl.
See [Installation](@ref tutorial_installation) for how to add these packages to your environment.

MPSKit does **not** ship its own save/load functions.
States such as [`FiniteMPS`](@ref) and [`InfiniteMPS`](@ref) are ordinary Julia objects that wrap TensorKit `TensorMap`s, so any general-purpose Julia serializer stores and restores them.
Two options cover essentially all use cases:

- **`Serialization`** — a standard-library module, always available, no extra dependency.
  Best for quick "save a result and pick it up in the next session" workflows.
  Its file format is **not** guaranteed stable across Julia or package versions (see [Caveats](@ref howto_saving_loading_caveats)).
- **[JLD2.jl](https://github.com/JuliaIO/JLD2.jl)** — a widely used HDF5-compatible format with named datasets and better long-term robustness.
  Recommended when you want to archive data or share files between machines.
  JLD2 is a separate package you must add with `] add JLD2`.

The runnable recipes below use `Serialization` so they execute with no extra dependency.
The JLD2 variants are shown separately and are equivalent in what they store.

```@example saveload
using MPSKit, TensorKit
using Serialization
```

---

## 1. Save and reload a finite MPS

[`serialize`](https://docs.julialang.org/en/v1/stdlib/Serialization/#Serialization.serialize) writes any object to a file; `deserialize` reads it back.
Here we write to a temporary path and check that the reloaded state is identical by taking the overlap `⟨ψ | ψ_loaded⟩`, which is `1` (up to rounding) when the two states coincide.

```@example saveload
ψ = FiniteMPS(10, ℂ^2, ℂ^16)

path = tempname()          # a fresh temporary file path
serialize(path, ψ)

ψ_loaded = deserialize(path)
abs(dot(ψ, ψ_loaded))      # ≈ 1: the reloaded state equals the original
```

The reloaded object is a genuine [`FiniteMPS`](@ref), ready for any further computation:

```@example saveload
ψ_loaded isa FiniteMPS
```

Nothing here is specific to a *random* state — the same holds for a state returned by [`find_groundstate`](@ref) or [`timestep`](@ref).
Save the state you actually care about the moment you have it.

## 2. Save and reload an infinite MPS

[`InfiniteMPS`](@ref) works exactly the same way.
Because an infinite state is normalized by its gauge, the overlap check above is not the natural diagnostic; instead compare the gauged tensors directly.

```@example saveload
ψ∞ = InfiniteMPS(ℂ^2, ℂ^16)

path∞ = tempname()
serialize(path∞, ψ∞)

ψ∞_loaded = deserialize(path∞)
ψ∞_loaded.AL[1] ≈ ψ∞.AL[1]   # left-gauged tensors match
```

## 3. Store several objects together

To keep a state alongside metadata (parameters, a description, the energy you measured), serialize a `NamedTuple` or `Dict` in one file.
This keeps everything that belongs together in a single artifact.

```@example saveload
result = (state = ψ, χ = 16, note = "TFIM ground state")

path_result = tempname()
serialize(path_result, result)

back = deserialize(path_result)
back.note
```

```@example saveload
abs(dot(back.state, ψ))     # the embedded state round-trips too
```

## 4. The JLD2 variant

[JLD2.jl](https://github.com/JuliaIO/JLD2.jl) stores objects under string keys and is the more portable choice for archival data.
Add it with `] add JLD2` first.
The following is equivalent to the `Serialization` recipes above; it is not executed here because JLD2 is not a dependency of this documentation build.

```julia
using JLD2

# save one or more named objects
jldsave("state.jld2"; ψ, χ = 16, note = "TFIM ground state")

# load them back by name
ψ_loaded = load("state.jld2", "ψ")
note     = load("state.jld2", "note")
```

Symmetric states round-trip through JLD2 without any extra work: the `TensorMap`s carry their own symmetry sectors and vector spaces, so a state built on, e.g., `Z2Space` is restored with its full symmetry structure intact.

---

## Environments

Cached [environments](@ref concept_environments) (the objects returned by `environments`, held inside the value from [`find_groundstate`](@ref) and friends) are serializable in exactly the same way as states — they are also just tensors.
In practice, however, **it is usually not worth saving them**: environments are derived data, tied to one specific state, and recomputing them from a stored state is cheap compared to the optimization that produced the state.

The recommended workflow is therefore to save only the *state* and rebuild the environments after loading:

```@example saveload
using MPSKitModels    # for the Hamiltonian
H = transverse_field_ising(FiniteChain(10))

envs = environments(ψ_loaded, H, ψ_loaded)   # rebuilt from the reloaded state
nothing # hide
```

<!-- REVIEW: physics/perf claim — asserting that recomputing environments is "cheap compared to the optimization" and that saving them is "usually not worth it". True for a converged ground state (one sweep of environment builds vs. many optimization sweeps), but I have not benchmarked the crossover; for very large bond dimensions or expensive contractions a maintainer may want to soften or qualify this. -->

If you do have a reason to persist environments (e.g. to resume an expensive iterative build), `serialize`/`deserialize` them just like a state.

## [Caveats](@id howto_saving_loading_caveats)

- **Version compatibility.**
  `Serialization` files are **not** guaranteed to be readable by a different Julia version, nor after MPSKit or TensorKit change their internal type layout.
  Treat `Serialization` output as a scratch artifact within one environment; use **JLD2** for anything you need to reopen weeks later or on another machine.
  <!-- REVIEW: this restates the documented behavior of the stdlib Serialization module (its own docs warn the format is not stable across versions); please confirm the phrasing matches what you want to promise about JLD2's cross-version robustness, which is strong but also not unconditional. -->

- **Symmetric tensors are self-describing.**
  A saved state carries the vector spaces and symmetry sectors of every tensor, so you do not need to record the symmetry separately — loading reconstructs the full space structure.
  This was verified for a `Z2`-symmetric state round-tripping through both `Serialization` and JLD2.

- **File size.**
  A stored state is roughly the size of its tensors, which grows with the bond dimension (and, for symmetric states, the sector structure).
  For large-bond-dimension states these files can be substantial; write them to scratch/bulk storage rather than a quota-limited home directory, and consider saving only the final state rather than every intermediate.

- **What to save.**
  Prefer saving the state (and the parameters needed to rebuild its Hamiltonian) over saving derived caches like environments.
  A state plus its model definition is enough to reconstruct everything else.
