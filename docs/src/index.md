```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: MPSKit.jl
  text: Matrix product states in Julia
  tagline: Efficient and versatile tools for working with matrix product states
  image:
    src: /logo.svg
    alt: MPSKit.jl
  actions:
    - theme: brand
      text: Get started
      link: /tutorials/installation
    - theme: alt
      text: Examples
      link: /examples/
    - theme: alt
      text: View on GitHub
      link: https://github.com/QuantumKitHub/MPSKit.jl

features:
  - icon: 🔗
    title: States
    details: Construct and manipulate finite and infinite matrix product states (MPS).
  - icon: 📏
    title: Observables
    details: Measure expectation values, correlations, and other observables.
  - icon: 🎯
    title: Optimization
    details: Ground states, time evolution, and excitations via DMRG, VUMPS, and more.
  - icon: ⚛️
    title: Symmetries
    details: Abelian, non-Abelian, fermionic, and anyonic symmetries out of the box.
---
```

MPSKit.jl is a Julia library for simulating one-dimensional quantum many-body systems with matrix product states and operators.
It provides both finite and infinite MPS, a range of ground-state, time-evolution, and excitation algorithms, and support for arbitrary symmetries through the [TensorKit.jl](https://github.com/Jutho/TensorKit.jl) tensor backend.
It is aimed at researchers and students who want to run tensor-network calculations without reimplementing the underlying machinery.

## Installation

MPSKit.jl is a part of the general registry.
Together with the packages used throughout this documentation, it can be installed via the
package manager as:
```
pkg> add MPSKit TensorKit TensorOperations MPSKitModels TensorKitTensors Plots
```
- `MPSKit` provides the matrix product state and operator types, together with the
  ground-state, time-evolution, and bond-dimension algorithms.
- `TensorKit` supplies the tensor backend (`TensorMap`s and vector spaces) that MPSKit is
  built on; installing it alongside MPSKit also gives access to truncation-scheme
  constructors such as `truncrank`, which TensorKit re-exports from MatrixAlgebraKit.
- `TensorOperations` provides the `@tensor` macro for contracting tensors by hand.
- `MPSKitModels` collects pre-defined Hamiltonians and local operators for common physical
  models.
- `TensorKitTensors` provides ready-made local operators, such as the Pauli operators used throughout the documentation.
- `Plots` is used to visualize results in several of the how-to guides and examples.

For a step-by-step walkthrough that sets up a dedicated environment and verifies the installation, see [Installation](@ref tutorial_installation).

## A 30-second example

Finding the ground state of the transverse-field Ising chain takes a handful of lines.

```@example index
using MPSKit, MPSKitModels, TensorKit
H = transverse_field_ising(FiniteChain(16); g = 0.5)
ψ₀ = FiniteMPS(16, ℂ^2, ℂ^4)
ψ, envs, ϵ = find_groundstate(ψ₀, H, DMRG(; verbosity = 0))
E = expectation_value(ψ, H)
```

For a guided version of this calculation that explains each step and measures more observables, see [Your first ground state](@ref tutorial_first_groundstate).

## Where next

**Tutorials** walk you through complete calculations from scratch.
Start with [Installation](@ref tutorial_installation), then run [Your first ground state](@ref tutorial_first_groundstate), and continue to [The thermodynamic limit](@ref tutorial_thermodynamic_limit) to work directly at infinite system size.

**How-to guides** are focused task recipes for when you already know what you want to do.
See [Constructing states](@ref howto_states), [Building Hamiltonians](@ref howto_hamiltonians), and [Computing observables](@ref howto_observables), among others.

**The manual** explains the concepts behind the library — the [States](@ref um_states), [Operators](@ref um_operators), and [Algorithms](@ref um_algorithms) that make up MPSKit.

**The examples** gallery collects longer, fully worked scripts covering symmetries, infinite systems, and less common algorithms; browse it at [Examples](@ref).

**The library** is the API reference; the curated, stable entry point is the [Public API](@ref public_api).

## Ecosystem

MPSKit builds on [TensorKit.jl](https://github.com/Jutho/TensorKit.jl), which supplies the tensors and vector spaces and handles the symmetries.
Models and ready-made operators come from [MPSKitModels.jl](https://github.com/QuantumKitHub/MPSKitModels.jl) and [TensorKitTensors.jl](https://github.com/QuantumKitHub/TensorKitTensors.jl).
All of these are part of the [QuantumKitHub](https://github.com/QuantumKitHub) organization; the TensorKit documentation is available [here](https://quantumkithub.github.io/TensorKit.jl/stable/).

## Community and support

Questions and bug reports are welcome on the [issue tracker](https://github.com/QuantumKitHub/MPSKit.jl/issues).
If you would like to contribute, see [CONTRIBUTING.md](https://github.com/QuantumKitHub/MPSKit.jl/blob/main/CONTRIBUTING.md) on GitHub.
