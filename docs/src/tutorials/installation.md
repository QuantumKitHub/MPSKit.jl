# [Installation](@id tutorial_installation)

This page walks you through setting up a Julia environment for working with MPSKit.jl, and ends with a small snippet you can run to check that everything works.

## Prerequisites

You need a working installation of Julia, version 1.10 or later.
If you don't have Julia yet, install it via [juliaup](https://github.com/JuliaLang/juliaup) or download it directly from [julialang.org](https://julialang.org/downloads/).
This tutorial assumes you are comfortable starting the Julia REPL and typing commands into it, but does not assume any prior experience with Julia's package manager.

## Set up a project environment

Before installing any packages, create a dedicated environment for this tutorial.
Working in a fresh, named environment (rather than the global default environment) keeps the exact package versions you use here reproducible, and avoids clashes with other projects on your machine.

Start Julia, enter the package manager by pressing `]`, and activate a new environment:

```
pkg> activate mpskit-tutorial
```

Julia will create the environment the first time you add a package to it.

## Install the packages

With the environment activated, install MPSKit.jl and the packages used throughout this documentation:

```
pkg> add MPSKit TensorKit TensorOperations MPSKitModels TensorKitTensors Plots
```

- `MPSKit` provides the matrix product state and operator types, together with the ground-state, time-evolution, and bond-dimension algorithms.
- `TensorKit` supplies the tensor backend (`TensorMap`s and vector spaces) that MPSKit is built on; installing it alongside MPSKit also gives access to truncation-scheme constructors such as `truncrank`, which TensorKit re-exports from MatrixAlgebraKit.
- `TensorOperations` provides the `@tensor` macro used to contract tensors by hand.
- `MPSKitModels` collects pre-defined Hamiltonians (such as the transverse-field Ising model) and lattices for common physical models.
- `TensorKitTensors` provides ready-made local operators, such as the Pauli operators.
- `Plots` is used to visualize results in several of the how-to guides and examples; it is optional if you only intend to run computations without plotting.

MPSKit.jl is registered in Julia's General registry, so `pkg> add` fetches it directly; you do not need to add any custom registries.

!!! note "First `using` is slow"
    The first time you load these packages with `using`, Julia precompiles them, which can take a minute or two.
    Subsequent loads in the same environment are much faster.

## Verify your setup

Once the packages have finished installing, exit the package manager (backspace) and run the following in the same environment to check that MPSKit, TensorKit, and MPSKitModels work together.

```@example verify-install
using MPSKit, TensorKit, MPSKitModels

H = transverse_field_ising(FiniteChain(8); J = 1.0, g = 0.5)
ψ = FiniteMPS(8, ℂ^2, ℂ^8)
```

If this runs without error and prints a `FiniteMPS`, your environment is ready.

From here, continue with [Your first ground state](@ref tutorial_first_groundstate), which uses this same Hamiltonian and initial state to find the ground state of the transverse-field Ising model with DMRG.

## Troubleshooting

- **Long precompilation on first use:** this is expected the first time you `using` a package (or after updating one), especially for a large dependency stack; it is not a sign that anything is wrong.
- **Version resolver conflicts:** if `pkg> add` reports that it cannot find a compatible set of versions, try creating a fresh environment (as above) rather than adding these packages to an existing environment that already has other constraints.
