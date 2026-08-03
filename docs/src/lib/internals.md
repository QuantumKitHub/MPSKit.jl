# [Internals](@id lib_internals)

!!! warning "Non-public API"
    The symbols documented on this page are **internal**: they are unexported, not
    part of the public API, and may change or be removed in any release without notice
    or a deprecation cycle. They are collected here as a reference for contributors and
    advanced users reading the source, not as a stable interface to build on. For the
    supported surface see the [Public API](@ref public_api). Experimental features (for
    example the current GPU support, discussed in [Parallelism and GPU support](@ref howto_parallelism_gpu)) are
    likewise unstable and subject to change.

```@meta
CurrentModule = MPSKit
```

## Effective (derivative) operators

The local eigenvalue and time-evolution problems solved by DMRG, VUMPS, TDVP and friends are phrased in terms of effective "derivative" operators acting on a single gauge tensor.
These are built internally from the Hamiltonian and the surrounding [environments](@ref lib_environments).

```@docs; canonical=false
DerivativeOperator
C_hamiltonian
AC_hamiltonian
AC2_hamiltonian
```

## Transfer matrices

Low-level application of (regularized) transfer matrices to boundary vectors, used when building infinite-MPS environments.

```@docs; canonical=false
transfer_left
transfer_right
```

## Environment algorithm resolution

Helpers that pick and instantiate the iterative solver used to compute environments for a given bra/operator/ket combination.

```@docs; canonical=false
environment_alg
resolve_environment_solver
```

## Defaults and scheduling

Global configuration lives in the `MPSKit.Defaults` submodule, including the multi-threading scheduler used across the package.

```@docs
Defaults
Defaults.set_scheduler!
```
