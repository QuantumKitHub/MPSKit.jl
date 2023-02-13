# MPSKit.jl

[![docs][docs-dev-img]][docs-dev-url] [![codecov][codecov-img]][codecov-url] ![CI][ci-url]

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://maartenvd.github.io/MPSKit.jl/dev/

[codecov-img]: https://codecov.io/gh/maartenvd/MPSKit.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/maartenvd/MPSKit.jl

[ci-url]: https://github.com/maartenvd/MPSKit.jl/workflows/CI/badge.svg

Contains code for tackling 1 dimensional quantum (and 2d classical) problems using tensor network algorithms. While it is still in beta, exported algorithms should just work. If you encounter an issue, feel free to open a bug report.

We implemented different algorithms for finding the groundstate (both finite and infinite systems), performing time evolution, finding excitations and much more. Check out the [tutorials](https://maartenvd.github.io/MPSKit.jl/dev/#Tutorials-1) or [examples](https://github.com/maartenvd/MPSKit.jl/tree/master/examples) (the documentation itself is still quite terse).

## Installation

First, install this package by opening julia and pressing "]". Then type

```julia
pkg> add MPSKit
```

MPSKit works on Tensormap objects, which are defined in [another package](https://github.com/Jutho/TensorKit.jl).
You will have to add this pacakge as well to create the basic building blocks.
```julia
pkg> add TensorKit
```

Last but not least, we have already implemented a few hamiltonians in [MPSKitModels.jl](https://github.com/maartenvd/MPSKitModels.jl). It is recommended to install this package too.
```julia
pkg> add MPSKitModels
```

## Quickstart

After following the installation process, you should now be able to call
```julia
julia> using MPSKit,MPSKitModels,TensorKit
```

You can create a random 1 site periodic infinite mps (bond dimension 10) by calling
```julia
julia> state = InfiniteMPS([ℂ^2],[ℂ^10]);
```

We can use a pre-defined hamiltonian from MPSKitModels
```julia
julia> hamiltonian = nonsym_ising_ham();
```

And find the groundstate
```julia
julia> (groundstate,_) = find_groundstate(state,hamiltonian,VUMPS());
```
