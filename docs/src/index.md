# MPSKit.jl

**Efficient and versatile tools for working with matrix product states**

## Table of contents

```@contents
Pages = ["man/intro.md","man/conventions.md","man/states.md","man/operators.md","man/algorithms.md","man/parallelism.md"]
Depth = 1
```

## Installation

MPSKit.jl is a part of the general registry, and can be installed via the package manager
as:
```
pkg> add MPSKit
```

## Key Features

- Construction and manipulation of Matrix Product States (MPS)
- Calculation of observables and expectation values
- Various optimization methods for obtaining MPS fixed points
- Support for both finite and infinite MPS
- Support for wide variety of symmetries, including Abelian, non-Abelian, fermionic and anyonic symmetries

## Usage

To get started with MPSKit, we recommend also including
[TensorKit.jl](https://github.com/Jutho/TensorKit.jl) and
[MPSKitModels.jl](https://github.com/maartenvd/MPSKitModels.jl). The former defines the
tensor backend which is used throughout MPSKit, while the latter includes some common
operators and models.

```julia
using TensorOperations
using TensorKit
using MPSKit
using MPSKitModels
using LinearAlgebra: norm
```

### Finite Matrix Product States

```@setup finitemps
using LinearAlgebra
using TensorOperations
using TensorKit
using MPSKit
using MPSKitModels
```

Finite MPS are characterised by a set of tensors, one for each site, which each have 3 legs.
They can be constructed by specifying the virtual spaces and the physical spaces, i.e. the
dimensions of each of the legs. These are then contracted to form the MPS. In MPSKit, they
are represented by `FiniteMPS`, which can be constructed either by passing in the tensors
directly, or by specifying the dimensions of the legs.

```@example finitemps
d = 2 # physical dimension
D = 5 # virtual dimension
L = 10 # number of sites

mps = FiniteMPS(L, ComplexSpace(d), ComplexSpace(D)) # random MPS with maximal bond dimension D
```

The `FiniteMPS` object then handles the gauging of the MPS, which is necessary for many of
the algorithms. This is done automatically when needed, and the user can access the gauged
tensors by getting and setting the `AL`, `AR`, `CR`/`CL` and `AC` fields, which each
represent a vector of these tensors.

```@example finitemps
al = mps.AL[3] # left gauged tensor of the third site
@tensor E[a; b] := al[c, d, b] * conj(al[c, d, a])
@show isapprox(E, id(left_virtualspace(mps, 3)))
```
```@example finitemps
ar = mps.AR[3] # right gauged tensor of the third site
@tensor E[a; b] := ar[a, d, c] * conj(ar[b, d, c])
@show isapprox(E, id(right_virtualspace(mps, 2)))
```

As the mps will be kept in a gauged form, updating a tensor will also update the gauged
tensors. For example, we can set the tensor of the third site to the identity, and the
gauged tensors will be updated accordingly.

```@example finitemps
mps.CR[3] = id(domain(mps.CR[3]))
println(mps)
```

These objects can then be used to compute observables and expectation values. For example,
the expectation value of the identity operator at the third site, which is equal to the norm
of the MPS, can be computed as:

```@example finitemps
N1 = LinearAlgebra.norm(mps)
N2 = expectation_value(mps, id(physicalspace(mps, 3)), 3)
println("‖mps‖ = $N1")
println("<mps|𝕀₃|mps> = $N2")
```

Finally, the MPS can be optimized in order to determine groundstates of given Hamiltonians.
Using the pre-defined models in `MPSKitModels`, we can construct the groundstate for the
transverse field Ising model:

```@example finitemps
H = transverse_field_ising(; J=1.0, g=0.5)
find_groundstate!(mps, H, DMRG(; maxiter=10))
E0 = expectation_value(mps, H)
println("<mps|H|mps> = $(sum(real(E0)) / length(mps))")
```

### Infinite Matrix Product States

```@setup infinitemps
using LinearAlgebra
using TensorOperations
using TensorKit
using MPSKit
using MPSKitModels
```

Similarly, an infinite MPS can be constructed by specifying the tensors for the unit cell,
characterised by the spaces (dimensions) thereof.

```@example infinitemps
d = 2 # physical dimension
D = 5 # virtual dimension
mps = InfiniteMPS(d, D) # random MPS
```

The `InfiniteMPS` object then handles the gauging of the MPS, which is necessary for many of
the algorithms. This is done automatically upon creation of the object, and the user can
access the gauged tensors by getting and setting the `AL`, `AR`, `CR`/`CL` and `AC` fields,
which each represent a (periodic) vector of these tensors.

```@example infinitemps
al = mps.AL[1] # left gauged tensor of the third site
@tensor E[a; b] := al[c, d, b] * conj(al[c, d, a])
@show isapprox(E, id(left_virtualspace(mps, 1)))
```
```@example infinitemps
ar = mps.AR[1] # right gauged tensor of the third site
@tensor E[a; b] := ar[a, d, c] * conj(ar[b, d, c])
@show isapprox(E, id(right_virtualspace(mps, 2)))
```

As regauging the MPS is not possible without recomputing all the tensors, setting a single
tensor is not supported. Instead, the user should construct a new mps object with the
desired tensor, which will then be gauged upon construction.

```@example infinitemps
als = 3 .* mps.AL
mps = InfiniteMPS(als)
```

These objects can then be used to compute observables and expectation values. For example,
the norm of the MPS, which is equal to the expectation value of the identity operator can be
computed by:

```@example infinitemps
N1 = norm(mps)
N2 = expectation_value(mps, id(physicalspace(mps, 1)), 1)
println("‖mps‖ = $N1")
println("<mps|𝕀₁|mps> = $N2")
```

!!! note "Normalization of infinite MPS"
    Because infinite MPS cannot sensibly be normalized to anything but $1$, the `norm` of
    an infinite MPS is always set to be $1$ at construction. If this were not the case, any
    observable computed from the MPS would either blow up to infinity or vanish to zero.

Finally, the MPS can be optimized in order to determine groundstates of given Hamiltonians.
Using the pre-defined models in `MPSKitModels`, we can construct the groundstate for the
transverse field Ising model:

```@example infinitemps
H = transverse_field_ising(; J=1.0, g=0.5)
mps, = find_groundstate(mps, H, VUMPS(; maxiter=10))
E0 = expectation_value(mps, H)
println("<mps|H|mps> = $(sum(real(E0)) / length(mps))")
```

### Additional Resources

For more detailed information on the functionality and capabilities of MPSKit, refer to the
Manual section, or have a look at the [Examples](@ref) page.

Keep in mind that the documentation is still a work in progress, and that some features may
not be fully documented yet. If you encounter any issues or have questions, please check the
library's [issue tracker](https://github.com/maartenvd/MPSKit.jl/issues) on the GitHub
repository and open a new issue.