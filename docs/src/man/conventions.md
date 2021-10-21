# Conventions

MPSKit is build on top of TensorKit, which in turn defines general (codomain ← domain) tensormaps.
It is necessary to concretely define what we should consider an mps tensor (for example - what leg is its physical leg?), a bond tensor, and an mpo tensor.

## MPS tensors

The general definition of an mps tensor is as follows

![](mps_tensor_definition.png)

mps tensors are well defined for any amount of physical legs and InfiniteMPS/FiniteMPS handle these general N-leg tensors just fine. This is necessary in for example peps code, where the number of physical spaces of a boundary mps tensor is 2.

However our operators MPOHamiltonian/DenseMPO both only work on mps tensors with one leg. There is no fundamental problem, anyone is free to implement new operators working on the more general mps tensors.

## Bond tensors

The mps bond tensors are square tensormaps, mapping a virtual space to the same virtual space.
![](bond_tensor_definition.png)

## MPO tensors

MPO tensors are used in both statistical mechanics problems and in the definition of quantum hamiltonians. Graphically, they can be represented as follows

![](mpo_tensor_definition.png)

The numbering of spaces results in an at first glance not very intuitive ordering of (virtual,physical,physical,virtual) spaces.

## Planarity

MPSKit tries to be as planar as possible, braidings are avoided as often as possible. As a result, it can handle anyonic/fermionic symmetries. Behind the scenes we always check if a contraction is done with tensors for which the braiding rules are trivial - and if not we use TensorKit's `@planar` macro. The logic of this switching between normal and planar tensor contractions is contained in a private `@plansor` macro. This is mostly an implementation detail, and the user should not have to worry about this.
