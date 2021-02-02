# [States](@id um_states)


## FiniteMPS

### Usage
A [`FiniteMPS`](@ref) can be created by passing in a vector of tensormaps:

```julia
data = fill(TensorMap(rand,ComplexF64,ℂ^1*ℂ^2,ℂ^1),10);
FiniteMPS(data);
```

Or alternatively by
```julia
len = 10;
max_bond_dimension = ℂ^10;
physical_space = ℂ^2;
FiniteMPS(rand,ComplexF64,len,physical_space,max_bond_dimension);
```

You can take dot products, renormalize!, expectation values,....

In our algorithms we typically make use of the fields .AC/.AR/.AL. Calling
```julia
state.AC[3]
```
gauges the state such that every tensor left of 3 is a left unitary matrix and to the right we have right unitary matrices.
As a result you should have

```julia
norm(state) == norm(state.AC[3])
```

Note that every tensor should be a map from the virtual space to the virtual space ⊗ physical space.
In other words, we need the input tensormaps to be of the type AbstractTensorMap{S,N,1}.

### Implementation details

Behind the scenes, a finite mps has 4 fields
```julia
ALs::Vector{Union{Missing,A}}
ARs::Vector{Union{Missing,A}}
ACs::Vector{Union{Missing,A}}
CLs::Vector{Union{Missing,B}}
```

calling state.AC returns an "orthoview" instance, which is a very simple dummy object. When you call get/setindex on an orthoview, it will move the gauge for the underlying state, and return the result. The idea behind this construction is that one never has to worry about how the state is gauged, as this gets handled automagically.

The following bit of code shows the logic in action:

```julia
state = FiniteMPS(10,ℂ^2,ℂ^10); # a random initial state
@show ismissing.(state.ALs) # all AL fields are already calculated
@show ismissing.(state.ARs) # all AR fields are missing

#if we now query state.AC[2], it should calculate and store all AR fields left of position 2
state.AC[2];
@show ismissing.(state.ARs)
```

## InfiniteMPS

An infinite mps can be created by passing in a vector of tensormaps:
```julia
data = fill(TensorMap(rand,ComplexF64,ℂ^10*ℂ^2,ℂ^10),2);
InfiniteMPS(data);
```

The above code would create an infinite mps with an A-B structure (a 2 site unit cell).

an infinite mps has 4 fields:

```julia
AL::PeriodicArray{A,1}
AR::PeriodicArray{A,1}
CR::PeriodicArray{B,1}
AC::PeriodicArray{A,1}
```

## MPSComoving

MPSComoving is a bit of a mix between an infinite mps and a finite mps. It represents a window of mutable tensors embedded in an infinite mps.

It can be created using:
```julia
mpco = MPSComoving(left_infinite_mps,window_of_tensors,right_infinite_mps)
```

Algorithms will then act on this window of tensors, while leaving the left and right infinite mps's invariant.

Behind the scenes it uses the same orthoview logic as finitemps.

## Multiline

Statistical physics partition functions can be represented by an infinite tensor network which then needs to be contracted.
This is done by finding approximate fixpoint infinite matrix product states.
However, there is no good reason why a single mps should suffice and indeed we find in practice that this can require a nontrivial unit cell in both dimensions.

In other words, the fixpoints can be well described by a set of matrix product states.

Such a set can be created by

```julia
data = fill(TensorMap(rand,ComplexF64,ℂ^10*ℂ^2,ℂ^10),2,2);
MPSMultiline(data);
```
MPSMultiline is also used extensively in as of yet unreleased peps code.

You can access properties by calling
```julia
state.AL[row,collumn]
state.AC[row,collumn]
state.AR[row,collumn]
state.CR[row,collumn]
```

Behind the scenes, we have a type called Multiline, defined as:

```julia
struct Multiline{T}
    data::PeriodicArray{T,1}
end
```

MPSMultiline/MPOMultiline are then defined as
```julia
const MPSMultiline = Multiline{<:InfiniteMPS}
const MPOMultiline = Multiline{<:InfiniteMPO}
