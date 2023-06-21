# [States](@id um_states)


## FiniteMPS

A FiniteMPS is - at its core - a chain of mps tensors.

![](finite_mps_definition.png)

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

### Gauging

There is residual gauge freedom in such a finite mps :

![](mps_gauge_freedom.png)

which is often exploited in mps algorithms. The gauging logic is handled behind the scenes, if you call

```julia
state.AL[3]
```

then the state will be gauged such that the third tensor is a left isometry (similarly for state.AR).

```julia
state.AC[3]
```
gauges the state in such a way that all tensors to the left are left isometries, and to the right will be right isometries.As a result you should have

```julia
norm(state) == norm(state.AC[3])
```

lastly there is also the CR field, with the following property:

```julia
@tensor a[-1 -2;-3] := state.AL[3][-1 -2;1]*state.CR[3][1;-3]
@tensor b[-1 -2;-3] := state.CR[2][-1;1]*state.AR[3][1 -2;-3]
a ≈ state.AC[3];
b ≈ state.AC[3];
```

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

An infinite mps can be thought of as a finite mps, where the set of tensors is repeated periodically.

It can be created by passing in a vector of tensormaps:
```julia
data = fill(TensorMap(rand,ComplexF64,ℂ^10*ℂ^2,ℂ^10),2);
InfiniteMPS(data);
```

The above code would create an infinite mps with an A-B structure (a 2 site unit cell).

much like a finite mps, we can again query the fields state.AL, state.AR, state.AC and state.CR. The implementation is much easier, as they are now just plain fields in the struct

```julia
AL::PeriodicArray{A,1}
AR::PeriodicArray{A,1}
CR::PeriodicArray{B,1}
AC::PeriodicArray{A,1}
```

The periodic array is an array-like type where all indices are repeated periodically.

## WindowMPS

WindowMPS is a bit of a mix between an infinite mps and a finite mps. It represents a window of mutable tensors embedded in an infinite mps.

It can be created using:
```julia
mpco = WindowMPS(left_infinite_mps,window_of_tensors,right_infinite_mps)
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
const MPOMultiline = Multiline{<:DenseMPO}
