# Basics

## TensorMap
MPSKit works on "TensorMap" objects defined in (TensorKit.jl)[https://github.com/Jutho/TensorKit.jl]. These abstract objects can represent not only plain arrays but also symmetric tensors. A TensorMap is a linear map from its domain to its codomain.

Initializing a TensorMap can be done using
```julia
TensorMap(initializer,eltype,codomain,domain);
TensorMap(inputdat,codomain,domain);
```

As an example, the following creates a random map from ℂ^10 to ℂ^10 (which is equivalent to a random matrix)
```julia
TensorMap(rand,ComplexF64,ℂ^10,ℂ^10);
dat = rand(ComplexF64,10,10); TensorMap(dat,ℂ^10,ℂ^10);
```
Similarly, the following creates a symmetric tensor
```julia
TensorMap(rand,ComplexF64,ℂ[U₁](0=>1)*ℂ[U₁](1//2=>3),ℂ[U₁](1//2=>1,-1//2=>2))
```

TensorKit defines a number of operations on TensorMap objects
```julia
a = TensorMap(rand,ComplexF64,ℂ^10,ℂ^10);
3*a; a+a; a*a; a*adjoint(a); a-a; dot(a,a); norm(a);
```

but the primary workhorse is the @tensor macro
```julia
a = TensorMap(rand,ComplexF64,ℂ^10,ℂ^10);
b = TensorMap(rand,ComplexF64,ℂ^10,ℂ^10);
@tensor c[-1;-2]:=a[-1,1]*b[1,-2];
```
creates a new TensorMap c equal to a*b.

## Creating states

Using these TensorMap building blocks we can create states; representing physical objects. An mps tensor is defined as a TensorMap from the bond dimension space (D) to bond dimension space x physical space (D x d). For example, the following creates a finite mps of length 3 :

```julia
A = TensorMap(rand,ComplexF64,ℂ^1*ℂ^2,ℂ^2);
B = TensorMap(rand,ComplexF64,ℂ^2*ℂ^2,ℂ^2);
C = TensorMap(rand,ComplexF64,ℂ^2*ℂ^2,ℂ^1);
FiniteMPS([A,B,C]);
```

Infinite matrix product states are also supported. A uniform mps representing ... ABABAB... can be created using
```julia
A = TensorMap(rand,ComplexF64,ℂ^10*ℂ^2,ℂ^10);
B = TensorMap(rand,ComplexF64,ℂ^10*ℂ^2,ℂ^10);
MPSCenterGauged([A,B]);
```

## Operators

We can act with operators on these states. A number of operators are defined, but the most commonly used one is the MPOHamiltonian. This object represents a regular 1d quantum hamiltonian and can act both on finite and infinite states. As an example, this creates the spin 1 heisenberg :
```julia
(sx,sy,sz,id) = nonsym_spintensors(1)

@tensor tham[-1 -2;-3 -4]:=sx[-1,-3]*sx[-2,-4]+sy[-1,-3]*sy[-2,-4]+sz[-1,-3]*sz[-2,-4]
ham = MPOHamiltonian(tham)
```

This code is already included in the juliatrack, just call
```julia
nonsym_xxz_ham();
```

## Algorithms

Armed with an operator and a state, we can start to do physically useful things; such as finding the groundstate:
```julia
find_groundstate(state,hamiltonian,algorithm);
```

or perform time evolution:
```julia
timestep(state,hamiltonian,dt,algorithm);
```

## Environments

We can often reuse certain environments in the algorithms, these things are stored in cache objects. The goal is that a user should not have to worry about these objects. Nevertheless, they can be created using:
```julia
params(state,opperator)
```


## Tips & tricks

- More information can be found in the documentation, provided someone writes it first.
- There is an examples folder
- Julia inference is taxed a lot; so use jupyter notebooks instead of re-running a script everytime
