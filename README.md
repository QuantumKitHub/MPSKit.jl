# MPSKit.jl

Contains code for tackling 1 dimensional (quantum) problems using tensor network algorithms.
Very early beta - things may break and the documentation is a stub!

## Basics

### TensorMap
mpskit works on "TensorMap" objects defined in TensorKit.jl (a different package). These abstract objects can represent not only plain arrays but also symmetric tensors. A TensorMap is a linear map from its domain to its codomain.

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

## states
### Uniform Mps

An mps tensor is defined as a TensorMap from the bond dimension space (D) to bond dimension space x physical space (D x d). A uniform mps representing ... ABABAB... can be created using
```julia
A = TensorMap(rand,ComplexF64,ℂ^10*ℂ^2,ℂ^10);
B = TensorMap(rand,ComplexF64,ℂ^10*ℂ^2,ℂ^10);
MpsCenterGauged([A,B]);
```

This MpsCenterGauged structure has the fields AL,AR,AC,CR where AL[i]CR[i]=CR[i-1]AR[i]=AC[i] and AL/AR is left/right unitary.

### FiniteMps

A finite mps is an array of mps tensors starting with bond dimension 1 and ending with bond dimension 1
```julia
A = TensorMap(rand,ComplexF64,ℂ^1*ℂ^2,ℂ^2);
B = TensorMap(rand,ComplexF64,ℂ^2*ℂ^2,ℂ^2);
C = TensorMap(rand,ComplexF64,ℂ^2*ℂ^2,ℂ^1);
FiniteMps([A,B,C]);
```

### MpsComoving

When doing local quenches in a uniform state, we only need to time evolve a "window" of tensors. MpsComoving(state_left,window,state_right) creates such a set of states, and can be passed to find_groundstate() or timestep(). Another possible use is to study the effect of impurities.

## Operators
### MpoHamiltonian

MpoHamiltonian is a hamiltonian in the mpo representation. As an example, this creates the spin 1 heisenberg :
```julia
(sx,sy,sz,id) = nonsym_spintensors(1)

@tensor tham[-1 -2;-3 -4]:=sx[-1,-3]*sx[-2,-4]+sy[-1,-3]*sy[-2,-4]+sz[-1,-3]*sz[-2,-4]
ham = MpoHamiltonian(tham)
```

This code is already included in the juliatrack, just call
```julia
nonsym_xxz_ham();
```
### periodic mpo

PeriodicMpo is a periodic nxm array of matrix product operators, intended to be used for classical statmech problems.

## Algorithms
### find groundstate

To find the groundstate of a given state (be it a uniform mps or a finite mps), just call
```julia
find_groundstate(state,hamiltonian,algorithm);
```

where algorithm can be Vumps(),Dmrg(),.... As an example, the following code finds the groundstate of spin-1 heisenberg:
```julia
ham = nonsym_xxz_ham();
st = MpsCenterGauged([TensorMap(rand,ComplexF64,ComplexSpace(10)*ComplexSpace(3),ComplexSpace(10))]);
find_groundstate(st,ham,Vumps());
```

### timestep


```julia
timestep(state,hamiltonian,dt,algorithm);
```

evolves state forward in time by dt. Algorithm can be either Tdvp() or Tdvp2().


## Tips & tricks

- More information can be found in the documentation, provided someone writes it first.
- There is an examples folder
- Julia inference is taxed a lot; so use jupyter notebooks instead of re-running a script everytime
