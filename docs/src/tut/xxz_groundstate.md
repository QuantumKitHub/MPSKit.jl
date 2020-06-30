# Analyzing the xxz model

In this file we will give step by step instructions to analyze the spin 1/2 xxz model.
The necessary packages to follow this tutorial are :
```julia
using MPSKit,TensorKit,Plots,LinearAlgebra
```

## Failure

First we should define the hamiltonian we want to work with. The following code does so in the mpo representation of the hamiltonian.
```julia
(sx,sy,sz,id) = nonsym_spintensors(1//2);
ham_data = fill(zero(id),1,5,5);
ham_data[1,1,1] = id; ham_data[1,end,end] = id;
ham_data[1,1,2] = sx; ham_data[1,2,end] = sx;
ham_data[1,1,3] = sy; ham_data[1,3,end] = sy;
ham_data[1,1,4] = sz; ham_data[1,4,end] = sz;
ham = MPOHamiltonian(ham_data);
```


We then need an intial state, which we shall later optimize. In this example we work directly in the thermodynamic limit.
```julia
random_data = TensorMap(rand,ComplexF64,ℂ^50*ℂ^2,ℂ^50);
state = InfiniteMPS([random_data]);
```

The groundstate can then be found by calling find_groundstate.
```julia
(groundstate,cache,delta) = find_groundstate(state,ham,Vumps());
```

As you can see, this struggles to converge. To understand why, we can look at schmidth decomposition of our 'groundstate'.
Our states are centergauged, which means that they can be represented as:
![](centergauge.png)

with AL and AR respectively left and right unitary. The schmidth decomposition is therefore the svd of this C matrix:
```julia
(U,S,V) = tsvd(groundstate.CR[1]);
```

We would like to plot S, but it is still a TensorMap. At the moment, you can plot the diagonal elements by calling:
```julia
S_array = convert(Array,S); #convert it to an array
S_diag = diag(S_array); #get the diagonal elements
plot(S_diag,yscale=:log10,seriestype=:scatter) #plot them
```
![](S_diag.png)


We clearly see that the dominant schmidth coefficient is doubly degenerate, implying that the groundstate of xxz can only be represented using a 2-site periodic unit cell.

## Success

Let's initialize a different initial state, this time with a 2-site unit cell:
```julia
A = TensorMap(rand,ComplexF64,ℂ^50*ℂ^2,ℂ^50);
B = TensorMap(rand,ComplexF64,ℂ^50*ℂ^2,ℂ^50);
state = InfiniteMPS([A,B]);
```

In MPSKit, we require that the periodicity of the hamiltonian equals that of the state it is applied to. This is not a big obstacle, you can simply repeat the original hamiltonian:
```julia
@assert ham.period == 1
ham = repeat(ham,2);
@assert ham.period == 2
```

Running vumps once again we get convergence:
```julia
(groundstate,cache,delta) = find_groundstate(state,ham,Vumps(maxiter=400,tol_galerkin=1e-12));
```

One may be worried that if ... - [A B] - [A B] - [A B] - ... is a groundstate, the one-site shifted version may also be a totally different groundstate. It's easy to check that this is not true.
```julia
shifted_groundstate = InfiniteMPS([groundstate.AL[2],groundstate.AL[1]]);
overlap = abs(dot(shifted_groundstate,groundstate))
@assert isapprox(overlap,1,atol=1e-1)
```
overlap is close to 1, indicating that both states are identical.

## Symmetries

The xxz hamiltonian is su(2) symmetric and we can exploit this to greatly speed up the simulation.

It is cumbersome to construct symmetric hamiltonians, but luckily su(2) symmetric xxz is already implemented:
```julia
ham = repeat(su2_xxx_ham(spin=1//2),2);
@assert ham.pspaces[1] == ℂ[SU₂](1//2 => 1)
```
Our initial state should also be su(2) symmetric. It now becomes apparant why we have to use a 2 site periodic state. The physical space carries a half-integer charge and the first tensor maps the first virtual space ⊗ the physical space to the second virtual space. Half integer virtual charges will therefore map only to integer charges, and vice versa. The staggering happens on the virtual level!

An alternative constructor for the initial state is
```julia
D1 = ℂ[SU₂](1//2 => 10,3//2=>5,5//2=>2);
D2 = ℂ[SU₂](0=>15,1=>10,2=>5);
state = InfiniteMPS([ℂ[SU₂](1//2 => 1),ℂ[SU₂](1//2 => 1)],[D1,D2])
```

Even though the bond dimension is higher then in the non symmetric example:
```julia
@assert dim(D1) == 52;
@assert dim(D2) == 70;
```

Vumps converges much much faster
```julia
(groundstate,cache,delta) = find_groundstate(state,ham,Vumps(maxiter=400,tol_galerkin=1e-12));
```
