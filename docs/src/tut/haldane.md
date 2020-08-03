# [The haldane gap](@id tut_haldane)

In this tutorial we will calculate the haldane gap (the energy gap in spin 1 heisenberg) in 2 different ways. To follow the tutorial you need the following packages.

```julia
using MPSKit,TensorKit
```

We will enforce the su(2) symmetry, our hamiltonian will therefore be

```julia
ham = su2_xxx_ham(spin=1);
```

## Finite size extrapolation

The first step is always the same, we want to find the groundstate of our system.
```julia
len = 10;
physical_space = ℂ[SU₂](1=>1);
virtual_space = ℂ[SU₂](0=>20,1=>20,2=>10,3=>10,4=>5);

initial_state = FiniteMPS(rand,ComplexF64,len,physical_space,virtual_space);
(gs,pars,delta) = find_groundstate(initial_state,ham,Dmrg());
```

The typical way to find excited states is to minmize the energy while adding an error term ``lambda | gs > < gs | ``. Here we will instead use the [quasiparticle ansatz](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.111.080401).

In steven white's original DMRG paper it was remarked that the S=1 excitations correspond to edge states, and that one should define the haldane gap as the difference in energy between the S=2 and S=1 states. This can be done as follows.

```julia
(En_1,_) = quasiparticle_excitation(ham,gs,pars,excitation_space = ℂ[SU₂](1=>1))
(En_2,_) = quasiparticle_excitation(ham,gs,pars,excitation_space = ℂ[SU₂](2=>1))
En_2[1]-En_1[1]
```

We can now extrapolate this value for different len, and approximately find the haldane gap.

![](haldane_finite.png)

## Thermodynamic limit

A much nicer way of obtaining the haldane gap is by working directly in the thermodynamic limit.

```julia
initial_state = InfiniteMPS([physical_space],[virtual_space]);
(gs,pars,delta) = find_groundstate(initial_state,ham,Vumps());
```

The haldane gap in the thermodynamic limit is the energy of the S=1 excitation at momentum pi.

```julia
(Energies,_) = quasiparticle_excitation(ham,Float64(pi),gs,pars,excitation_space=ℂ[SU₂](1=>1))
Energies[1]
```

Which prints out 0.4104791728966182, in agreement with known results.
