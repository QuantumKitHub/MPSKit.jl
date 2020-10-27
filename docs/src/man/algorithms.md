# [Algorithms](@id um_algorithms)

## minimizing the energy

```julia
state = FiniteMPS(rand,ComplexF64,20,ℂ^2,ℂ^10);
operator = nonsym_ising_ham();
(groundstate,environments,delta) = find_groundstate(state,operator,Dmrg())
```

will use dmrg to minimize the energy. Sometimes it can be useful to do more extensive logging or to perform dynamical bond dimension expansion. That's why the Dmrg() constructor allows you to specify a finalize function
```julia
function finalize(iter,state,ham,envs)
    println("Hello from iteration $iter")
    return state,envs;
end

Dmrg(finalize=my_finalize)
```

Similar functionality is provided (or soon to be implemented) in other groundstate algorithms. Other algorithms are provided and can be found in the [library documentation](@ref lib_gs_alg).

## timestep

```julia
state = FiniteMPS(rand,ComplexF64,20,ℂ^2,ℂ^10);
operator = nonsym_ising_ham();
(newstate,environments) = timestep(state,operator,0.3,Tdvp2(trscheme=truncdim(20)))
```

will evolve 'state' forwards in time by 0.3 seconds. Here we use a 2 site update scheme, which will truncate the 2site tensor back down, truncating at bond dimension 20. An overview of all time evolution algorithms is in the [library documentation](@ref lib_time_alg).

## dynamicaldmrg

Dynamical dmrg has been described in other papers and is a way to find the propagator. The basic idea is that to calculate ``G(z) = < V | (H-z)^{-1} | V > `` , one can variationally find ``(H-z) |W > = | V > `` and then the propagator simply equals ``G(z) = < V | W >``.

## quasiparticle excitations

We export code that implements the [quasiparticle excitation ansatz](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.111.080401) for finite and infinite systems.
For example, the following calculates the haldane gap for spin-1 heisenberg.

```julia
th = nonsym_xxz_ham()
ts = InfiniteMPS([ℂ^3],[ℂ^48]);
(ts,envs,_) = find_groundstate(ts,th,Vumps(maxiter=400,verbose=false));
(energies,Bs) = quasiparticle_excitation(th,Float64(pi),ts,envs);
@test energies[1] ≈ 0.41047925 atol=1e-4
```

For infinite systems you have to specify the momentum of your particle. In contrast, momentum is not a well defined quantum number and you therefore do not have to specify it when finding excitations on top of a finite mps.

## fidelity susceptibility

The fidelity susceptibility measures how much the groundstate changes when tuning a parameter in your hamiltonian. Divergences occur at phase transitions, making it a valuable measure when no order parameter is known.

```julia
suscept = fidelity_susceptibility(groundstate,H_0,perturbing_Hams::AbstractVector)
```
