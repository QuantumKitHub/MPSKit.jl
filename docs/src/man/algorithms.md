# [Algorithms](@id um_algorithms)

## minimizing the energy

```julia
state = FiniteMPS(rand,ComplexF64,20,ℂ^2,ℂ^10);
operator = nonsym_ising_ham();
(groundstate,environments,delta) = find_groundstate(state,operator,Dmrg())
```

will use dmrg to minimize the energy. Sometimes it can be useful to do more extensive logging or to perform dynamical bond dimension expansion. That's why the Dmrg() constructor allows you to specify a finalize function
```julia
function finalize(iter,state,ham,pars)
    println("Hello from iteration $iter")
    return state,pars;
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
