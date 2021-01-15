# [Algorithms](@id um_algorithms)

## minimizing the energy

### dmrg

```julia
state = FiniteMPS(20,ℂ^2,ℂ^10);
operator = nonsym_ising_ham();
(groundstate,environments,delta) = find_groundstate!(state,operator,Dmrg())
```

The dmrg algorithm sweeps through the system, optimizing every site. This - on it's own - cannot increase the bond dimension. If you do want to increase the bond dimension dynamically, then there are two options. Either you use the two-site variant of dmrg, aptly called Dmrg2(), or you make use of the finalize option. Finalize is a function that gets called every iteration of dmrg and can modify the state.

```julia
function my_finalize(iter,state,ham,envs)
    println("Hello from iteration $iter")
    return state,envs;
end

(groundstate,environments,delta) = find_groundstate!(state,operator,Dmrg(finalize = my_finalize))
```

### vumps

Vumps is a dmrg inspired algorithm that can be used to find the groundstate of infinite matrix product states
```julia
state = InfiniteMPS([ℂ^2],[ℂ^10]);
operator = nonsym_ising_ham();
(groundstate,environments,delta) = find_groundstate(state,operator,Vumps())
```

much like dmrg, it cannot modify the bond dimension, and this has to be done manually in the finalize function.

### gradient descent

Both finite and infinite matrix product states can be parametrized by a set of unitary matrices, and we can then perform gradient descent on this unitary manifold. Due to some technical reasons (gauge freedom), this manifold further restricts to a grassmann manifold.

```julia
state = InfiniteMPS([ℂ^2],[ℂ^10]);
operator = nonsym_ising_ham();
(groundstate,environments,delta) = find_groundstate(state,operator,GradientGrassmann())
```

### Idmrg

We export a toy implementation of one-site idmrg. It converges slower then vumps, but is more reliable. Two-site idmrg support is planned in the future.

## time evolution


### Tdvp

There is an implementation of the one-site tdvp scheme for finite mps and infinite mps:
```julia
(newstate,environments) = timestep(state,operator,dt,Tdvp())
```

and the two-site scheme for finite mps (Tdvp2()). Similarly to dmrg, the one site scheme will preserve the bond dimension, and expansion has to be done manually.

### Time evolution mpo

We have rudimentary support for turning an mpo hamiltonian into a time evolution mpo.

```julia
make_time_mpo(ham,dt,alg::WI)
make_time_mpo(ham,dt,alg::WII)
```

two algorithms are available, corresponding to different orders of precision. It is possible to then multiply a state by this mpo, or to approximate (mpo,state) by a new state

```julia
state = InfiniteMPS([ℂ^2],[ℂ^10]);
operator = nonsym_ising_ham();
mpo = make_time_mpo(operator,0.1,WII());
approximate(state,(state,mpo),Vumps())
```

This feature is at the moment not very well supported.

## excitations

### Quasiparticle ansatz

We export code that implements the [quasiparticle excitation ansatz](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.111.080401) for finite and infinite systems.
For example, the following calculates the haldane gap for spin-1 heisenberg.

```julia
th = nonsym_xxz_ham()
ts = InfiniteMPS([ℂ^3],[ℂ^48]);
(ts,envs,_) = find_groundstate(ts,th,Vumps(maxiter=400,verbose=false));
(energies,Bs) = excitations(th,QuasiparticleAnsatz(),Float64(pi),ts,envs);
@test energies[1] ≈ 0.41047925 atol=1e-4
```

For infinite systems you have to specify the momentum of your particle. In contrast, momentum is not a well defined quantum number and you therefore do not have to specify it when finding excitations on top of a finite mps.

### Finite excitations

For finite systems we can also do something else - find the groundstate of the hamiltonian + ``weight \sum_i | psi_i > < psi_i ``. This is also supported by calling

```julia
th = nonsym_ising_ham()
ts = FiniteMPS(10,ℂ^2,ℂ^12);
(ts,envs,_) = find_groundstate(ts,th,Dmrg(verbose=false));
(energies,Bs) = excitations(th,FiniteExcited(),ts,envs);
```

## changebonds

### optimal expand

One possible way to expand the bond dimension is described in the [original vumps paper](https://arxiv.org/abs/1701.07035). The idea is to look at the 2site derivative and add the most important blocks orthogonal to the current mps. From the point of view of a local 2site update, this procedure is 'optimal'.

The state will remain physically unchanged, but a one-site scheme will now be able to push the optimization further.

```julia
th = nonsym_ising_ham()
ts = FiniteMPS(10,ℂ^2,ℂ^12);
changebonds(ts,OptimalExpand(trscheme = truncdim(1))) # expand the bond dimension by 1
```

### svd cut

It is possible to truncate a state using the svd decomposition, this is implemented in svdcut.

```julia
th = nonsym_ising_ham()
ts = FiniteMPS(10,ℂ^2,ℂ^12);
changebonds(ts,SvdCut(trscheme = truncdim(10))) # truncate the state to one with bond dimension 10
```

### vumps svd cut

A particularly simple scheme useful when doing vumps is to do a 2site update, and then truncating this back down. It changes the state itself, so cannot be used to do time evolution, but that is no problem for energy minimization.

## dynamicaldmrg

Dynamical dmrg has been described in other papers and is a way to find the propagator. The basic idea is that to calculate ``G(z) = < V | (H-z)^{-1} | V > `` , one can variationally find ``(H-z) |W > = | V > `` and then the propagator simply equals ``G(z) = < V | W >``.


## fidelity susceptibility

The fidelity susceptibility measures how much the groundstate changes when tuning a parameter in your hamiltonian. Divergences occur at phase transitions, making it a valuable measure when no order parameter is known.

```julia
suscept = fidelity_susceptibility(groundstate,H_0,perturbing_Hams::AbstractVector)
```

## periodic boundary conditions

You can impose periodic boundary conditions on the hamiltonian itself, while still using a normal OBC finite matrix product states. This is straightforward to implement but competitive with more advanced PBC mps algorithms.

## exact diagonalization

As a side effect, our code support exact diagonalization. The idea is to construct a finite matrix product state with maximal bond dimension, and then optimize the middle site. Because we never truncated the bond dimension, this single site effectively parametrizes the entire hilbert space.

```julia
exact_diagonalization(periodic_boundary_conditions(su2_xxx_ham(spin=1),10)) # find the groundstate on 10 sites
```

## leading boundary

For statmech partition functions we want to find the approximate leading boundary mps. Again this can be done with vumps:

```julia
th = nonsym_ising_mpo()
ts = InfiniteMPS([ℂ^2],[ℂ^20]);
(ts,envs,_) = leading_boundary(ts,th,Vumps(maxiter=400,verbose=false));
```
