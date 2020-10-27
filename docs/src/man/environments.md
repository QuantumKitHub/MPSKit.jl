# [Environments](@id um_environments)

In many tensor network algorithms we encounter partially contracted tensor networks. In dmrg for example, one needs to know the sum of all the hamiltonian contributions left and right of the site that we want to optimize. If you then optimize the neighboring site to the right, you only need to add one new contribution to the previous sum of hamiltonian contributions.

This kind of information is stored in the environment objects (at the moment called "Cache" in our code, but the name is subject to change). The goal is that the user should preferably never have to deal with the caches, but being aware of the inner workings may allow you to write more efficient code. That is why they are nonetheless included in the manual.

## Finite Environments

When you create a state and a hamiltonian:

```julia
state = FiniteMPS(rand,ComplexF64,20,ℂ^2,ℂ^10);
operator = nonsym_ising_ham();
```

an environment object can be created by calling
```julia
cache = environments(state,operator)
```

The partially contracted mpohamiltonian left of site i can then be queried using:
```julia
leftenv(cache,i,state)
```

This may take some time, but a subsequent call to
```julia
leftenv(cache,i-1,state)
```

Should pretty much be free. Behind the scenes the cache stored all tensors it used to calculate leftenv (state.AL[1 .. i]) and when queried again, it checks if the tensors it previously used are identical (using ===). If so, it can simply return the previously stored results. If not, it will recalculate again. If you update a tensor in-place, the caches cannot know using === that the actual tensors have changed. If you do this, you have to call poison!(state,i).

As an optional argument, many algorithms allow you to pass in an environment object, and they also return an updated one. Therefore, for time evolution code, it is more efficient to give it the updated caches every time step, instead of letting it recalculate.

## Infinite Environments

Infinite Environments are very similar :
```julia
state = InfiniteMPS([ℂ^2],[ℂ^10]);
operator = nonsym_ising_ham();
cache = environments(state,operator)
```

There are also some notable differences. Infinite environments typically require solving linear problems or eigenvalue problems iteratively with finite precision. To find out what precision we used we can type:
```julia
(cache.tol,cache.maxiter)
```

To recalculate with a different precision :
```julia
cache.tol=1e-8;
recalculate!(cache,state)
```

Similar to finite environments, when queried with a different state:
```julia
different_state = InfiniteMPS([ℂ^2],[ℂ^10]);
leftenv(cache,3,different_state)
```

the cache will simply check if the states match up and if not, recalculate! behind the scenes.
