# [Environments](@id concept_environments)

Almost every MPS algorithm spends most of its time contracting the same tensor network over and over.
In DMRG, optimizing the tensor on one site requires the sum of all Hamiltonian contributions sitting to its left and to its right; in time evolution the same partial contractions reappear at every step.
Recomputing them from scratch each time would be wasteful, because moving attention from one site to a neighbour changes only a little of the network.
The *environment* objects are what let MPSKit avoid that waste.

This page explains what environments are and why they exist, so that the optional `environments` argument that appears throughout the API stops looking like a mystery.
It is about understanding, not tuning: for the mechanics of a particular algorithm follow the links into the how-to pages.

## What an environment is

An environment is a partially contracted piece of a tensor network — the part that does not change when you shift your focus by one site.
Consider the network whose value an algorithm ultimately wants: a state `below` (the bra), an operator, and a state `above` (the ket), all contracted together.
Fixing attention on a single site splits that network into three parts: the tensor at the site itself, everything to its left, and everything to its right.
The left and right parts are exactly the *left environment* and *right environment* of that site.

The key observation is that these two blocks are shared between neighbouring sites.
The left environment at site `i+1` is the left environment at site `i` with a single extra column contracted onto it.
So once you have paid to build the environment at one site, advancing to the next is cheap: you add one new contribution instead of recontracting the whole chain.
Caching the environments and reusing them across a sweep is what turns an algorithm that would be quadratic in the system size into a linear one.

In MPSKit these cached blocks live in an environment object, constructed with the exported [`environments`](@ref) function.
The canonical form sandwiches an operator between two states,

```julia
using MPSKit, MPSKitModels, TensorKit

state = FiniteMPS(20, ℂ^2, ℂ^10)
H = transverse_field_ising(FiniteChain(20); g = 0.5)
envs = environments(state, H, state)
```

while the two-argument form `environments(below, above)` builds the operator-free *overlap* environments between two states.
The individual blocks are then queried with the exported [`leftenv`](@ref) and [`rightenv`](@ref) functions,

```julia
GL = leftenv(envs, 10, state)   # everything to the left of site 10
GR = rightenv(envs, 10, state)  # everything to the right of site 10
```

each of which returns a tensor gauge-compatible with the state tensor at that site, ready to be contracted onto it.

## Why you rarely build them yourself

Most of the time you never touch an environment object at all.
The high-level entry points — [`find_groundstate`](@ref), [`timestep`](@ref), [`excitations`](@ref), and the rest — build whatever environments they need internally.
What they also do, uniformly, is accept an *optional* environments argument and return an updated environment object alongside their main result.

That return value is the reason to care about environments even when you never construct one.
Handing the environments from one call into the next lets the algorithm reuse the cached blocks instead of rebuilding them from nothing.
For iterated procedures such as time evolution — where each step starts from a state only slightly different from the last — feeding the updated environments back in every step avoids repeating work that the previous step already did.
The [Time evolution](@ref howto_time_evolution) how-to shows this threading pattern in a concrete recipe.

## Finite environments and the `===` cache

For a finite state the environment object manages its own validity automatically, and understanding how is worthwhile because it comes with one sharp edge.

When it computes a left environment, the cache records *which* state tensors it contracted to get there — the gauged tensors of `state` up to that site.
On a later query it compares the tensors it would need now against the ones it used before, testing them with Julia's identity operator `===`.
If they are the same objects, the cached block is still valid and is returned immediately.
If some differ, the cache recomputes only the affected part of the network and updates its record.
This is what makes repeated queries during a sweep cheap: the first `leftenv` at a far site pays for the full contraction, and neighbouring queries reuse almost all of it.

The sharp edge is that `===` tests object identity, not numerical equality.
If you mutate a state tensor *in place* — changing its data while keeping the same object — the cache still sees the same object under `===` and concludes, wrongly, that its stored environment is still valid.
It will then hand back a block computed from the old data.
Building a *new* tensor and assigning it into the state is fine, because that is a different object and the `===` check catches it; only in-place mutation defeats the mechanism.
Because algorithms that use the public API replace tensors rather than mutating them, this is rarely a problem in normal use, but it is the thing to suspect if a hand-written routine that mutates tensors starts returning stale results.

!!! warning "In-place mutation is invisible to the cache"
    The finite-environment cache detects changes by object identity (`===`), so mutating a
    state tensor in place leaves the cache convinced its stored environment is still current.
    The internal, non-exported helper `MPSKit.poison!(envs, i)` marks the dependencies at
    site `i` as stale so the next query recomputes them; needing it is a sign that a tensor
    was mutated in place rather than replaced.

## Infinite environments

Infinite environments serve the same role but are computed differently, and the difference matters for how you use them.

A finite chain has genuine boundaries, so its environments can be built by contracting inward from the ends in a finite number of steps.
An infinite chain has no ends.
Its environments are instead the fixed points of the transfer operator — the object you get by contracting one repeating unit cell — and finding a fixed point means solving a linear or eigenvalue problem.
Those problems are solved *iteratively*, to a finite tolerance, rather than by an exact finite contraction.
Building an infinite environment is therefore a small numerical solve, and its result is only as accurate as the tolerance of that solve.

The precision of that solve is controlled by the `tol`, `maxiter`, and `krylovdim` keyword arguments to [`environments`](@ref), which configure the underlying iterative solver — an Arnoldi eigensolver for a transfer-matrix fixed point, or GMRES for the linear problem that arises with an `InfiniteMPOHamiltonian`.
It is fixed when the environments are built, rather than read from or written to the environment object afterwards.

The second difference is that infinite environments are **not** recomputed automatically.
The finite cache re-validates itself against the current state on every query; the infinite one does not.
If the state changes, the stored fixed points no longer correspond to it, and there is no automatic re-solve.
Bringing an infinite environment up to date for a changed state is an explicit step, handled internally by the non-exported `MPSKit.recalculate!`.In practice the high-level algorithms perform this recomputation for you as part of their own iteration, which is again why threading the returned environments through successive calls is the efficient pattern.

## Where to go next

For the full signatures and docstrings of the environment functions, see [`environments`](@ref), [`leftenv`](@ref), and [`rightenv`](@ref) in the library reference.
For the algorithm-facing side — how ground-state and time-evolution routines consume and return environments — see [Ground-state algorithms](@ref howto_groundstate_algorithms) and [Time evolution](@ref howto_time_evolution).
