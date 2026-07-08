# [How-to guides](@id howto_index)

These pages are task recipes: short, runnable answers to "how do I do X?".
They assume you already know the basics — if you are new to MPSKit, start with [Your first ground state](@ref tutorial_first_groundstate) instead.
Each recipe below stands on its own, so feel free to jump straight to the one you need.

## States and operators

**[Constructing states](@ref howto_states)** — building `FiniteMPS`, `InfiniteMPS`, `WindowMPS`, and `MultilineMPS` objects.
- A finite MPS — random states, initializers and element types, per-site spaces, product states, and wrapping your own site tensors.
- An infinite MPS — single- and multi-site unit cells, from spaces or from tensors.
- A window MPS — a mutable finite region embedded in infinite environments.
- A multiline MPS — stacking `InfiniteMPS` rows for boundary-MPS methods.
- States with symmetries — building MPS with `Rep[G]` graded spaces.

**[Building Hamiltonians](@ref howto_hamiltonians)** — assembling MPO Hamiltonians from local operators.
- Finite Hamiltonian from local terms — `FiniteMPOHamiltonian` from `inds => operator` pairs.
- Infinite (translation-invariant) Hamiltonian — `InfiniteMPOHamiltonian` on a unit cell.
- Converting between boundary conditions — open vs. periodic finite chains from an infinite Hamiltonian.
- A window Hamiltonian — carving a finite interval out of an infinite Hamiltonian with `WindowMPOHamiltonian`.

## Finding ground states

**[Ground-state algorithms](@ref howto_groundstate_algorithms)** — configuring `find_groundstate`.
- Get a ground state with defaults — letting `find_groundstate` pick an algorithm for you.
- Configure finite-system DMRG — explicit `DMRG`/`DMRG2`, and chaining them with `&`.
- Configure infinite-system algorithms — `VUMPS`, `IDMRG`, and `IDMRG2`.
- Refine convergence with GradientGrassmann — Riemannian gradient descent after a cheaper warm-up.
- Control output and tolerances — `verbosity` levels and reusing `envs` between calls.

**[Controlling bond dimension](@ref howto_bond_dimension)** — inspecting, growing, and shrinking bond dimension.
- Inspecting the current bond dimension — `left_virtualspace`/`right_virtualspace` and `dim`.
- Growing bond dimension — `RandExpand` (no Hamiltonian needed) and `OptimalExpand`.
- Reducing bond dimension — `SvdCut` and the in-place `changebonds!`.
- Truncation schemes — `truncrank`, `trunctol`, `notrunc`, `truncspace`, and combining them with `&`.
- Growing during finite MPS optimization — `DMRG2` and the `trscheme` keyword of `find_groundstate`.
- Growing during infinite MPS optimization — `IDMRG2` and `VUMPSSvdCut`.
- Chaining algorithms — composing bond-change and ground-state algorithms with `&`.

## Dynamics

**[Time evolution](@ref howto_time_evolution)** — real- and imaginary-time evolution of an MPS.
- Evolve a state through one time step — `timestep` with `TDVP`.
- Evolve over a time span — `time_evolve` across a vector of time points.
- Grow the bond dimension while evolving — `TDVP2` with a mandatory `trscheme`.
- Evolve an infinite state — single-site `TDVP` on an `InfiniteMPS`.
- Imaginary-time evolution — `imaginary_evolution = true` to cool towards the ground state.
- Build a time-evolution MPO — `make_time_mpo` (`WII`, `TaylorCluster`, `WI`) plus `approximate`.

## Measurements

**[Computing observables](@ref howto_observables)** — extracting physical quantities from an MPS.
- Local (one-site) expectation value — `expectation_value(ψ, i => O)`.
- Multi-site (contiguous) expectation value — tensor-product operators on an index tuple.
- Energy (full-MPO expectation value) — `expectation_value(ψ, H)` for a Hamiltonian MPO.
- Two-point correlators — `correlator`, including a full correlation profile over a range.
- Energy variance as a convergence check — `variance` as a diagnostic after a ground-state search.

**[Entanglement entropy and spectrum](@ref howto_entanglement)** — reading off entanglement from the gauge tensors.
- Entanglement entropy at a single cut — `entropy(ψ, site)`.
- Entropy profile across every cut — collecting `entropy` over all sites.
- The entanglement spectrum — `entanglement_spectrum` as a sector-resolved vector.
- Sector-resolved spectrum — indexing the spectrum by symmetry sector with `keys`/`pairs`.
- Entanglement of an infinite MPS — `entropy`/`entanglement_spectrum` per unit-cell site.
- Plotting the spectrum — the `entanglementplot` recipe (requires Plots.jl).

## Excitations

**[Excited states](@ref howto_excitations)** — computing energy eigenstates beyond the ground state.
- Get a single excitation gap on an infinite chain — `QuasiparticleAnsatz` at a fixed momentum.
- Scan the dispersion relation — passing a range of momenta in one call.
- Target a symmetry sector — a charged quasiparticle via the `sector` keyword.
- Excited states on a finite chain — `QuasiparticleAnsatz`, `FiniteExcited`, and the Chepiga ansätze.
- Check excitation quality — `variance` on a quasiparticle state.
- Domain-wall excitations — quasiparticles interpolating between two distinct ground states.

## Performance and hardware

**[Parallelism and GPU support](@ref howto_parallelism_gpu)** — tuning how MPSKit uses the hardware.
- Setting BLAS threads — `BLAS.set_num_threads` and the OpenBLAS/MKL difference.
- Setting the MPSKit scheduler — `MPSKit.Defaults.set_scheduler!` with `:serial`/`:greedy`/`:dynamic`.
- Diagnosing the thread layout — ThreadPinning.jl `threadinfo`.
- Reducing memory usage — disabling multithreading to avoid `OutOfMemory`.
- GPU support — the experimental Adapt-based path for moving states onto a GPU.

## Missing a recipe?

If the task you're after isn't listed here, please open an issue at [QuantumKitHub/MPSKit.jl](https://github.com/QuantumKitHub/MPSKit.jl/issues) describing what you're trying to do.
Concrete task descriptions make the best new recipes.
