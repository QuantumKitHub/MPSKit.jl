# [The thermodynamic limit](@id tutorial_thermodynamic_limit)

In [Your first ground state](@ref tutorial_first_groundstate) we put the transverse-field Ising model on a finite chain of `L = 16` sites.
That is a perfectly good calculation, but it carries two prices: the open ends of the chain are physically different from its middle (boundary effects), and every quantity we measured still depends on the length `L` (finite-size effects).
To read off the true physics of the model we would have to repeat the calculation at several lengths and extrapolate `L → ∞`.

MPSKit lets you skip that extrapolation and work *directly* at `L = ∞`.
The trick is translation invariance: instead of storing one tensor per site, we store a single tensor and imagine it repeated forever along the chain — an [`InfiniteMPS`](@ref).
There are no ends, so there are no boundary effects, and there is no `L` to extrapolate.
Best of all, as you are about to see, the code barely changes: the same model, the same workflow, two edits.

!!! note "Infinite states are always normalized"
    An `InfiniteMPS` is normalized to 1 by construction, and you cannot choose otherwise.
    Any other normalization would make expectation values either blow up or vanish as the (infinite) chain length is taken to infinity, so per-site quantities are the only ones that make sense here.

## Loading the packages

As before, every code block on this page shares one Julia session, so we load the packages once.

```@example thermodynamic-limit
using MPSKit, MPSKitModels, TensorKit
using TensorKitTensors.SpinOperators: σˣ, σᶻ
using Plots
```

## 1. Build the Hamiltonian and initial state

Here are the only two lines that differ from the finite tutorial.

For the Hamiltonian, we drop the lattice argument.
Where the finite version wrote `transverse_field_ising(FiniteChain(L); g = 0.5)`, we simply omit `FiniteChain(L)`: with no lattice, `transverse_field_ising` builds the Hamiltonian for the infinite chain.

```@example thermodynamic-limit
H = transverse_field_ising(; g = 0.5)
```

For the state, we swap `FiniteMPS` for `InfiniteMPS`.
There is no length to pass, so the constructor takes just the physical and virtual spaces — the physical space `ℂ^2` of a spin-1/2 site and the bond space `ℂ^D` whose dimension `D` is again the accuracy knob.

```@example thermodynamic-limit
D = 4
ψ₀ = InfiniteMPS(ℂ^2, ℂ^D)
```

That is the whole difference.
The bond dimension means exactly what it did on the finite chain (see [Controlling bond dimension](@ref howto_bond_dimension)), and `ℂ^2`/`ℂ^D` are the same physical/virtual spaces.

!!! note "`InfiniteMPS` also accepts bare integers"
    Unlike `FiniteMPS`, the infinite constructor happily takes plain integers: `InfiniteMPS(2, D)` is equivalent to `InfiniteMPS(ℂ^2, ℂ^D)`.
    We stick with the explicit spaces to match the rest of the documentation.

## 2. Find the ground state

We optimize with [`VUMPS`](@ref), the infinite-chain workhorse, passing it explicitly so it is visible.

```@example thermodynamic-limit
ψ, envs, ϵ = find_groundstate(ψ₀, H, VUMPS())
```

The lines printed above are VUMPS's per-iteration convergence log, shown at the default `verbosity`.
VUMPS (the variational uniform matrix product state algorithm) optimizes the single repeated tensor directly in the thermodynamic limit, iterating until it reaches a fixed point.

The return value has the same shape as on the finite chain: the optimized state `ψ`, the reusable `envs`, and a convergence-error measure `ϵ`.

!!! note "The algorithm is optional here too"
    Just as `find_groundstate(ψ₀, H)` selected DMRG for a finite input, calling it with no algorithm on an *infinite* input selects VUMPS automatically.
    `VUMPS` accepts the familiar keywords `tol` (default `1e-10`), `maxiter` (default `200`), and `verbosity` (default `3`); we use `verbosity = 0` later to silence the log inside a loop.
    Note there is no `find_groundstate!` for infinite states — VUMPS returns a fresh state and leaves `ψ₀` untouched.

## 3. Measure observables

For the default single-site unit cell used here, `expectation_value(ψ, H)` returns the energy of that one-site unit cell, which is exactly the **energy per site**:

```@example thermodynamic-limit
E = expectation_value(ψ, H)
```

The magnetization is the local order parameter ``\langle\sigma^z\rangle``.
Because the state is translation-invariant, every site is identical, so we measure it at site 1 of the unit cell:

```@example thermodynamic-limit
expectation_value(ψ, 1 => σᶻ())
```

So far these are the same quantities we computed on the finite chain.
The infinite setting also unlocks an observable with no finite-chain analogue: the [`correlation_length`](@ref), extracted from the transfer-matrix spectrum of the uniform state.

```@example thermodynamic-limit
correlation_length(ψ)
```

The correlation length tells us how far apart two spins can still "feel" each other; it is measured in units of the lattice spacing.
It grows as we approach the critical point `g = 1`, where correlations become long-ranged.
We can see this by optimizing a second state right at criticality and comparing:

```@example thermodynamic-limit
H_crit = transverse_field_ising(; g = 1.0)
ψ_crit, = find_groundstate(ψ₀, H_crit, VUMPS(; verbosity = 0))
correlation_length(ψ_crit)
```

At a genuine critical point the correlation length diverges, but a finite bond dimension `D` can only capture correlations out to a finite range, so what we measure is large but capped rather than infinite.

## 4. Magnetization across the transition

As on the finite chain, we finish by sweeping the field `g` and recording the magnetization.
The structure mirrors the finite sweep exactly — only `InfiniteMPS` and `VUMPS` have changed.

```@example thermodynamic-limit
g_values = 0.1:0.1:2
M = map(g_values) do g
    Hg = transverse_field_ising(; g = g)
    ψg, = find_groundstate(ψ₀, Hg, VUMPS(; verbosity = 0))
    return abs(expectation_value(ψg, 1 => σᶻ()))
end
scatter(g_values, M; xlabel = "g", ylabel = "M", label = "D = $D", title = "TFIM magnetization (L = ∞)")
```

Compare this with the finite-chain sweep of the previous tutorial, where the magnetization dropped to zero well before `g = 1`, at a point set by the algorithm rather than by the physics.
The infinite curve instead tracks the transition itself: the magnetization stays on its ordered branch all the way up to the critical point and collapses to zero right at `g = 1`.
What little smearing remains around the critical point is a finite-bond-dimension effect, and it shrinks as `D` grows.

We still take the **absolute value** of the magnetization, but for a subtly different reason than on the finite chain.
On the finite chain the nonzero magnetization was an artifact of the algorithm: the exact ground state there is symmetric, and DMRG landed on a symmetry-broken state only because it carries less entanglement.
In the thermodynamic limit the symmetry breaking is genuine — the two oppositely magnetized states become true ground states — and an infinite MPS at finite bond dimension settles into one of them on the ordered side, landing on a definite nonzero magnetization of either sign; `abs` again puts both branches onto a single order-parameter curve.

## Where to go next

You have now run the same TFIM calculation twice — once at finite size, once directly at `L = ∞` — and seen how little the code had to change.

From here you can go beyond ground states.
A natural next step is to [compute the excitations above this infinite ground state](@ref tutorial_excitations) (the model's quasiparticle spectrum), or to [exploit the symmetries of the model](@ref tutorial_using_symmetries) to make the calculation cheaper and more accurate.

To go deeper on the individual steps used here, see [Constructing states](@ref howto_states), [Controlling bond dimension](@ref howto_bond_dimension), and [Entanglement entropy and spectrum](@ref howto_entanglement); the algorithm reference is [Ground-state algorithms](@ref lib_groundstate).
