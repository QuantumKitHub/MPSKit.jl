# [Quasiparticle excitations](@id tutorial_excitations)

The previous tutorials ended with a ground state: the lowest-energy state of the transverse-field Ising model, first on a finite chain and then directly in [the thermodynamic limit](@ref tutorial_thermodynamic_limit).
The natural next question is what lies *above* it: how much energy does it cost to excite the system?
For a translation-invariant chain the answer is organized by momentum — for each momentum ``k`` there is a lowest excitation energy ``\Delta E(k)``, and the resulting curve is the **dispersion relation** of the model.
Its minimum over all momenta is the **energy gap**, one of the most basic characterizations of a quantum phase.
<!-- REVIEW: physics framing — "lowest excitation energy at each momentum defines the dispersion relation, whose minimum is the gap" is standard for translation-invariant systems, but please confirm this phrasing is acceptable as stated. -->

In this tutorial we compute the dispersion relation of the infinite transverse-field Ising chain with MPSKit's quasiparticle ansatz, and finish with a plot of ``\Delta E(k)`` across the Brillouin zone — compared against the exact solution.

## Loading the packages

As in the previous tutorials, every code block on this page shares one Julia session, so we load the packages once.

```@example excitations
using MPSKit, MPSKitModels, TensorKit
using Plots
```

## 1. Find the ground state

Excitations are computed *on top of* a ground state, so the first step is the calculation you already know from [The thermodynamic limit](@ref tutorial_thermodynamic_limit): build the infinite Hamiltonian, make a random `InfiniteMPS`, and converge it with `VUMPS`.

This time we set the field to `g = 2.0`, deep in the paramagnetic phase, where the model is **gapped**: the lowest excitation costs a finite amount of energy, which is exactly what we want to measure.
<!-- REVIEW: physics claim — TFIM at g = 2 is in the gapped paramagnetic (disordered) phase; the transition is at g = 1. -->

```@example excitations
g = 2.0
H = transverse_field_ising(; g)
ψ₀ = InfiniteMPS(ℂ^2, ℂ^12)
ψ, envs, ϵ = find_groundstate(ψ₀, H, VUMPS(; verbosity = 0))
```

We keep all three return values this time: the optimized state `ψ` and the environments `envs` both feed directly into the excitation calculation below, so nothing has to be recomputed.

## 2. One excitation at one momentum

The **quasiparticle ansatz** builds an excited state directly on top of the uniform ground state.
The idea is simple to picture: take the converged ground state and perturb it locally, replacing the tensor at one site with a new one that we get to optimize.
Because the chain is infinite and translation invariant, we do not place this perturbation at any particular site; instead we superpose it across *all* sites with a plane-wave phase, which gives the excitation a definite momentum ``k``.
Optimizing the perturbation then yields the lowest excited state at that momentum.
<!-- REVIEW: quasiparticle-ansatz framing — "local perturbation of the uniform ground state, momentum-superposed over all sites, then variationally optimized" — please confirm this hand-held description does not oversimplify in a misleading way. -->

The call is [`excitations`](@ref) with the [`QuasiparticleAnsatz`](@ref) algorithm, a momentum (a real number, in radians per site), and the ground state with its environments.
Let us ask for the excitation at the edge of the Brillouin zone, ``k = \pi``:

```@example excitations
E, ϕ = excitations(H, QuasiparticleAnsatz(), π, ψ, envs)
E
```

Two things to note about the return values:

- `E` is a *vector* of excitation energies, of length `num` — the keyword controlling how many excitations to compute at this momentum, which defaults to `num = 1`, so here it has a single entry.
- The entries of `E` are energies **above the ground state** — gaps at this momentum, not total energies. The ground-state energy is subtracted internally, so you can read them off directly.

The second return value `ϕ` holds the corresponding quasiparticle states, which can be used for further post-processing; we will not need them in this tutorial.

## 3. The full dispersion

To trace out the whole dispersion relation we simply pass a *range* of momenta instead of a single number.
By symmetry it is enough to scan from ``0`` to ``\pi``, and we use 16 points to keep the runtime modest.
<!-- REVIEW: physics claim — restricting the scan to [0, π] uses the k → −k symmetry of the TFIM dispersion. -->

```@example excitations
momenta = range(0, π, 16)
Es, ϕs = excitations(H, QuasiparticleAnsatz(), momenta, ψ, envs; verbosity = 0)
size(Es)
```

With a range of momenta the energies come back as a matrix of size `(length(momenta), num)` — here `(16, 1)`, one row per momentum and one column because we kept the default `num = 1`.
We pass `verbosity = 0` to silence the progress line this method otherwise prints for every momentum.
The momenta are independent of one another, so MPSKit works on them in parallel by default.

## 4. Plot the dispersion

Now for the payoff.
This particular model is exactly solvable, so we can plot our numerical dispersion right on top of the known answer:

```math
\Delta E(k) = 2\sqrt{1 + g^2 - 2 g \cos k}.
```

For this Hermitian problem the computed energies come back as real numbers; the `real.(...)` below is a harmless safeguard for the general case, where the eigenvalue solver may return a complex number type with numerically vanishing imaginary parts.

```@example excitations
k_exact = range(0, π, 200)
ΔE_exact = @. 2 * sqrt(1 + g^2 - 2g * cos(k_exact))
plot(k_exact, ΔE_exact; label = "exact", xlabel = "momentum k", ylabel = "ΔE(k)", title = "TFIM dispersion (g = $g)")
scatter!(momenta, real.(Es); label = "quasiparticle ansatz (D = 12)")
```

The 16 computed points fall right on the exact curve.
The dispersion rises monotonically from ``k = 0`` to ``k = \pi``, so its minimum — the gap — sits at zero momentum, where the exact value is ``\Delta E(0) = 2(g - 1)``.
Our first matrix entry is precisely that point, so we can close with a numerical check:

```@example excitations
real(Es[1, 1]), 2 * (g - 1)
```

A ground state at bond dimension 12 plus a variational quasiparticle on top reproduces the exact gap of the model — that is the quasiparticle ansatz working as intended.
## Where to go next

You have computed a full dispersion relation on top of an infinite ground state and read off the energy gap.

The [`excitations`](@ref) entry point can do considerably more than what we used here: it can target excitations carrying a nontrivial symmetry charge, build topological (domain-wall) excitations that interpolate between two different ground states, and compute excited states of *finite* chains, where momentum is no longer a good quantum number and different algorithms take over.
All of these are recipes in [Excited states](@ref howto_excitations).
For what each excitation algorithm actually does and when to choose it, see the library reference [Excitations](@ref lib_excitations).

<!-- NOTE(other pages, do not edit here): tutorials/thermodynamic_limit.md line 134 has
"TODO(link): excitations tutorial" which can now point to (@ref tutorial_excitations);
the orchestrator wires that separately. This page also needs a nav entry in docs/make.jl. -->
