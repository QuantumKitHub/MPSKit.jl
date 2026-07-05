# [Excited states](@id howto_excitations)

The examples on this page use MPSKit.jl, MPSKitModels.jl, and TensorKit.jl.
See [Installation](@ref tutorial_installation) for how to add these packages to your environment.

[`excitations`](@ref) is the single entry point for computing energy eigenstates beyond the ground state.
This page shows how to call it for a gap, a full dispersion relation, a charged excitation, and a handful of excited states on a finite chain.
For what each algorithm actually does and why you would choose one over another, see [Excitations](@ref lib_excitations).
All examples share a single namespace:

```@example excitations_howto
using MPSKit, MPSKitModels, TensorKit
```

---

## 1. Get a single excitation gap on an infinite chain

On an `InfiniteMPS`, [`QuasiparticleAnsatz`](@ref) perturbs every site of the unit cell in a plane-wave superposition with a fixed `momentum`, given as a `Real` in radians per unit cell.
Pass the ground state and (optionally) its environments straight through from [`find_groundstate`](@ref):

```@example excitations_howto
g = 2.0
H_inf = transverse_field_ising(; g)
ψ₀_inf = InfiniteMPS(ℂ^2, ℂ^12)
ψ_inf, envs_inf, = find_groundstate(ψ₀_inf, H_inf; verbosity = 0)

Es_inf, ϕs_inf = excitations(H_inf, QuasiparticleAnsatz(), 0.0, ψ_inf, envs_inf; num = 1)
Es_inf[1]
```

The values in `Es_inf` are excitation *gaps* above the ground-state energy density, not total energies: internally the ground-state energy per site is subtracted before diagonalizing.
`ϕs_inf` holds the corresponding quasiparticle states (`num` of them), which behave like normal vectors for `eigsolve`-style post-processing but are not `FiniteMPS`/`InfiniteMPS` objects themselves.
Raise `num` to get more than one state at the same momentum, e.g. `num = 3` for the three lowest excitations at that momentum.

---

## 2. Scan the dispersion relation

Pass a range (or any vector) of momenta instead of a single number to sweep the whole Brillouin zone in one call:

```@example excitations_howto
momenta = range(0, π, 5)
Es_disp, ϕs_disp = excitations(
    H_inf, QuasiparticleAnsatz(), momenta, ψ_inf, envs_inf;
    num = 1, verbosity = 0
)
size(Es_disp)
```

With a vector of `length(momenta)` momenta, `Es_disp` and `ϕs_disp` come back as `(length(momenta), num)` matrices rather than plain vectors — index `Es_disp[:, n]` for the dispersion of the `n`-th branch, or use `vec(Es_disp)` when `num = 1`.
`verbosity = 0` silences the per-momentum `@info` line that this method otherwise prints; raise it to see progress on a longer scan.
Momenta are independent of each other, so this method also accepts `parallel = true` (the default) to distribute them over available threads/workers; pass `parallel = false` to force sequential evaluation.

---

## 3. Target a symmetry sector

By default the optimization looks for the lowest excitation with trivial (vacuum) total charge, `sector = leftunit(ψ)`.
Passing a different `TensorKit` sector targets a quasiparticle with that charge instead — only `QuasiparticleAnsatz` supports this keyword.
Build the ground state with symmetric tensors first, then request the sector on the excitation call:

```@example excitations_howto
g = 10.0
L = 12
H_Z2 = transverse_field_ising(Z2Irrep, FiniteChain(L); g)
ψ₀_Z2 = FiniteMPS(L, Z2Space(0 => 1, 1 => 1), Z2Space(0 => 8, 1 => 8))
ψ_Z2, envs_Z2, = find_groundstate(ψ₀_Z2, H_Z2; verbosity = 0)

Es_triv, = excitations(H_Z2, QuasiparticleAnsatz(), ψ_Z2, envs_Z2; num = 1)
Es_charged, ϕs_charged = excitations(
    H_Z2, QuasiparticleAnsatz(), ψ_Z2, envs_Z2;
    num = 1, sector = Z2Irrep(1)
)
Es_triv[1], Es_charged[1]
```

Here the `Z2Irrep(1)` excitation corresponds to a single flipped spin, the lowest physical excitation of the transverse-field Ising model.
[`ChepigaAnsatz`](@ref)/[`ChepigaAnsatz2`](@ref) do not support charged excitations at all: passing a nontrivial `sector` to either raises an error, and [`FiniteExcited`](@ref) has no `sector` keyword in the first place.

---

## 4. Excited states on a finite chain

Momentum is not a conserved quantity on a finite chain, so the finite method of `excitations` has no momentum argument; drop it entirely and call `QuasiparticleAnsatz` on the ground state directly:

```@example excitations_howto
L = 12
H_fin = transverse_field_ising(FiniteChain(L); g)
ψ₀_fin = FiniteMPS(L, ℂ^2, ℂ^16)
ψ_fin, envs_fin, = find_groundstate(ψ₀_fin, H_fin; verbosity = 0)

Es_qp, ϕs_qp = excitations(H_fin, QuasiparticleAnsatz(), ψ_fin, envs_fin; num = 1)
Es_qp[1]
```

[`FiniteExcited`](@ref) takes a different approach: it repeatedly finds the ground state of `H + weight * Σᵢ |ψᵢ⟩⟨ψᵢ|`, penalizing overlap with the ground state and any excited states already found, and returns full `FiniteMPS` objects instead of quasiparticle states:

```@example excitations_howto
fe_alg = FiniteExcited(; gsalg = DMRG(; verbosity = 0), weight = 10.0)
Es_fe, ψs_fe = excitations(H_fin, fe_alg, ψ_fin; num = 2)
Es_fe
```

Unlike `QuasiparticleAnsatz`, the values in `Es_fe` are total energies of the excited states, directly comparable to `expectation_value(ψ_fin, H_fin)` on the ground state.
<!-- REVIEW: confirm that running a full ground-state search (with its own maxiter/tol) once per requested state makes FiniteExcited noticeably more expensive per excitation than QuasiparticleAnsatz, and that this cost grows with `num` since each new state is optimized against all previous ones. -->
Because each call to `FiniteExcited` reruns a full ground-state optimization under the hood, it scales worse with `num` than the other methods here; reach for it when you need excited states of a genuinely different character than the ground state (so the projector penalty, rather than a local perturbation, is what finds them).

[`ChepigaAnsatz`](@ref) is a cheaper alternative for excitations that are qualitatively similar to the ground state: it diagonalizes the effective Hamiltonian at a single site `pos` (default the middle of the chain) using the ground-state environments, with no extra sweeping:

```@example excitations_howto
Es_ch, ψs_ch = excitations(H_fin, ChepigaAnsatz(), ψ_fin, envs_fin; num = 1, pos = L ÷ 2)
Es_ch[1]
```

[`ChepigaAnsatz2`](@ref) does the same with a two-site block at `pos, pos + 1`, which costs more but is typically more accurate; it truncates the optimized two-site tensor back down with a `trscheme` keyword (`notrunc()` by default):

```@example excitations_howto
Es_ch2, ψs_ch2 = excitations(H_fin, ChepigaAnsatz2(; trscheme = truncrank(16)), ψ_fin, envs_fin; num = 1)
Es_ch2[1]
```

Like `FiniteExcited`, the energies returned by both Chepiga variants are total energies, not gaps.
<!-- REVIEW: confirm the recommendation that ChepigaAnsatz/ChepigaAnsatz2 are specifically well-suited to critical (long correlation length) systems, where growing the bond dimension enough to resolve excitations directly becomes expensive, per the "Chepiga ansatz" discussion in man/algorithms.md. -->

---

## 5. Check excitation quality

[`variance`](@ref) accepts a quasiparticle state directly and reports the variance of the energy, with smaller values indicating a better-converged excitation:

```@example excitations_howto
variance(ϕs_qp[1], H_fin)
```

It also works on the infinite quasiparticle states from §1–§3:

```@example excitations_howto
variance(ϕs_inf[1], H_inf)
```

!!! warning "Variance of infinite quasiparticle states"
    `variance` on an infinite quasiparticle state carries an unresolved implementation note in `src/algorithms/toolbox.jl` and may be unreliable; verify its output before relying on it as a convergence diagnostic.
    It also throws an `ArgumentError` for domain-wall (topological) excitations, where it is not implemented at all.

To measure other observables on a finite quasiparticle state, convert it to a plain `FiniteMPS` first:

```@example excitations_howto
excited_state = convert(FiniteMPS, ϕs_qp[1])
real(expectation_value(excited_state, H_fin))
```

---

## 6. Domain-wall excitations

`excitations` also accepts two *different* ground states, `excitations(H, QuasiparticleAnsatz(), momentum, ψ_left, envs_left, ψ_right, envs_right; ...)`, which builds a quasiparticle that interpolates between them — a domain-wall (topological) excitation rather than a local perturbation on top of a single ground state.
This is real, exported functionality, but it has no dedicated test or example in the repository at the time of writing, and constructing two genuinely distinct, well-converged ground states to feed it (for instance the two symmetry-broken ground states of an ordered phase) is itself nontrivial to set up reliably in a short recipe.
<!-- REVIEW: this section is intentionally prose-only; a maintainer-verified worked example (e.g. two symmetry-broken ferromagnetic TFIM ground states at g < 1) would be a good addition once the two-ground-state construction is validated. -->

---

## Where to go next

For growing the bond dimension of the ground states these recipes start from, see [Controlling bond dimension](@ref howto_bond_dimension).
For background on the quasiparticle ansatz and the other algorithms used here, see [Excitations](@ref lib_excitations).
For general expectation values and correlators, including on the converted excited states from §5, see [Computing observables](@ref howto_observables).
Spectral functions built from these excitations (`propagator`, `DynamicalDMRG`, and related solvers) are a separate topic covered in the "Linear problems and spectral functions" section of the [Public API](@ref public_api) reference.
