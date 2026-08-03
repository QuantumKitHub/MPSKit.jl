```@meta
DocTestSetup = quote
    using MPSKit, TensorKit
end
```

# [Numerical considerations](@id concept_numerics)

Every MPSKit calculation is an approximation controlled by a handful of numerical knobs, and understanding what those knobs actually measure is what separates a trustworthy result from a plausible-looking one.
A ground state is only approached to a finite tolerance; a bond dimension only captures so much entanglement; a floating-point number only stores so many digits.
This page explains the three quantities that govern accuracy — the *truncation error* that bounds how well the ansatz can represent a state, the *convergence criterion* that tells an iterative algorithm when to stop, and the *precision* of the underlying element type — and then surveys the failure modes that these considerations give rise to.
It is about understanding *why* a calculation is or is not accurate; for the diagnostic recipe when one goes wrong, follow the links into [Convergence troubleshooting](@ref howto_convergence_troubleshooting).

## Truncation and the bond dimension

The single most important approximation in the whole framework is truncation.
An MPS represents the wavefunction as a chain of tensors joined along virtual bonds, and the dimension of those bonds — the *bond dimension* — is the ansatz's capacity: it is the number of Schmidt coefficients kept when the state is split into two halves at that bond.
Cutting the chain at one bond and performing a singular value decomposition of the resulting bipartition yields exactly the Schmidt decomposition, whose singular values are the Schmidt coefficients.
Keeping only the largest of them is the truncation, and the bond dimension is the number kept.

The quality of that truncation is measured by the *discarded weight*: the sum of the squares of the Schmidt coefficients that were thrown away.
Because the Schmidt coefficients of a normalized state satisfy ``\sum_i \lambda_i^2 = 1``, the discarded weight is the fraction of the state's norm that the truncation sacrifices, and it is the natural error measure of the approximation.
A small discarded weight means the kept bond dimension already captures almost all of the state's entanglement across that cut, so enlarging it further buys little.

We can watch this directly.
After optimizing a ground state we read off its Schmidt spectrum at the central bond with [`entanglement_spectrum`](@ref), and see how quickly the coefficients decay:

```@example numerics
using MPSKit, MPSKitModels, TensorKit
ψ = FiniteMPS(16, ℂ^2, ℂ^24)
H = transverse_field_ising(FiniteChain(16); g = 1.0)
ψ, envs, ϵ = find_groundstate(ψ, H, DMRG(; tol = 1e-10, verbosity = 0))
schmidt = sort(collect(entanglement_spectrum(ψ, 8)); rev = true)
round.(schmidt[1:6]; digits = 4)
```

The tail beyond the first few coefficients is tiny, so truncating the bond back down to keep only the six largest discards only a small weight:

```@example numerics
discarded = sum(abs2, schmidt[7:end])
```

Performing that truncation with [`SvdCut`](@ref) through [`changebonds`](@ref) and comparing the energy before and after shows the corresponding cost in the observable of interest:

```@example numerics
ψcut = changebonds(ψ, SvdCut(; trunc = truncrank(6)))
ΔE = real(expectation_value(ψcut, H) - expectation_value(ψ, H, envs))
```

The bond dimension is thus a genuine accuracy/cost dial: a larger bond dimension lowers the discarded weight and the truncation error, at the price of more expensive tensor contractions.

How much bond dimension a state actually *needs* is set by its entanglement.
Ground states of gapped, local one-dimensional Hamiltonians obey an entanglement *area law* — their bipartite entanglement entropy saturates to a constant as the system grows — which is precisely why a finite bond dimension can represent them efficiently; at a critical point the entropy instead grows without bound and no fixed bond dimension suffices.

### Choosing what to truncate

The rule for *which* coefficients to discard is a truncation scheme.
MPSKit itself does not define these; they come from the tensor backend, so the schemes below require `using TensorKit` (which re-exports them from `MatrixAlgebraKit`) rather than `using MPSKit` alone.
The ones you will meet most often are:

- `truncrank(n)` — keep a fixed number of coefficients (a hard bond-dimension cap).
- `trunctol(; atol)` — discard every coefficient below a threshold.
- `truncerror(; atol)` — keep as many coefficients as needed to hold the discarded weight below a target.
- `truncspace(V)` — truncate to a prescribed vector space, used mostly internally to match bond spaces.
- `notrunc()` — keep everything; this is the default `trunc` of the bond-preserving single-site algorithms.

These schemes compose with `&`, so `trunctol(; atol = 1e-8) & truncrank(16)` applies both bounds at once.

Every bond-growing algorithm — [`DMRG2`](@ref), [`IDMRG2`](@ref), [`TDVP2`](@ref) — and every explicit bond-surgery tool — [`SvdCut`](@ref), [`OptimalExpand`](@ref) — requires a `trunc` keyword, because their whole job is to decide a new bond dimension.
The single-site workhorses ([`DMRG`](@ref), [`VUMPS`](@ref), [`TDVP`](@ref)) default to `notrunc()` and keep the bond dimension fixed.
The recipes for growing and shrinking bonds live in [Controlling bond dimension](@ref howto_bond_dimension).

## Convergence criteria

Every iterative algorithm needs a rule for when it has done enough, and "converged" means something specific and measurable rather than "looks stable."
For the single-site variational algorithms the measure is the **Galerkin error**: the norm of the component of the local energy gradient that points out of the current tangent space of the MPS.
Intuitively, it is how far the exact update at a site wants to push the state in a direction the fixed-bond-dimension ansatz cannot follow; when it is small everywhere, the state is a fixed point of the update to within the ansatz's reach.
This is the quantity [`DMRG`](@ref), [`VUMPS`](@ref), and [`VOMPS`](@ref) drive to zero, and it is returned to you as the third output of the entry point:

```@example numerics
ϵ
```

That returned `ϵ` is the final Galerkin error, and convergence is declared when it drops below the algorithm's `tol`.
The default tolerance is `1e-10`, defined together with the other numerical defaults in the (public but unexported) `MPSKit.Defaults` module, alongside a default `maxiter` of `200` and a default Krylov dimension of `30`:

```@example numerics
MPSKit.Defaults.tol
```

Not every algorithm reports the same measure, and the differences matter when comparing runs:

- The two-site [`DMRG2`](@ref) does not use the Galerkin error during its sweep; it monitors instead the local infidelity between each two-site tensor before and after truncation, which is a different — and generally less directly interpretable — proxy for convergence.
- [`IDMRG`](@ref) and [`IDMRG2`](@ref) judge convergence by the change in the bond matrix between successive iterations, ``\lVert C - C_\text{old}\rVert``, rather than by a gradient norm.
- [`GradientGrassmann`](@ref) converges on the Riemannian gradient norm reported by its underlying optimizer, with `tol` passed through as the gradient tolerance.

A subtlety worth knowing is that these tolerances are, by default, *dynamic*: MPSKit tightens the tolerances of the inner linear and eigenvalue solvers as the outer iteration converges, so that early iterations are not over-solved and late ones are not under-solved.
This adaptive behavior is on by default (`dynamic_tols = true` in `Defaults`) and is why the inner solvers do not simply run at the outer `tol` from the first sweep.

The Galerkin error certifies that the algorithm reached a fixed point of *its own* update, which is necessary but not sufficient for the state to be a good eigenstate.
An independent check is the energy [`variance`](@ref) ``\langle H^2\rangle - \langle H\rangle^2``, which vanishes exactly for a true eigenstate and does not rely on the ansatz's tangent space:

```@example numerics
variance(ψ, H, envs)
```

## Precision and the element type

Underneath the tensors is an ordinary floating-point element type, and by default it is complex double precision, `ComplexF64`.
A randomly initialized state carries that type unless you ask for another:

```@example numerics
scalartype(FiniteMPS(16, ℂ^2, ℂ^24))
```

Double precision is the right default: at `Float64` the relative rounding error is about ``10^{-16}``, comfortably below the `1e-10` convergence tolerance, so floating-point noise is rarely what limits an MPSKit result — truncation and incomplete convergence dominate long before precision does.

The choice between a real and a complex element type is occasionally load-bearing rather than cosmetic.
Real-time evolution and the ``W^{II}`` time-evolution MPO intrinsically require complex arithmetic, so a real-valued state must be promoted before it can be evolved; MPSKit provides `Base.complex` on an MPS for exactly this, and it is a no-op when the state is already complex.
Conversely, a purely real problem — a real Hamiltonian with a real ground state — can in principle be run in `Float64` to save memory and time, but this is an optimization to reach for deliberately, not the default.

## Common pitfalls

Most non-convergence has one of a small number of causes, and recognizing them conceptually is half the battle; the concrete diagnostics are collected in [Convergence troubleshooting](@ref howto_convergence_troubleshooting).

**Too small a bond dimension.**
If the state genuinely needs more entanglement than the bond dimension can hold, no amount of iterating will converge it — the discarded weight is bounded away from zero by the ansatz itself.
This is aggravated by the single-site algorithms, which cannot enlarge the bond dimension: with a symmetry, a single-site sweep freezes not only the total bond dimension but its distribution over charge sectors, so a poor initial distribution cannot be repaired without a two-site pass or an explicit [`changebonds`](@ref) expansion.

**Local minima.**
The variational optimization is non-convex, and an algorithm can settle into a state that is a fixed point of its update but not the global ground state.
The single-site methods are more prone to this than bond-growing ones, which is part of why a two-site warm-up (or a gradient-descent polishing stage, chained with `&`) is often used before or after a single-site run.

**Symmetry-sector trapping.**
When the state carries a conserved quantum number, the optimization runs within a fixed set of symmetry sectors on each bond.
If the true ground state lives in a sector distribution the initial state does not span, the algorithm converges — cleanly, by its own criterion — to the best state in the *wrong* variational space.
This is a sharper, symmetry-specific version of the too-small-bond-dimension trap, and it is why the sector structure of the initial state matters.

**Non-injectivity and a near-degenerate transfer matrix (infinite systems).**
[`VUMPS`](@ref) assumes a unique, injective fixed point.
When the state it should converge to is non-injective — for instance a cat state superposing symmetry-broken sectors, or a genuinely degenerate ground space — the transfer matrix has more than one eigenvalue of magnitude one, and the algorithm has no well-defined single fixed point to find.
MPSKit's [`correlation_length`](@ref) machinery detects this: it is computed from the gap between the leading and next-to-leading transfer-matrix eigenvalues (the correlation length is the inverse of that gap), and [`transfer_spectrum`](@ref) exposes the spectrum directly.
Internally the routine emits a `"Non-injective mps?"` warning when it finds more than one eigenvalue near magnitude one at the same complex angle — a heuristic flag, not a hard error, so it is worth watching for.

**Finite-entanglement effects at criticality.**
At or near a critical point the true correlation length diverges, but a finite bond dimension can only support a finite correlation length, so the simulated correlation length saturates at a value set by the bond dimension rather than by the physics.
Extracting critical data therefore requires studying how results drift as the bond dimension grows, rather than trusting any single bond dimension.

## Where to go next

For the step-by-step diagnosis of a calculation that will not converge, see [Convergence troubleshooting](@ref howto_convergence_troubleshooting).
For the mechanics of changing the bond dimension, see [Controlling bond dimension](@ref howto_bond_dimension), and for the reasoning behind each algorithm's convergence behavior see [The algorithm landscape](@ref concept_algorithm_landscape).
The full signatures of the diagnostic functions named here — [`entanglement_spectrum`](@ref), [`entropy`](@ref), [`variance`](@ref), [`correlation_length`](@ref), and [`transfer_spectrum`](@ref) — are in the library reference.
