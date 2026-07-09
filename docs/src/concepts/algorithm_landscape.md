# [The algorithm landscape](@id concept_algorithm_landscape)

MPSKit deliberately separates *what* you want to compute from *how* it gets computed.
Entry points such as [`find_groundstate`](@ref), [`timestep`](@ref), [`excitations`](@ref), [`leading_boundary`](@ref), and [`approximate`](@ref) each accept several interchangeable algorithm structs, and the package ships more than a dozen of them.
That flexibility exists because no single algorithm wins everywhere: some only apply to finite or only to infinite systems, some can grow the bond dimension while others cannot, and their relative performance depends on the model at hand.
<!-- REVIEW: "their relative performance depends on the model at hand" — this is the framing claim inherited from man/algorithms.md ("figuring out the optimal algorithm is not always straightforward, since this may strongly depend on the model"); please confirm it is the message we want to lead with. -->

This page is the decision guide.
It starts from a table that maps each task onto the algorithm(s) of choice, and then walks through the reasoning behind each row.
It explains *why* you would pick one algorithm over another; for the *how* — the actual calls, keywords, and worked recipes — follow the links into the how-to pages.

## The decision table

| Task | Finite system | Infinite system |
|:-----|:--------------|:----------------|
| **Ground state** ([`find_groundstate`](@ref)) | [`DMRG`](@ref) (workhorse, fixed bond dimension); [`DMRG2`](@ref) (grows bond dimension, requires `trscheme`); [`GradientGrassmann`](@ref) (final polish) | [`VUMPS`](@ref) (workhorse, needs a unique ground state); [`IDMRG`](@ref) / [`IDMRG2`](@ref) (two-site requires `trscheme` and a unit cell of at least two sites); [`GradientGrassmann`](@ref) (final polish) |
| **Time evolution** ([`timestep`](@ref) / [`time_evolve`](@ref)) | [`TDVP`](@ref) (fixed bond dimension); [`TDVP2`](@ref) (grows bond dimension, requires `trscheme`); or [`make_time_mpo`](@ref) ([`WI`](@ref) / [`WII`](@ref) / [`TaylorCluster`](@ref)) applied with [`approximate`](@ref) | [`TDVP`](@ref) (no two-site variant exists); or [`make_time_mpo`](@ref) applied with [`approximate`](@ref) |
| **Excitations** ([`excitations`](@ref)) | [`QuasiparticleAnsatz`](@ref) (the only one supporting charged `sector`s); [`FiniteExcited`](@ref) (penalty method); [`ChepigaAnsatz`](@ref) / [`ChepigaAnsatz2`](@ref) (cheap, from ground-state environments) | [`QuasiparticleAnsatz`](@ref) (momentum-resolved, the only choice) |
| **Boundary / statistical mechanics** ([`leading_boundary`](@ref)) | apply the transfer MPO row by row with [`approximate`](@ref) | [`VUMPS`](@ref); [`VOMPS`](@ref) (power method); [`IDMRG`](@ref) / [`IDMRG2`](@ref); [`GradientGrassmann`](@ref) (hermitian, positive transfer matrices) |
| **Compression / approximation** ([`approximate`](@ref), [`changebonds`](@ref)) | [`approximate`](@ref) with [`DMRG`](@ref) / [`DMRG2`](@ref); [`SvdCut`](@ref) via [`changebonds`](@ref) for local truncation | [`approximate`](@ref) with [`IDMRG`](@ref) / [`IDMRG2`](@ref) / [`VOMPS`](@ref); [`SvdCut`](@ref) via [`changebonds`](@ref) for local truncation |


A few structural facts hold across the whole table and are worth internalizing early.
Every two-site algorithm (`DMRG2`, `IDMRG2`, `TDVP2`) requires an explicit `trscheme` keyword and can change the bond dimension as it runs; the single-site variants with their default settings cannot.
`IDMRG2` additionally needs a unit cell of at least two sites, and `TDVP2` exists only for finite MPS.
Finally, algorithms compose: the `&` operator chains two algorithms into one, running the first to completion and handing its result to the second, which is how two-site warm-up passes and gradient-descent polishing stages are combined with a workhorse algorithm in a single call.

## Ground states

The classic approach is alternating local optimization: [`DMRG`](@ref) sweeps back and forth through a finite chain, optimizing one site while all others are held fixed, which in practice converges to the ground state.
The catch is the fixed bond dimension: a single-site update can never enlarge the virtual spaces, so the precision of the calculation is locked in by the initial state.
This bites hardest when symmetries are involved, because then not just the total bond dimension but its distribution over charge sectors is frozen, and a poor initial distribution cannot be repaired.
[`DMRG2`](@ref) fixes this by optimizing two neighbouring sites jointly and truncating back down, which lets the bond dimension (and its sector distribution) adapt, at a higher cost per sweep.

For infinite systems, two philosophies compete.
[`IDMRG`](@ref) grows the system from the middle outwards, repeatedly inserting and optimizing new sites until the boundary is no longer felt; [`IDMRG2`](@ref) is its two-site, bond-growing variant.
Because convergence requires the effective system to outgrow the correlation length, IDMRG can be slow to converge for critical systems, where that length diverges.
[`VUMPS`](@ref) instead works with a genuinely uniform state: each local update is followed by a re-gauging step that replaces *every* tensor in the infinite chain with the updated one, so the effect of an update is felt throughout the system immediately.
This often gives VUMPS a higher convergence rate than IDMRG, which is why it is the default infinite-system workhorse.
The price is an injectivity requirement: VUMPS assumes a unique ground state, and it is not the right tool when the state it should converge to is non-injective.
Like DMRG, VUMPS is single-site and cannot alter the bond dimension.

[`GradientGrassmann`](@ref) approaches the problem from a third direction: the MPS tensors form a Riemannian manifold (a Grassmann manifold), and one can run gradient descent directly on it, for finite and infinite states alike.
Its niche is the tail of the optimization: close to convergence its rate is often the best of the lot, while far from convergence the sweeping algorithms tend to make faster progress.
The practical consequence is the chaining pattern: run a cheap workhorse first, then hand over to gradient descent, e.g. `VUMPS(...) & GradientGrassmann(...)`.
This pattern is baked into `find_groundstate` itself: called with only keywords, it picks `DMRG` for a finite state and `VUMPS` for an infinite one, appends a `GradientGrassmann` stage on infinite states when the requested tolerance is tighter than `1e-4`, and prepends a two-site pass (`DMRG2` or `IDMRG2`) whenever you supply a `trscheme`.
Since gradient descent is also a single-site method, growing the bond dimension remains the job of that two-site pre-pass or of [`changebonds`](@ref).

For call syntax, keyword tables, and worked chaining examples, see [Ground-state algorithms](@ref howto_groundstate_algorithms).

## Time evolution

MPSKit solves the time-dependent Schrödinger equation along two distinct routes, and the choice between them is a genuine trade-off rather than a finite/infinite split.

The first route, [`TDVP`](@ref), never builds the evolution operator at all.
It projects the Schrödinger equation onto the tangent space of the current MPS, solves the projected equation for a small time step, and repeats.
Its two-site variant [`TDVP2`](@ref) plays the same role as `DMRG2` does for `DMRG`: it lets the bond dimension grow to absorb the entanglement generated by the evolution, at extra cost, and it exists only for finite systems.

The second route splits the problem in two: first approximate the evolution operator ``\exp(-iH\,dt)`` itself as an MPO using [`make_time_mpo`](@ref) — with [`WI`](@ref), [`WII`](@ref), or [`TaylorCluster`](@ref) as the approximation scheme — and then apply that MPO to the state with [`approximate`](@ref).
The appeal is amortization: for a time-independent Hamiltonian and a fixed step size the MPO is built once and reused for every step, and the accuracy of the operator approximation is controlled independently of the accuracy of its application.

Both routes accept an `imaginary_evolution` keyword for evolution in imaginary time.
For step-by-step recipes along either route, see [Time evolution](@ref howto_time_evolution).

## Excitations

Resolving states deep in the spectrum is generally out of reach, but three families of algorithms target the low-lying part, each with a distinct character.

The [`QuasiparticleAnsatz`](@ref) is the most broadly applicable: it works for finite and infinite systems, and it is the only algorithm that can target excitations carrying a nontrivial symmetry charge, via the `sector` keyword.
It builds an excited state by replacing a single tensor of the ground-state MPS — summed over all positions on a finite chain, or in a momentum-carrying plane-wave superposition on an infinite one — and solves the resulting eigenvalue problem.
Because the variational class consists of local perturbations on top of the ground state, it is the natural choice for quasiparticle-like excitations, and on infinite systems it is the only option, giving direct access to dispersion relations.

[`FiniteExcited`](@ref) takes a brute-force approach available only on finite chains: it reruns a full ground-state optimization on a modified Hamiltonian that carries an energy penalty for overlapping with all previously found states.
Each new excited state therefore costs another complete ground-state search, and the orthogonality to earlier states is only approximate (enforced by the penalty `weight`, not exactly).
Its advantage is that it makes no assumption about the *form* of the excited state: since each state is a fully variational `FiniteMPS`, it can in principle capture excitations that a local perturbation of the ground state would describe poorly.

The [`ChepigaAnsatz`](@ref) (and its two-site refinement [`ChepigaAnsatz2`](@ref)) is the cheapest of the three, also finite-only.
It observes that the gauged ground-state MPS tensors act as isometries projecting the Hamiltonian into a low-energy subspace, so the low-lying spectrum can be read off by diagonalizing the effective Hamiltonian already available from the ground-state environments, with no additional sweeping.
This works best precisely where excitations are hard for the other methods: in critical systems with long-range correlations, where the excitation weight is spread across the whole chain.

For the call signatures, momentum scans, and sector-targeting recipes, see [Excited states](@ref howto_excitations).

## Boundaries and statistical mechanics

MPS algorithms are not limited to Hamiltonian problems.
A two-dimensional classical partition function can be written as an infinite power of a row-to-row transfer MPO, and contracting the network amounts to finding that operator's dominant eigenvector — a boundary MPS.
This is the job of [`leading_boundary`](@ref), which accepts a familiar cast: [`VUMPS`](@ref) and [`IDMRG`](@ref)/[`IDMRG2`](@ref) carry over directly from the ground-state problem, [`GradientGrassmann`](@ref) applies when the transfer MPO is hermitian and positive, and [`VOMPS`](@ref) is a power method specific to this setting, which iteratively approximates the operator-times-state product by a new state of the same bond dimension.

## Compression and changing bond dimension

Two mechanisms round out the landscape by manipulating states rather than solving for new ones.
[`approximate`](@ref) variationally fits a new MPS, typically of different bond dimension, to the result of applying an MPO to a state; the sweeping ground-state algorithms (`DMRG`/`DMRG2` for finite, `IDMRG`/`IDMRG2`/`VOMPS` for infinite) double as its optimization engines.
This is the same machinery that applies time-evolution MPOs, and combined with [`SvdCut`](@ref) it yields a globally optimal truncation of a state.
[`changebonds`](@ref), by contrast, performs direct local surgery on a state: truncating with [`SvdCut`](@ref), or expanding with [`OptimalExpand`](@ref), [`RandExpand`](@ref), or [`VUMPSSvdCut`](@ref) so that the single-site algorithms above have room to work with.
The trade-offs between those expansion schemes, and recipes for when to grow, are covered in [Controlling bond dimension](@ref howto_bond_dimension).

## Where to go next

The how-to pages turn each row of the table into runnable recipes: [Ground-state algorithms](@ref howto_groundstate_algorithms), [Time evolution](@ref howto_time_evolution), and [Excited states](@ref howto_excitations), with [Controlling bond dimension](@ref howto_bond_dimension) supporting all three.
For the complete signatures, keyword lists, and docstrings of every algorithm named here, see the library reference: [Ground-state algorithms](@ref lib_groundstate), [Time evolution](@ref lib_time_evolution), and [Excitations](@ref lib_excitations).

<!--
Notes for the maintainer / plan:
- This page names SketchedExpand nowhere, although it is exported; the bond-dimension
  how-to and lib/bond_dimension.md cover it, and it felt too experimental for a
  decision-guide table. Add it to the compression row if you disagree.
- Dropped from man/algorithms.md rather than salvaged here: Dynamical DMRG /
  propagator, fidelity_susceptibility, periodic/open_boundary_conditions, and
  exact_diagonalization ("Varia" section) — they are not decision-table tasks per the
  plan entry. If the plan wants them surfaced, a short "Beyond the table" paragraph
  could be appended.
- The finite cell of the boundary/statmech row is thin (leading_boundary has no
  finite methods); if a finite-partition-function workflow page ever lands, that cell
  should link to it.
-->
