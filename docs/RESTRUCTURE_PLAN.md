# MPSKit.jl Documentation Restructure Plan

## Problem diagnosis
The current Manual is organized around the library's **type taxonomy** (States,
Operators, Algorithms, Lattices), which mirrors the architecture rather than the
reader's goals. Content answers "what is a `FiniteMPS`" instead of "how do I get a
ground state and a plot." The docs read as "terse" not because of word count but
because they're reference content lacking a learning path and a conceptual layer.

## Organizing principle: Diátaxis
Sort every piece of content by the reader's mode, and don't blend modes:
- **Tutorials** — learning-oriented, hand-held, sequential
- **How-to guides** — task recipes, assume competence
- **Concepts** — understanding-oriented prose (the "why")
- **Reference** — lookup, mostly auto-generated

Current state: heavy on reference, thin on how-to, missing tutorials and concepts.

## Target page tree (`docs/src/`)

```
docs/src/
├── index.md                          # Landing / router (VitePress hero frontmatter)
│
├── tutorials/                        # LEARNING — sequential, hand-held
│   ├── installation.md
│   ├── first_groundstate.md          # FLAGSHIP
│   ├── thermodynamic_limit.md
│   ├── time_evolution.md
│   ├── excitations.md
│   └── using_symmetries.md
│
├── howto/                            # TASK RECIPES
│   ├── states.md
│   ├── hamiltonians.md
│   ├── groundstate_algorithms.md
│   ├── bond_dimension.md
│   ├── observables.md
│   ├── entanglement.md
│   ├── time_evolution.md
│   ├── excitations.md
│   ├── statmech.md
│   ├── quasi_1d_geometries.md
│   ├── convergence_troubleshooting.md
│   ├── parallelism_gpu.md
│   └── saving_loading.md
│
├── concepts/                         # UNDERSTANDING
│   ├── vector_spaces.md
│   ├── matrix_product_states.md
│   ├── finite_vs_infinite.md
│   ├── operators_and_hamiltonians.md
│   ├── symmetries.md
│   ├── algorithm_landscape.md
│   ├── environments.md
│   ├── parallelism_model.md
│   └── numerics.md
│
├── examples/                         # SHOWCASE — full case studies
│   ├── index.md
│   ├── haldane_gap.md
│   ├── hubbard_model.md
│   ├── classical_ising_2d.md
│   └── heisenberg_dispersion.md
│
├── lib/                              # REFERENCE — mostly @autodocs
│   ├── public.md                     # curated stable API surface
│   ├── states.md
│   ├── operators.md
│   ├── groundstate.md
│   ├── time_evolution.md
│   ├── excitations.md
│   ├── bond_dimension.md
│   ├── observables.md
│   ├── environments.md
│   ├── internals.md                  # non-public, marked unstable
│   └── index.md                      # symbol index
│
├── references.md
├── changelog.md
├── migration.md                      # breaking-change / upgrade guide
├── contributing.md
└── citing.md
```

## Per-page contents

### Landing
- **index.md** — what MPSKit is + who for, key features, install, one 30-sec runnable
  example, a "where next" router to the four sections, and an ecosystem map. Use the
  VitePress home-page hero layout (YAML frontmatter: hero title, tagline, action
  buttons linking to the four Diátaxis sections).

### Tutorials (each runnable end-to-end, no forward references)
- **installation.md** — full env: MPSKit + TensorKit + MPSKitModels + optional plotting; verify snippet.
- **first_groundstate.md** — FLAGSHIP: FiniteMPS -> TFIM Hamiltonian -> find_groundstate/DMRG -> observable -> magnetization plot.
- **thermodynamic_limit.md** — same physics with InfiniteMPS + VUMPS; the "near-identical code" payoff.
- **time_evolution.md** — simple quench with TDVP/timestep, one observable over time.
- **excitations.md** — quasiparticle ansatz, plot a dispersion.
- **using_symmetries.md** — redo flagship with U(1); sector syntax + bond-dim benefit.

### How-to (task-shaped)
- **states.md** — construct FiniteMPS/InfiniteMPS/WindowMPS/MultilineMPS; from tensors, product, random, with sectors.
- **hamiltonians.md** — Finite/InfiniteMPOHamiltonian via @mpoham, manual, longer-range, boundary conditions.
- **groundstate_algorithms.md** — DMRG/DMRG2/VUMPS/IDMRG/GradientGrassmann config + tolerances.
- **bond_dimension.md** — changebonds, OptimalExpand/RandExpand/SvdCut, dynamic expansion.
- **observables.md** — local/multi-site expectation values, correlators.
- **entanglement.md** — entanglement entropy + spectrum from gauge tensors.
- **time_evolution.md** — real/imaginary time; TDVP/TDVP2; make_time_mpo (WI/WII).
- **excitations.md** — QuasiparticleAnsatz/FiniteExcited; momentum-resolved; dispersions.
- **statmech.md** — 2D classical/transfer matrix; leading_boundary; MultilineMPS.
- **quasi_1d_geometries.md** — cylinders/ladders via MPSKitModels; ordering choices.
- **convergence_troubleshooting.md** — diagnostic recipe for non-convergence.
- **parallelism_gpu.md** — threading settings; GPU state/caveats (mark experimental).
- **saving_loading.md** — serialize/reload states and environments.

### Concepts (prose, the "why")
- **vector_spaces.md** — TensorKit mental model; the steepest cliff.
- **matrix_product_states.md** — gauging/canonical forms; AL/AR/AC/C meaning.
- **finite_vs_infinite.md** — unit cells; why infinite MPS normalize to 1.
- **operators_and_hamiltonians.md** — MPO structure, Jordan-block form, sparse vs dense.
- **symmetries.md** — abelian/non-abelian/fermionic/anyonic; sectors; "when does SU(2) pay off." Differentiator — make it the most polished.
- **algorithm_landscape.md** — DMRG vs VUMPS vs IDMRG vs TDVP vs gradient: decision guide.
- **environments.md** — what caches are, why stored/reused, API impact.
- **parallelism_model.md** — internal structure, what scales.
- **numerics.md** — truncation error, convergence criteria, precision, pitfalls.

### Examples (full literate case studies)
- gallery index + Haldane gap (SU(2)), Hubbard (fermionic), 2D classical Ising (statmech), Heisenberg dispersion (excitations).

### Reference (mostly @autodocs, split by category)
- **public.md** is the curated stable surface (the API contract). Category pages group docstrings. **internals.md** = unstable. **index.md** = alphabetical symbol index.

### Meta
- references (DocumenterCitations), changelog, migration, contributing, citing (Zenodo DOI).

## Process notes
- **Migration is mostly *splitting*, not rewriting**: existing Manual pages map onto
  concepts/ (prose) + lib/ (API). Split each along that seam.
- **Incremental, not big-bang**: keep old pages live, build the new layer alongside,
  then narrow the Manual's role to pure reference once the learning path exists.
- **Self-testing docs**: wire every code block in tutorials/ and examples/ through
  Documenter @example/jldoctest and gate in CI. Doubles as integration tests.
- **Rendering backend**: DocumenterVitepress (VitePress). This does not change the
  content plan; it changes build/preview/deploy plumbing only.
- **Priority order**: (1) re-sort existing content; (2) flagship tutorial;
  (3) algorithm_landscape.md + vector_spaces.md; (4) symmetry progression;
  (5) convert examples to tested doctests + CI gating.
