# MPSKit.jl Accessibility & Adoption Plan

Companion to `RESTRUCTURE_PLAN.md`.
That file defines *where content lives* — the Diátaxis page tree — and remains the source of truth for structure.
This file defines *what to improve, why, and in what order* to flatten the learning curve and make the package more attractive to adopt.
Every content action below is tagged with its Diátaxis mode (**Tutorial** / **How-to** / **Concept** / **Reference**, plus **Meta** for the out-of-quadrant pages like changelog and citing) and, where applicable, its target page in the planned tree.
Nothing here invents structure outside that plan, with one deliberate, flagged addition: a `benchmarks.md` meta page (§4).

Basis: a full review of the current docs (`man/`, `howto/`, `lib/`, `examples/`, landing page), the README, the entry-point docstrings, the error-message surface in `src/`, project metadata, and a survey of how ITensor(MPS.jl), TeNPy, and quimb onboard their users.

---

## 1. Diagnosis: why the learning curve is steep

### 1.1 There is no happy path (the dominant problem)

The reader is asked to climb from reference prose directly to research-grade case studies, with nothing in between.

- The landing page (`docs/src/index.md`), immediately after the hero, has the reader contracting gauge tensors with `@tensor` and verifying isometry conditions — before they have ever run a ground-state search.
- The Manual entry point (`man/intro.md`, "Prerequisites") teaches TensorKit by deliberately triggering two errors ("incompatible partition", "incompatible arrows") before the reader can *do* anything, and buries its one motivating sentence (planarity, fermion signs) at the bottom.
- The examples ladder starts at research grade: the first quantum example extracts a CFT spectrum with a translation MPO, the first 2D example uses Fibonacci anyons.
  There is no beginner rung.
- The five existing how-to pages cover *setup* (states, Hamiltonians, bond dimension, observables, entanglement) but none of the headline workflows a newcomer arrives for: ground state end-to-end, time evolution, excitations.
- `man/algorithms.md` — where the first runnable `find_groundstate` should be — is a wall of `@docs` blocks; the first runnable ground-state call appears ~140 lines in, as scaffolding for excitations.

**Fix**: the Tutorials quadrant of the plan, which is currently the empty one.
This is the single highest-leverage work (§5, W1).

### 1.2 The TensorKit cliff is unmanaged

- TensorKit vocabulary (`ℂ^2`, `⊗`, domain/codomain, sectors, `Rep[U₁]`, `repartition`, `oneunit`, `id`, `right_virtualspace`) is assumed on essentially every page, with no point-of-use links.
  The `@extref` mechanism is set up and used correctly in exactly two spots of `operators.md` — the fix pattern exists but is not applied.
- The load-bearing index-ordering convention ($V_l \otimes P \leftarrow P \otimes V_r$) is conveyed *only* through three PNG images with no text/math fallback.
- Operators like `σˣ()`, `σᶻ()` materialize from hidden `@setup` blocks, so the rendered page never shows where they come from.

**Fix**: `concepts/vector_spaces.md` written as "TensorKit for MPS users" with a spin-1/2 running example (§5, W3), plus systematic `@extref` links and visible setup code everywhere.

### 1.3 Breadth reads as intimidation, not power

MPSKit's real differentiators — many algorithms, finite *and* infinite MPS, generic symmetries up to anyons — are presented as an undifferentiated list.
`man/algorithms.md` itself admits "figuring out the optimal algorithm is not always straightforward", then provides no decision guidance.
The genuinely useful comparison prose that exists (DMRG vs DMRG2 bond-dimension behaviour, VUMPS vs IDMRG convergence, "gradient descent is best in the tail", the four `changebonds` trade-offs) is buried between docstring dumps.

**Fix**: `concepts/algorithm_landscape.md` with a decision table, plus how-to recipes for each headline workflow (§5, W2).

### 1.4 Trust-eroding defects

Small things that individually are minor but together signal "unfinished" to an evaluating user:

- `man/lattices.md` is an empty "Coming soon!" stub — and `man/operators.md` *actively directs readers to it* right after a hard 30-line manual 2D-indexing example.
- Three how-to pages ship stale `<!-- CLOSING NOTES FOR MAINTAINERS -->` blocks in their source, including the admission "none of the numeric outputs above have been hand-verified against a real build".
- `man/environments.md` mixes `envs` and `cache` for the same object, uses two different model constructors, and its `@time` examples — whose entire point is showing the caching speedup — display no output.
- Assorted: an off-by-one comment (`fill(ℂ^2, 3) # a finite chain of 4 sites` in `operators.md`), garbled LaTeX in the finite-excitations formula in `algorithms.md`, an empty `[MKL.jl]()` link in `parallelism.md`.
- The docs code depends on packages the install instructions never mention (§1.5).

**Fix**: workstream W0 — a one-time sweep, largely mechanical, that should land before anything else because it is cheap and the payoff is credibility.

### 1.5 The out-of-docs surface undersells the package

- **Install instructions are incomplete.** The how-tos require `TensorKitTensors` (spin operators) and rely on `truncrank` from MatrixAlgebraKit; the README quickstart needs `MPSKitModels`, `Plots`, `ProgressMeter`.
  No page lists the actual working environment.
  A newcomer copy-pasting the first how-to hits `Package TensorKitTensors not found`.
- **Docstrings have no runnable examples.** Of the ~10 first-touch entry points, only `expectation_value` has a `jldoctest`.
  `find_groundstate` — the most important function — does not list keyword defaults, and its docstring says "an optimization algorithm will be attempted based on the supplied keywords" without documenting the actual heuristic (VUMPS→GradientGrassmann for infinite, DMRG/DMRG2 for finite, the `trscheme` branches).
- **Error messages are bimodal.** MPO-construction paths interpolate site/level context nicely, but core plumbing throws `error("Invalid state")`, `error("method ambiguity")`, `ArgumentError("dimension mismatch")` with no values, and message-less `@assert`s that surface as raw `AssertionError` to users.
- **Project metadata gaps.** The changelog lives only inside the docs tree (invisible on GitHub); no `CONTRIBUTING.md`; no issue/PR templates; `CITATION.cff` is stale relative to `Project.toml`; no BibTeX snippet in the README.
- **Version-resolution friction.** Many tightly-pinned pre-1.0 ecosystem deps (`TensorKit = "0.17"`, `BlockTensorKit = "0.3.14"`, …) mean a user with slightly older ecosystem versions hits resolver dead-ends.

---

## 2. Assets to build on (do not rewrite these)

- **The README quickstart** — two complete runnable examples with rendered magnetization plots — is currently the best onboarding asset in the whole project.
  The docs landing page should mirror it, not the other way round.
- **The examples content** is genuinely attractive: real research workflows, Literate.jl pages with captured output, plots, SVG diagrams, Binder/nbviewer badges.
  The problem is discoverability (a bare auto-generated index, no difficulty ordering, no beginner rung), not quality.
- **The five `howto/` pages are the right shape**: numbered task recipes, short prose, runnable `@example` blocks, consistent TFIM running example.
  `howto/bond_dimension.md` (truncation-scheme table, `&`-chaining, "warm up with DMRG2, refine with DMRG") is the template to copy.
- **`lib/public.md` is a real API contract** — curated, grouped, with an explicit stability note.
  Promote it; fill its hollow categories (§5, W4).
- **`DOCSTRING_STYLE.md` and its three templates** already solve the docstring-consistency problem on paper; W4 is mostly *applying* them.
- **Hidden gems in `man/` prose** that must survive the migration as concept material: the gauge "automagic" framing and the overlap-via-center-gauge example (`man/states.md`), the algorithm-comparison paragraphs and `changebonds` trade-off list (`man/algorithms.md`), the DMRG environment-reuse motivation (`man/environments.md`), the OpenBLAS-vs-MKL threading explanation (`man/parallelism.md`).

---

## 3. Competitor practices worth adopting — mapped into Diátaxis

Survey targets: ITensor / ITensorMPS.jl, TeNPy, quimb.
All three share: a ground-state hero example visible within one screen (with *real output*), a named tutorial track separate from a recipe collection, heavy use of tensor-network diagrams as core onboarding, a citable pedagogical paper, and a visible Q&A venue.

| Practice | Seen at | Diátaxis home in our tree |
|---|---|---|
| Hero example with real convergence output on the first screen | ITensorMPS.jl README (Heisenberg DMRG, energies shown) | Landing `index.md` + README (Meta) |
| Browsable one-page recipe index ("code formulas"), with "suggest a recipe" invitation | itensor.org formulas page | `howto/` section index (How-to) |
| "Protocol for reliable (i)DMRG" best-practices page | TeNPy | `howto/convergence_troubleshooting.md` (How-to) + `concepts/numerics.md` (Concept) |
| Progressive worked-example gallery, easy → hard | quimb (16 notebooks), TeNPy notebooks | `examples/` gallery curation |
| Diagrams as onboarding, not decoration | itensor.org animations, quimb `.draw()` | Concept pages (static SVGs; asset precedent already exists in `examples/`) |
| Per-topic FAQ mined from recurring user questions | ITensor (DMRG FAQ, QN FAQ) | **Not a standalone FAQ page** — see note below |
| Community venue linked prominently | ITensor Discourse, TeNPy forum | README + landing (Meta) |
| Citable pedagogical paper doubling as library paper | TeNPy SciPost Lecture Notes | Long-term; `citing.md` neighborhood (Meta) |
| "Papers using X" showcase | itensor.org | New meta page near `citing.md` (Meta) |

**Diátaxis note on FAQs**: a standalone FAQ is a mode-blending dumping ground.
Instead, mine GitHub issues for recurring questions and route each answer to its proper home: task-shaped answers become how-to recipes (most will land in `convergence_troubleshooting.md`), understanding-shaped answers become sections of the relevant concept page.
If discoverability demands it, a short *routing list* of question → link can live on the how-to index, but the content itself lives in-quadrant.

---

## 4. Benchmarks: performance as a headline feature

MPSKit is believed to be significantly faster than ITensor across the board.
<!-- REVIEW: performance claim to be validated by the benchmark suite itself; do not publish any comparative number that the public scripts do not reproduce. -->
If the numbers bear that out, this is the single most compelling adoption argument for the target audience and is currently advertised nowhere.
Speed claims only convert people when they are reproducible and fair, so the suite and the guardrails matter as much as the numbers.

### 4.1 Where it lives (Diátaxis)

- A top-level **`benchmarks.md` meta page** in the docs, alongside `references.md` / `changelog.md` / `citing.md` — factual, reference-toned, versioned.
  This is the one deliberate addition to the `RESTRUCTURE_PLAN.md` page tree.
- One **headline chart** in the README and one feature bullet on the landing hero ("Fast: see the benchmarks").
  Marketing lives in Meta surfaces; tutorials and how-tos never carry comparative claims.
- Scripts in a public **`MPSKitBenchmarks` repo** (or `benchmark/` workspace project) so every published number is one `julia --project` away from being checked.

### 4.2 Proposed suites

Ordered by marketing value per unit of implementation effort.

1. **Finite DMRG time-to-solution** — spin-1 Heisenberg chain and TFIM, N = 100, no symmetries: wall time to reach fixed accuracy (e.g. `ΔE/E ≤ 1e-8` vs a reference energy) and time per sweep, at χ ∈ {64, 128, 256, 512, 1024}.
   Competitors: ITensorMPS.jl, TeNPy.
2. **Abelian symmetry** — same models with U(1) conservation.
   Competitors: ITensorMPS.jl (QN), TeNPy.
3. **Non-abelian SU(2)** — spin-1 Heisenberg with full SU(2).
   ITensor and TeNPy support only abelian QNs, so this row is a *capability* advertisement; benchmark against MPSKit's own U(1) run to show the effective-χ payoff.
4. **Infinite MPS: VUMPS / IDMRG** — infinite Heisenberg and TFIM time-to-tolerance.
   VUMPS as a first-class citizen is itself a differentiator; compare against TeNPy iDMRG where a comparison exists.
5. **TDVP throughput** — global quench on N = 100 chain, cost per unit time evolved at fixed χ.
6. **Quasi-2D DMRG** — Heisenberg or Hubbard cylinder (width 4–6), time per sweep; this is the workload reviewers of "serious" packages look for.
7. **Thread scaling** — speedup vs `nthreads` for suite 1, since the multithreaded environment/derivative code is an MPSKit strength worth showing.

### 4.3 Methodology guardrails (non-negotiable for credibility)

- Compare **time-to-accuracy**, not time-per-iteration — sweep/iteration semantics differ across libraries.
- Identical Hamiltonians, χ schedules, truncation thresholds, BLAS backend, and thread counts; pinned versions of every package; hardware documented.
- Use each competitor's *official* example/recommended settings as the starting point, so nobody can call it a strawman; ideally have the comparison scripts sanity-checked by someone fluent in that library.
- Publish everything: scripts, environment manifests, raw timings, and the plotting code.
- Report losses as honestly as wins; a table where MPSKit wins 9 of 11 rows is *more* persuasive than a table where it wins 11 of 11.
- Re-run and re-date the page per MPSKit minor release (automate as far as practical).

### 4.4 Result placeholders

To be filled by the benchmark runs; the page ships only when at least suites 1–2 are populated.

**Suite 1 — Finite DMRG, spin-1 Heisenberg, N = 100, time to ΔE/E ≤ 1e-8 (seconds, lower is better):**

| χ | MPSKit.jl | ITensorMPS.jl | TeNPy | speedup vs best competitor |
|---|---|---|---|---|
| 64 | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| 128 | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| 256 | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| 512 | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| 1024 | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

**Suite 2 — same, with U(1) symmetry:** (same table shape)

**Suite 3 — SU(2) capability payoff (MPSKit only):**

| symmetry | effective χ | wall time | memory |
|---|---|---|---|
| none | _tbd_ | _tbd_ | _tbd_ |
| U(1) | _tbd_ | _tbd_ | _tbd_ |
| SU(2) | _tbd_ | _tbd_ | _tbd_ |

Headline README chart: log–log wall-time vs χ for suite 1, one line per library.

---

## 5. Workstreams, prioritized

### W0 — Trust and correctness sweep (cheap, do first, independent of restructure)

- [ ] List the *actual* working environment everywhere installation is discussed: `MPSKit`, `TensorKit`, `MPSKitModels`, `TensorKitTensors`, `Plots` (+ note that `truncrank` etc. come from MatrixAlgebraKit).
      Targets: `index.md` now, `tutorials/installation.md` when it exists. (Tutorial/Meta)
- [ ] Strip the stale `<!-- CLOSING NOTES FOR MAINTAINERS -->` blocks from `howto/observables.md`, `howto/entanglement.md`, `howto/hamiltonians.md` after running their outputs through doctest verification. (How-to)
- [ ] Kill the `man/lattices.md` dead end: remove from nav or write the minimal real content; fix the forward reference in `man/operators.md`. (Reference)
- [ ] `man/environments.md` (currently commented out of the build): unify `envs`/`cache` naming, one model constructor, convert to `@example` blocks with real shown output — or keep it out of nav until its concept-page successor exists. (Concept-to-be)
- [ ] Mechanical fixes: off-by-one comment in `man/operators.md`; broken LaTeX in `man/algorithms.md` finite-excitations formula; empty `[MKL.jl]()` link in `man/parallelism.md`; make hidden `@setup` imports visible in rendered pages. 
- [ ] Metadata: root `CHANGELOG.md` (copy or symlink from docs), `CONTRIBUTING.md`, `.github/ISSUE_TEMPLATE/` + PR template, bump `CITATION.cff` and automate it in the release process, BibTeX snippet in README. (Meta)

### W1 — The happy path (Tutorials + landing + examples ladder) — highest leverage

- [ ] Rewrite `index.md`: keep the VitePress hero, replace the gauge-tensor walkthrough with a ≤20-line TFIM ground-state example showing real printed output (mirror the README quickstart), route the action buttons to the tutorial track instead of "Manual", add the "where next" router to the four sections. (Meta/landing)
- [ ] `tutorials/installation.md` — full environment + verify snippet. (Tutorial)
- [ ] `tutorials/first_groundstate.md` — the flagship; the single most important page in the plan. (Tutorial)
- [ ] `tutorials/thermodynamic_limit.md` — the "near-identical code for infinite systems" payoff, MPSKit's wow moment. (Tutorial)
- [ ] Examples gallery: add one beginner rung (e.g. TFIM magnetization across the transition — bridges directly from the flagship tutorial), and curate `examples/index.md` with one-line descriptions, difficulty ordering, and thumbnail plots; suppress the 40-line `[ Info: DMRG …]` log dumps in rendered pages. (Examples showcase)

### W2 — Decision guidance (converts breadth from intimidating to attractive)

- [ ] `concepts/algorithm_landscape.md` with an explicit decision table (finite/infinite × ground state/time evolution/excitations/boundary), salvaging the comparison prose buried in `man/algorithms.md`. (Concept)
- [ ] `howto/groundstate_algorithms.md`, `howto/time_evolution.md`, `howto/excitations.md` — the headline workflows currently missing task recipes. (How-to)
- [ ] Make the How-to section index a browsable recipe list à la ITensor's code formulas, with a "missing a recipe? open an issue" invitation. (How-to)

### W3 — The TensorKit bridge

- [ ] `concepts/vector_spaces.md` replaces `man/intro.md`'s TensorKit section: motivated by a spin-1/2 chain, positive-path examples first (errors shown later, as diagnostics), the planarity/fermion "why" up front, keep the ℤ₂ sector example. (Concept)
- [ ] Conventions (index ordering, MPS/MPO tensor definitions) become a reference page with a text + math fallback next to each image. (Reference)
- [ ] Apply `@extref` TensorKit links at point of first use on every page; the pattern already exists in `man/operators.md`. (All modes)
- [ ] Every tutorial and how-to page opens with a visible "packages used" line. (Tutorial/How-to)

### W4 — Reference depth and docstrings

- [ ] Add `jldoctest` examples to the first-touch entry points, copying the `expectation_value` pattern per `DOCSTRING_STYLE.md` Template B — priority order: `find_groundstate`, `FiniteMPS`, `InfiniteMPS`, `FiniteMPOHamiltonian`/`InfiniteMPOHamiltonian`, `DMRG`, `VUMPS`, `timestep`/`time_evolve`, `TDVP`, `changebonds`. (Reference)
- [ ] `find_groundstate`: document keyword defaults and the automatic algorithm-selection heuristic explicitly. (Reference)
- [ ] Fill the hollow `lib/public.md` categories — Environments, `approximate`, `leading_boundary`/`VOMPS`, spectral functions (`propagator`, `DynamicalDMRG`, …) currently live only in the `lib.md` autodocs dump. (Reference)
- [ ] Split the remaining `man/` pages along the concept/reference seam and retire `man/` (per the plan's migration note): `states.md` → `concepts/matrix_product_states.md` + lib; `operators.md` → `concepts/operators_and_hamiltonians.md` + lib; `algorithms.md` prose → `algorithm_landscape.md` + how-tos, its `@docs` → lib; `parallelism.md` → `concepts/parallelism_model.md` + `howto/parallelism_gpu.md`; `environments.md` → `concepts/environments.md`. (Concept/Reference)
- [ ] Error-message pass in `src/`: replace bare `error()` / `error("Invalid state")` / generic `"dimension mismatch"` with messages that interpolate the offending values and suggest a fix; add messages to `@assert`s on user-reachable paths; keep developer-voice asserts internal. (Code, not docs — but part of the perceived learning curve)

### W5 — Reliability content (the TeNPy move)

- [ ] `howto/convergence_troubleshooting.md`: symptom → diagnosis → fix recipes (not converging, sector starvation with U(1), bond dimension saturating, TDVP norm drift, …). (How-to)
- [ ] `concepts/numerics.md`: truncation error, convergence criteria, extrapolation in χ, precision pitfalls. (Concept)
- [ ] Mine closed GitHub issues for recurring questions and route each into the two pages above or the relevant concept page (see FAQ note in §3). (How-to/Concept)
- [ ] These pages are physics-heavy: maintainer-led per the REVIEW workflow in `CLAUDE.md`.

### W6 — Adoption surface and community

- [ ] Benchmarks per §4: build the suite, populate `benchmarks.md`, headline chart in README. (Meta)
- [ ] README: add a short "How MPSKit compares" positioning paragraph — generic symmetries up to non-abelian/fermionic/anyonic, infinite MPS/VUMPS as first-class citizens, Julia-native composability, speed (link benchmarks). (Meta)
- [ ] Pick and prominently link one community Q&A venue (GitHub Discussions or a #quantumkithub channel) from README and landing. (Meta)
- [ ] "Publications using MPSKit" page near `citing.md`. (Meta)
- [ ] Static tensor-network SVG diagrams on the concept pages (MPS/gauge structure, environments, sweep); the examples tree already contains style precedents (`translation_mpo.svg`, `spt-tensors.svg`). (Concept)
- [ ] Loosen the tightest pre-1.0 `[compat]` pins where feasible to reduce resolver dead-ends. (Code)
- [ ] Long-term: a pedagogical lecture-notes-style paper à la TeNPy's SciPost notes, doubling as the citable library paper. (Meta)

---

## 6. Diátaxis guardrails while executing

- One mode per page, no exceptions: tutorials never survey options (they link to the how-to/concept instead); how-tos assume competence and *link* to concepts rather than teaching them; concept pages contain no `@docs` dumps; reference pages contain no narrative.
- The `man/` migration is *splitting, not rewriting* — nearly every `man/` page is good prose fused to API dumps; pull them apart along that seam (see W4).
- Troubleshooting and FAQ material is expressed as how-to recipes and concept sections, never as a standalone mixed-mode page.
- Benchmarks and comparisons are Meta: factual on `benchmarks.md`, promotional only in README/landing hero, absent from tutorials and how-tos.
- Internals and experimental features (`SketchedExpand`, GPU, everything in `lib/internals.md`) stay quarantined behind `!!! warning`, per the hard rules.

## 7. Suggested sequencing

| Order | Work | Rationale |
|---|---|---|
| 1 | W0 | Cheap, restores credibility, unblocks nothing |
| 2 | W1 | The missing quadrant; biggest learning-curve payoff |
| 3 | W2 + W3 | Matches plan priority (3)–(4): decision guide + TensorKit cliff |
| 4 | W4 | Reference depth; large but parallelizable page-by-page |
| 5 | W5 | Physics-heavy, maintainer-led sessions |
| 6 | W6 | Adoption layer; benchmarks can start earlier in parallel since they are independent of docs restructuring |

---

## Appendix A — file-level defect list (from the review)

- `docs/src/index.md` — gauge-tensor walkthrough before any ground-state search; action buttons route to Manual; install section omits `TensorKitTensors`/`Plots`; "Additional Resources" admits the docs are a work in progress.
- `docs/src/man/intro.md` — error-first TensorKit teaching; motivation buried at the bottom; TensorKit link only at line ~62 after all confusing material; conventions conveyed only via images.
- `docs/src/man/states.md` — capability list with trailing ellipsis and zero examples; `repartition` unexplained; `Union{Missing, A}` implementation details interrupt intro-level flow; good gauge prose worth salvaging.
- `docs/src/man/operators.md` — hidden `@setup` imports; "Jordan block" used ~90 lines before definition; off-by-one comment (`fill(ℂ^2, 3)` / "4 sites"); forward reference to empty lattices page; `!!! warning` work-in-progress section shipped; excellent Ising example and Symbolics verification worth salvaging.
- `docs/src/man/algorithms.md` — `@docs` wall; no top-level runnable ground-state example; no decision table; garbled LaTeX in finite-excitations formula; `verbosity=false` vs `verbosity=0` inconsistency; valuable comparison prose and `changebonds` section worth salvaging.
- `docs/src/man/environments.md` — non-running ` ```julia ` blocks; `envs` vs `cache` naming drift; two different model constructors; `@time` examples show no output; currently commented out of `make.jl`.
- `docs/src/man/lattices.md` — empty stub in nav.
- `docs/src/man/parallelism.md` — empty `[MKL.jl]()` link; advanced material unlabeled as such; threading explanation worth keeping as-is.
- `docs/src/howto/{observables,entanglement,hamiltonians}.md` — stale maintainer-comment blocks incl. "outputs not verified" admissions.
- `docs/src/howto/*` (all) — depend on `TensorKitTensors`/MatrixAlgebraKit never mentioned in install docs; no prerequisite links up to concept material.
- `docs/src/lib/public.md` — categories promised without category pages (Environments, spectral functions, `leading_boundary`, `VOMPS`, `approximate`).
- `docs/src/lib/observables.md` — two `!!! warning`s exposing unreliable methods (`expectation_value(::MultilineMPS, …)` "TODO: fix environments"; `variance(::InfiniteQP, …)` "may be unreliable"): fix or quarantine.
- `docs/src/examples/index.md` — bare auto-generated title list; no descriptions, ordering, or thumbnails; no beginner example; verbose solver logs dumped into rendered pages.
- `src/` error messages — bare `error()`, `error("Invalid state")`, `error("method ambiguity")`, `ArgumentError("dimension mismatch")`, message-less `@assert`s on user-reachable paths.
- Repo metadata — no root `CHANGELOG.md`/`CONTRIBUTING.md`/issue templates; `CITATION.cff` version stale vs `Project.toml`.
