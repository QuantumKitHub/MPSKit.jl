# Documentation inspiration from quimb, TeNPy, and ITensor

Actionable ideas for the MPSKit.jl docs, distilled from a structured review (2026-07) of the documentation of three peer libraries:
[quimb](https://quimb.readthedocs.io/), [TeNPy](https://tenpy.readthedocs.io/), and [ITensor / ITensorMPS.jl](https://itensor.org).
Each item names its source and its relation to `RESTRUCTURE_PLAN.md` (the source of truth for structure — nothing here overrides it; items either reinforce a planned page, sharpen its spec, or propose an addition for maintainer sign-off).

Legend: **[quick win]** doable within the current content waves · **[plan addition]** new page/section, needs a plan edit · **[needs package work]** blocked on code, not prose.

---

## 1. What the review validates about the current plan

No action needed — these confirm choices already made, and are listed so we don't relitigate them.

- **The Diátaxis split is the right call.**
  None of the three does it cleanly, and in each case the blending is a named weakness: TeNPy's "Introductions" is a 12-page grab-bag mixing explanation, how-to, and reference detail; quimb's guides mix hand-holding with option dumps; only ITensor separates tutorials / code examples / reference in the sidebar, and it reads best.
- **Executing every example at build time is a real edge — keep the CI gate.**
  quimb commits notebook outputs (`nb_execution_mode = "off"`), so outputs silently rot; TeNPy renders its `.py` scripts and toycodes as dead source listings with no output at all.
  MPSKit's Literate + `@example` pipeline already avoids the exact failure mode both of them have.
- **A runnable example on the landing page (planned in `index.md`) is proven, not speculative.**
  ITensorMPS.jl puts a complete ~20-line Heisenberg DMRG with its real final energy on page one; quimb and TeNPy both lack front-page code and it is their most-felt onboarding gap.
- **`symmetries.md` as the most-polished differentiator is well aimed.**
  TeNPy's charge-conservation page is one of its most-visited docs despite being a 4000-line monolith — there is demand; serving it with several well-scoped pages is the opportunity.

---

## 2. Quick wins (content style, current waves)

- [ ] **Give how-to recipes imperative, search-shaped titles.** *(ITensor "code formulas")*
  Section headings like "Compute the entanglement entropy of an MPS", "Write and read an MPS to disk", "Target an excited state" — titles that match what users type into a search box.
  The planned `howto/` pages are topical (good); the actionable delta is at the *section* level within them, plus a flat index page listing every recipe title as a link so the how-to layer doubles as a self-service FAQ.
- [ ] **Make every recipe self-contained with visible output.** *(ITensor + quimb)*
  Each recipe includes the short setup that produces the state it measures (or shares one via `@setup`), ends with real printed output, and cross-links down to the `lib/` docstrings it uses (`@ref expectation_value` etc.).
- [ ] **Show real convergence logs, not sanitized snippets.** *(quimb)*
  quimb's 1D-algorithms page prints per-sweep energies and progress output so readers learn what healthy convergence *looks like*.
  In the flagship tutorial and the ground-state how-to, keep (a trimmed slice of) the actual iteration log in the rendered output instead of suppressing verbosity.
- [ ] **Upgrade the examples index from a bare list to an annotated gallery.** *(quimb's weakness)*
  One line per example: the physics question, the MPSKit features it exercises (e.g. "SU(2) symmetry, quasiparticle excitations"), and a rough difficulty/runtime marker.
  No VitePress components needed — plain Markdown descriptions already fix quimb's "numbered list with no descriptions" problem.
- [ ] **Group optional dependencies by capability on the install page.** *(quimb)*
  "For plotting: …; for saving states: JLD2; for GPU (experimental): CUDA" — grouped by what the user wants to do, not alphabetically.
  Add the performance notes users actually need at install time: BLAS threading interaction (`OMP_NUM_THREADS` vs Julia threads), and a short HPC/cluster note (ITensor documents cluster installs and precompile latency explicitly because that's where their users run — ours too).
- [ ] **Put a minimal runnable ground-state example and a BibTeX block in the GitHub README.** *(ITensor strength, explicit TeNPy gap)*
  TeNPy's README has install commands but zero code; ITensor's front page shows the flagship calculation in seconds.
  A GitHub visitor should see `find_groundstate` working before clicking anything.
- [ ] **Add redirects for every page the restructure moves.** *(quimb's weakness)*
  quimb's guide URLs churned and old links 404 while search engines still index them.
  As `man/` pages dissolve into `concepts/` + `lib/`, ship redirects (or stub pages with a link) from the old URLs — cheap now, impossible to retrofit into other people's bookmarks and forum answers later.
- [ ] **Coin one design-thesis sentence and repeat it verbatim everywhere.** *(ITensor)*
  ITensor's "interface independent of memory layout / inspired by tensor diagrams" line appears identically on the front page, in the type docstring, and in the paper — it *is* their brand.
  MPSKit's equivalent (roughly: symmetries as first-class citizens via TensorKit, with the same code driving finite and infinite systems) should be crafted once and reused on `index.md`, the README, and `citing.md`.
  <!-- REVIEW: maintainer should own the exact wording of the thesis sentence. -->

---

## 3. Sharpen pages already in the plan

- [ ] **Model `convergence_troubleshooting.md` on TeNPy's "Protocol for using (i)DMRG" — the single best page in any of the three doc sets.**
  Its structure: (1) verify the model is correct *before* trusting any run (contract the MPO to a dense operator for small N and cross-check against exact diagonalization); (2) cheap preliminary runs; (3) production runs; (4) explicit convergence confirmation.
  It names concrete warning signs (non-monotonic energy or entropy, premature termination, sensitivity to parameters) and pairs each with remedies keyed to *actual config options*.
  The MPSKit version should key every remedy to real keywords/algorithms (`tol`, `maxiter`, schedules of `changebonds`, switching DMRG ↔ DMRG2, unit-cell size for incommensurate order) and be honest about limits (critical states vs the area law).
  <!-- REVIEW: the warning-signs-to-remedies mapping is physics judgement — maintainer must own it. -->
- [ ] **Spec `migration.md` + `changelog.md` concretely.** *(all three)*
  Changelog entries grouped per version with **"Breaking changes" as the first section**, hyperlinks from each entry into the `lib/` page of the touched symbol (quimb does this and it turns the changelog into a "what's new to learn" feed), and issue/PR links.
  Per-breaking-release upgrade guides à la ITensor (0.2→0.3→0.4), including the "run the previous version first to surface deprecation warnings" trick from TeNPy.
- [ ] **Make `index.md`'s ecosystem map an architecture diagram with drill-down.** *(TeNPy Overview)*
  TeNPy's best onboarding page draws the layer stack (simulations → algorithms → networks → models → linalg) then walks each layer with a runnable snippet.
  The MPSKit version is the TensorKit → MPSKit → MPSKitModels stack; each layer gets one sentence plus one line of code, ending with explicit "where next" routing.
- [ ] **Cite theory out, don't re-derive it, in `concepts/` pages.** *(TeNPy + ITensor)*
  TeNPy delegates derivations to the Hauschild–Pollmann lecture notes; ITensor delegates to tensornetwork.org; both keep package docs lean and give readers an authoritative "why".
  MPSKit has a natural companion in the tangent-space lecture notes (Vanderstraeten–Haegeman–Verstraete, SciPost Phys. Lect. Notes 7) — concept pages should follow the pattern "physics concept → why it matters computationally → exact API mapping → pitfalls" and cite the notes for the derivations.
  <!-- REVIEW: maintainer to confirm the canonical companion reference(s) and whether quantumghent tutorial material should be linked as a teaching track. -->
- [ ] **Systematic bidirectional cross-linking between layers.** *(TeNPy)*
  TeNPy's `dmrg` module reference links *up* to the DMRG-protocol guide; guide pages link down to class references.
  Adopt as a checklist rule per page: every `lib/` category page links to its how-to and concept page and vice versa (Documenter `@ref`s, already the convention — the delta is doing it *systematically*, including from docstrings in `src/`).

---

## 4. Plan additions (maintainer sign-off needed)

- [ ] **A symptom-first Troubleshooting & FAQ page, separate from convergence troubleshooting.**
  *(TeNPy `troubleshooting.html` + ITensor's FAQ sidebar sections)*
  Headings phrased as symptoms — "I get an error when …", "the run is much slower than expected", "results changed after updating" — covering environment issues (threading/BLAS), precompilation latency, and common TensorKit space mismatches.
  FAQ sections capture forum-shaped knowledge before it has to be asked; symptom-first headings match how users search.
- [ ] **A failure-mode example page.** *(TeNPy's "Why you need the Mixer in DMRG" notebook)*
  An example whose entire point is a documented failure and its fix — e.g. single-site DMRG/VUMPS getting stuck at fixed bond dimension and how `changebonds`/two-site variants escape, shown with real (bad and good) convergence curves.
  Teaching a failure mode is more convincing than asserting a best practice.
  <!-- REVIEW: choice of failure mode and its physics framing is the maintainer's call. -->
- [ ] **An algorithm-options ("all the knobs") reference.** *(TeNPy's Config system — their reference layer's standout)*
  TeNPy auto-generates per-class options tables, marks inherited options with their origin, and maintains two global config indices; options are the *real* API surface of a physics code.
  MPSKit analogue: every algorithm struct's `lib/` entry gets a complete keyword table (meaning, default, when to touch it), shared knobs (`tol`, `maxiter`, `verbosity`) documented once and referenced, plus one index page listing every keyword across all algorithms.
  Start manual; consider generating from the structs later.
- [ ] **Document extension points with a reference page + paired recipe.** *(ITensor's Observer system docs)*
  ITensor documents its observer contract (what you may access, when hooks fire, result shapes) *and* ships a worked custom-observer recipe.
  MPSKit analogue: whatever hooks actually exist (e.g. the `finalize!` machinery, custom `changebonds` schedules, swapping Krylov solvers via alg structs) — each gets both a `lib/` contract description and a how-to recipe.
  Hard rule 1 applies doubly here: establish the real hook API via `api-explorer` before writing a word.
- [ ] **A "Papers using MPSKit" page.** *(TeNPy: ~311 papers; ITensor: ~800+)*
  Social proof, a discovery tool ("someone already did my model"), and a motivation loop for maintainers; populated via a GitHub issue template so it costs contributors one click.
  Group by topic or year from day one — TeNPy's flat chronological list of 311 entries has become unnavigable.
  Pairs with the planned `citing.md`: zero-friction BibTeX directly feeds this page.
- [ ] **A curated, opinionated literature page.** *(TeNPy `literature.html`)*
  Beyond the planned DocumenterCitations `references.md`: a short curated section with categories (introductions, algorithm papers, reviews, lecture notes) and one-line guidance ("start here", "encyclopedic but long"), so every `[cite]` in the docs resolves somewhere useful.
- [ ] **Designate one Q&A surface and link it from every page footer.** *(ITensor Discourse)*
  ITensor's forum has a "DMRG and Numerical Methods" category distinct from library questions — the forum hosts "is my physics right?" and its Google-indexed threads become permanent long-tail documentation.
  For MPSKit that likely means GitHub Discussions with a methods category; the docs action is a consistent "where to ask" pointer plus a habit of harvesting recurring questions back into the FAQ page.
  <!-- REVIEW: which surface (Discussions vs forum) is an org decision. -->

---

## 5. Needs package work first

- [ ] **Rich terminal display of states: bond dimensions + canonical form at a glance.** *(quimb's `.show()` — the single cheapest-per-payoff idea in the review)*
  quimb prints an ASCII schematic of an MPS with bond dimensions and orthogonality-center arrows; it renders in terminals, docstrings, and docs alike with zero dependencies, and the docs use it on nearly every page to *show* the object being discussed.
  An `MPS`/`show` method (or a `summary`-style function) printing per-bond dimensions and gauge structure would let every tutorial *display* what `FiniteMPS` did instead of describing it.
- [ ] **A self-illustrating docs culture.** *(quimb throughout; tensornetwork.org's diagram-heavy style)*
  quimb draws the tensor network under discussion, inline, on essentially every guide page — the docs double as the gallery for the visualization feature.
  Short of full TN drawing, MPSKit pages can standardize cheap plots where they teach: entanglement spectra, bond-dimension profiles along the chain, convergence-vs-sweep curves; concepts pages need actual tensor-diagram figures (static SVGs are fine).
- [ ] **Interop examples as first-class docs.** *(quimb's torch/jax/flax pages)*
  Examples embedding MPSKit in the wider ecosystem: JLD2/HDF5 serialization round-trip (feeds the planned `saving_loading.md`), CUDA.jl usage (marked experimental per hard rule 5), DifferentialEquations/Optimisers-style composition if and only if it genuinely works.
  Signals ecosystem citizenship and gets indexed by searches for the *other* package.

---

## 6. Anti-patterns observed (do not copy)

- **Mega-pages.** TeNPy's `np_conserved` intro is ~4000 lines mixing what a beginner must know with internal storage formats; split beginner path from internals across pages (directly relevant to how `symmetries.md` + `vector_spaces.md` get scoped).
- **Dead source listings.** Example code rendered without executed output (TeNPy scripts/toycodes) rots invisibly and teaches nothing about what running it looks like.
- **Committed rather than build-time-executed outputs** (quimb) — same rot, one level up.
- **Type-of-page ambiguity in the nav.** TeNPy's "Introductions" heading hides how-to, explanation, and reference under one label; users can't tell what kind of page they're clicking into.
- **Multi-site fragmentation and duplicated content.** ITensor's five loosely-joined surfaces scatter search results, and every C++/Julia duplicated formula ages independently.
- **Package-split churn breaking documented examples.** The ITensors.jl → ITensorMPS.jl extraction invalidated years of `using ITensors`-only snippets; when MPSKit moves API across package boundaries, budget an upgrade guide in the same PR.
- **Bare example indices** (quimb) and **flat unnavigable community lists** (TeNPy's papers page) — annotate and group from the start.
- **Landing pages without code** (quimb, TeNPy) — the plan's `index.md` example is the fix; don't let it get cut for space.

---

## Appendix: what each library is best at (one paragraph each)

**quimb** — presentation and immediacy.
Executed-notebook guides with real convergence output on every page, a self-illustrating `draw()`/`.show()` culture, a clean furo theme with dark mode and copy buttons, capability-grouped install instructions, and a changelog that hyperlinks into the API reference.
Weakest at: conceptual explanation (physics is assumed, never explained), example-index curation, and URL stability.

**TeNPy** — pedagogy and the options layer.
The DMRG-protocol methodology page, physics-pedagogy introductions that map every concept to an API object (Jordan-Wigner, charge conservation), a summer-school teaching track (toycodes + exercises + solutions + Colab), the Config/options system with per-class option tables and global indices, symptom-first troubleshooting, and the literature + papers-using pages.
Weakest at: page scoping (monoliths), landing page/README (no code), page-type labeling, and dead script listings.

**ITensor** — onboarding and the how-to layer.
Flagship DMRG with real output on the landing page, imperative task-titled code formulas that double as a search-term FAQ, first-class FAQ/upgrade-guide/operational docs (clusters, memory, HDF5), documented extension points (observers), a one-sentence design thesis repeated everywhere, and the citing → papers-using growth loop plus a methods-friendly Discourse.
Weakest at: fragmentation across five sites, C++/Julia duplication, and reference gaps for constructors.

### Key source pages

- TeNPy DMRG protocol: https://tenpy.readthedocs.io/en/latest/intro/dmrg-protocol.html
- TeNPy options system: https://tenpy.readthedocs.io/en/latest/intro/options.html
- TeNPy troubleshooting: https://tenpy.readthedocs.io/en/latest/troubleshooting.html
- TeNPy literature: https://tenpy.readthedocs.io/en/latest/literature.html
- quimb 1D algorithms (the `.show()` + real-output pattern): https://quimb.readthedocs.io/en/latest/tensor/tensor-1d.html
- quimb design/data-model explanation page: https://quimb.readthedocs.io/en/latest/tensor/tensor-design.html
- ITensor code formulas: https://itensor.org/docs.cgi?page=formulas
- ITensorMPS.jl landing page (DMRG on page one): https://itensor.github.io/ITensorMPS.jl/dev/
- ITensorMPS.jl recipe pages: https://itensor.github.io/ITensorMPS.jl/dev/examples/DMRG.html
- tensornetwork.org MPS article (concept-page gold standard): https://tensornetwork.org/mps/
