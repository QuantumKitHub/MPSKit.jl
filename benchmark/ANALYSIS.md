# MPSKit vs ITensorMPS — investigation notes

Status: **skeleton — no full-run results yet.** This file is filled in only from real
Rusty result JSONs in `results/` (never from smoke runs, never from memory). Every claim
must cite the result file(s) it comes from.

## Questions this round must answer

### Q1. Sanity gates (prerequisite for everything below)

- [ ] Suites 1-2: do MPSKit and ITensorMPS converge to the same energy (~1e-8 relative)
      at every matched χ? (`compare_dmrg_suite` table in `plot_results.jl` output.)
- [ ] Suite 5: do the ⟨Sz⟩(t) trajectories coincide at matched χ?
- [ ] Metadata parity: same node type, same `nthreads_julia` / `nthreads_blas` /
      `blas_config` on both sides of every compared pair?

### Q2. Headline: time-to-accuracy (suites 1-2)

- [ ] At each χ, which library reaches E_ref + 1e-8·|E_ref| first, and by what factor?
- [ ] Does the ratio change with χ (i.e. is the crossover at small or large χ)?
- [ ] Does the U(1) suite change the picture relative to the trivial suite (block-sparse
      overhead vs dense throughput)?
- [ ] Convergence-quality check: does either side plateau above the gate at some χ
      (local minimum), and does that correlate with the fixed-χ no-noise protocol?

### Q3. TDVP throughput (suite 5)

- [ ] Seconds of wall time per unit physical time vs χ, per library — who wins, and does
      the scaling exponent in χ differ (should be ~χ³; deviations are implementation
      overhead)?
- [ ] How large is the growth phase (two-site) cost relative to the measure phase?

### Q4. Thread scaling (suite 7)

- [ ] Where does each library peak on the (julia-threads × BLAS-threads) grid?
- [ ] MPSKit: how much does pure Julia-threading ((8,1) vs (1,1)) buy on this dense
      workload? ITensor: same question for BLAS ((1,8) vs (1,1)).
- [ ] Do mixed settings ((4,4), (8,8)) oversubscribe and regress?

### Q5. Where does the time go? (diagnostics)

- [ ] GC fraction and allocation volume per sweep, per library, vs χ (`gctimes` /
      `allocd_bytes` fields). Is either side GC-bound at any χ?
- [ ] `profile_sweep.jl` on the interesting χ points (pick after Q2): what fraction of a
      sweep is BLAS / SVD / eigsolve vs MPSKit's own environment and permutation code?

## Findings

_(empty — populate per question above, with result-file citations, after the Rusty runs.)_

## Follow-ups queued for later rounds

- χ-ramp warm-start protocol (both libraries) instead of fresh random starts per χ.
- SU(2) capability suite (MPSKit-only, vs its own U(1) run).
- TeNPy comparisons; quasi-2D cylinder workload.
- Publishing: populate `docs/src/benchmarks.md` + README chart only after the sanity
  gates pass and the maintainer has reviewed the parity tables.
