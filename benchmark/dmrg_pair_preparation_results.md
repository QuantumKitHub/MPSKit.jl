Pair-only DMRG2 preparation, 2026-10-07.

Plain DMRG2 now skips completed one-site AC preparation. The implementation is committed in `29395193`; all 594 focused cache assertions pass. In the chemistry-like χ=64 case, this removes 116 MiB of cumulative allocations from the measured second sweep (12.1%) and reduces separately measured operator preparation time by about 19%. Eight consecutive sweeps take 2.1% less wall time than the previous cache policy. GC still makes short timing comparisons variable, and ordinary environments remain faster in total over this particular eight-sweep sequence.

The cache has explicit `one_site` and `two_site` preparation flags. Plain DMRG2 selects pairs only; single-site DMRG selects AC, and DMRG with expansion selects both. Jordan preparation first constructs shared raw B/C/I/E pieces and the unfused continuing contraction. Completing AC then becomes an optional step, independent of completing AC2. Generic MPOs keep the left fused contraction needed by AC2 but skip an unused right dense environment. Explicit AC queries on a pair-only cache construct from its stored GL/GR snapshots without publishing AC records. Full-MPO transfers, pair algebra, snapshot ownership, and eigensolver settings are preserved.

The comparison uses three paths in the same process:

| Path | Environment and preparation policy |
|---|---|
| Ordinary | Existing ordinary environments and construction at each query |
| Both | Hybrid cache with the previous policy: complete AC and AC2 |
| Pairs | Hybrid cache with the new plain-DMRG2 policy: complete AC2 only |

Three timed second sweeps per path start from the same initial state, with GC before setup and no forced GC between the first and second sweeps. Path order rotates between samples. Setup and the first sweep are excluded from this table. Times include GC; the last time column is the median of each sample's time minus its measured GC time. Allocation numbers are exact cumulative allocated bytes, not retained or peak memory. Sampling instrumentation runs separately.

| Path | Median sweep (s) | Observed range (s) | Median GC (s) | Median excluding GC (s) | Allocations (MiB) |
|---|---:|---:|---:|---:|---:|
| Ordinary | 4.146 | 3.342–4.345 | 1.072 | 3.079 | 836 |
| Both | 3.273 | 3.059–4.064 | 0.207 | 3.012 | 964 |
| Pairs | 4.143 | 4.125–4.209 | 1.233 | 2.916 | 848 |

The pairs path is cheaper excluding measured GC and allocates less than the both path, yet its median total is higher in this small sample. The medians are not evidence of a reliable wall-time regression or speedup: collection timing varies substantially. Compared with ordinary construction, pairs allocates about 1.4% more in this second sweep. The consecutive-sweep experiment below gives a longer comparison without resetting collection between sweeps.

A separate adapter splits advancement into transfers, shared ingredients plus optional AC completion, and AC2 completion. Three samples per cached policy request GC immediately before the measured second sweep. Each adapter run is checked against its unwrapped production path for state agreement, local errors, and identical eigensolver application counts. The table gives median milliseconds / exact cumulative MiB; medians need not sum to the median total. The raw timer label `one-site preparation` now includes ingredients needed by pairs even when AC completion is disabled.

| Stage | Both ms / MiB | Pairs ms / MiB |
|---|---:|---:|
| Full-MPO transfers | 174.5 / 111.7 | 168.4 / 112.8 |
| Shared ingredients plus optional AC completion | 229.8 / 469.4 | 156.6 / 319.0 |
| AC2 completion | 231.1 / 342.7 | 215.5 / 375.8 |
| Final effective-operator assembly | 0.11 / 0.014 | 0.11 / 0.014 |

The two preparation stages together fall from approximately 461 to 372 ms and from 812 to 695 MiB. This is why the entire former 469 MiB stage could not be eliminated: 319 MiB remains for shared ingredients. AC2 completion's measured allocations increase despite unchanged pair algebra; these measurements do not establish the reason for that stage-level change. Combined preparation and complete-sweep allocation totals establish the net savings. Transfer allocations also remain higher than the ordinary path measured in the [previous report](dmrg_hybrid_profile_results.md); this change does not resolve that separate issue.

The main, unwrapped timing runs measure eigensolve medians of 2.248 s (ordinary), 2.243 s (both), and 2.238 s (pairs), with approximately 30.9 MiB allocated in each. All three diagnostic second sweeps use 163 eigensolver applications. Normalized overlaps differ from one by at most 2e-15, and maximum local-error differences are below 9e-16. This supports attributing the preparation savings to the cache policy rather than different solver work.

Eight consecutive complete sweeps use the same initial state, with GC before setup and no forced collections between sweeps. Setup is excluded; the first sweep is included. This is a progression toward convergence rather than eight independent repeats, and the three final states agree. It is one sequence per path, so the table does not establish statistical significance.

| Path | Eight sweeps (s) | GC (s) | Excluding measured GC (s) | Cumulative allocations (GiB) |
|---|---:|---:|---:|---:|
| Ordinary | 28.464 | 2.898 | 25.567 | 7.064 |
| Both | 30.065 | 5.396 | 24.669 | 7.644 |
| Pairs | 29.444 | 5.376 | 24.068 | 6.736 |

Relative to both, pairs saves 0.621 s overall and 0.601 s excluding GC, while allocating 11.9% less. GC time is essentially unchanged in this sequence, so reduced allocation volume has not yet produced a clear collection-time benefit. Relative to ordinary, pairs saves 1.499 s excluding GC but spends 2.479 s more in GC, leaving total time 0.980 s higher. The implementation removes unused work; GC remains a separate performance issue.

Allocation stack sampling at rate 0.01 runs independently of timing. It samples uniformly by allocation event, so sampled byte totals are not estimates of exact stage bytes. The event counts are location diagnostics from one profile per cached policy:

| Location | Both sampled events | Pairs sampled events |
|---|---:|---:|
| All locations | 21,109 | 20,167 |
| Operator preparation | 11,067 | 10,284 |
| Environment transfers | 9,409 | 9,252 |

Frequent library allocation sites remain `TensorKit.taskforeach`, `LRUCache._unsafe_getindex`, `Strided._mapreduce_block!`, and `LRUCache.get!`. Removing unused AC completion reduces buffer bytes more strongly than allocation-event counts. The next candidates remain the dependency metadata fast paths identified in the previous report and the transfer-allocation excess; neither is changed here. Previous numerical CPU profiles locate the main time cost in effective-Hamiltonian applications during eigensolve. This experiment measures preparation time directly and does not repeat CPU stack sampling.

The benchmark uses Julia 1.13.1 on Xeon Gold 6244, Float64 trivial planar tensors, one numerical Julia thread, one BLAS thread, and the serial scheduler. The seeded 14-orbital spinless chemistry surrogate has 2289 terms and maximum MPO bond 613; the MPS rank cap is 64. Solver settings are tol=1e-8, krylovdim=20, maxiter=4, with adaptive tolerance disabled. The operator is fixed and the finalizer is read-only. No disk offloading is used. Nearest-neighbor and sparse-operator measurements from before this targeted DMRG2 change remain in the previous report; they are not rerun here.

Reproduce with [dmrg_pair_preparation.jl](dmrg_pair_preparation.jl):

```sh
jld --project=test --name=dmrg-pair --idle-timeout=2h --timeout=1800 eval --scratch 'include("benchmark/dmrg_pair_preparation.jl"); benchmark_dmrg_pair_preparation(); nothing'
```

Raw results: [timings including setup and first sweep](dmrg_pair_preparation_results.csv), [unwrapped stages](dmrg_pair_preparation_results_stages.csv), [state and solver agreement](dmrg_pair_preparation_results_agreement.csv), [separated construction](dmrg_pair_preparation_results_construction.csv), [consecutive sweeps](dmrg_pair_preparation_results_continuous.csv), and [sampled allocation sites](dmrg_pair_preparation_results_allocations.csv).
