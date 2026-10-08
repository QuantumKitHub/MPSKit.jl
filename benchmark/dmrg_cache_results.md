DMRG cache measurements on 2026-10-06, implementation commit `59edccd0`.

The complete cache gives modest single-site gains here, but does not consistently improve two-site sweeps. Sparse two-site sweeps regress, with higher allocations and garbage-collection time. The latest `remainder` optimization is a clear improvement in construction time and allocations, measured separately below.

Reproduction and raw data: [benchmark script](dmrg_cache.jl), [individual samples](dmrg_cache_results.csv), and [diagnostic profile](dmrg_cache_profile.txt).

Run from the repository root:

```sh
jld --project=test --name=dmrg-bench --idle-timeout=2h --timeout=1200 eval --scratch 'include("benchmark/dmrg_cache.jl"); benchmark_dmrg_cache(); nothing'
```

Configuration uses `DMRG_BENCH_SAMPLES` (default 5), `DMRG_BENCH_DIMS` (default `16,32`), and `DMRG_BENCH_OUTPUT`. Set these inside `withenv(...)` in the Julia evaluation when using an already-running daemon. The separate diagnostic is `benchmark_dmrg_cache_profile()`. Default outputs go to the ignored `benchmark/results` directory; the linked files are this run’s saved snapshots.

Measurements used Julia 1.13.1, TensorKit 0.17.2, BlockTensorKit 0.3.20, and an Intel Xeon Gold 6244 at 3.60 GHz. Julia, BLAS, and the MPSKit scheduler each used one thread. Tensors were Float64 with planar trivial sectors, without symmetry reductions. No disk offloading was used.

**Operators.** All were built with the ordinary finite Hamiltonian constructor and fixed seeded coefficients. The chemistry-like surrogate contains onsite number operators, all-pairs hopping and density interactions, and `c†ᵢ c†ⱼ cₖ cₗ + h.c.` for every `i < j < k < l`. Jordan–Wigner strings and signs are explicit. This is not a molecular-integral benchmark or a specialized complementary-operator chemistry MPO.

| Operator | Sites | Terms | Maximum MPO bond | Nonzero blocks | Block density |
|---|---:|---:|---:|---:|---:|
| Nearest neighbor | 24 | 47 | 3 | 116 | 56.86% |
| Large sparse | 24 | 197 | 126 | 1671 | 0.89% |
| Chemistry-like | 12 | 1200 | 365 | 2740 | 0.83% |

The nearest-neighbor model is transverse-field Ising. The large sparse model adds 150 distinct, seeded four-site `X Z Z X` interactions with long gaps. Block density includes tensor blocks and scalar identity blocks over all sites, divided by the number of possible virtual-channel blocks.

**Comparison.** Both paths run the same complete-sweep iterator. The ordinary path bypasses `DMRGSweepCache` and uses the existing finite environments and derivative constructors. Initial states, truncation ranks, and eigensolver settings match: non-adaptive Lanczos, fixed tolerance `1e-8`, Krylov dimension 20, and maximum 4 iterations. Setup includes environments and iterator/cache construction, but excludes copying the initial MPS and constructing the Hamiltonian. Every sample starts from that same initial MPS, then performs two complete forward/backward sweeps. Ordinary environments defer work into the first sweep; cached initialization prepares the right-hand records eagerly.

Both paths were warmed before sampling, excluding compilation from the recorded runs. GC was requested before each sample, outside the timed sections; GC occurring inside a section remains included. Path order alternated between samples. Tables report medians of five samples. All 12 warm comparison cases passed overlap agreement within `1e-8` and local-error agreement within `1e-7` after two sweeps.

**Second-sweep timing and allocations.** Ratios greater than 1 mean the cache is faster. Allocations are cumulative bytes allocated during the sweep, not retained or peak memory.

| Operator | χ | Algorithm | Ordinary (ms) | Cached (ms) | Speed ratio | Ordinary → cached allocation (MiB) |
|---|---:|---|---:|---:|---:|---:|
| Nearest neighbor | 16 | DMRG | 16.32 | 11.81 | 1.38× | 10.7 → 7.3 |
| Nearest neighbor | 16 | DMRG2 | 23.76 | 20.85 | 1.14× | 14.0 → 12.4 |
| Nearest neighbor | 32 | DMRG | 23.50 | 16.41 | 1.43× | 18.9 → 13.5 |
| Nearest neighbor | 32 | DMRG2 | 35.74 | 32.79 | 1.09× | 26.4 → 25.3 |
| Large sparse | 16 | DMRG | 387.50 | 344.99 | 1.12× | 269.9 → 219.2 |
| Large sparse | 16 | DMRG2 | 528.14 | 574.26 | 0.92× | 254.9 → 295.4 |
| Large sparse | 32 | DMRG | 1183.06 | 1183.36 | 1.00× | 672.6 → 586.6 |
| Large sparse | 32 | DMRG2 | 2299.26 | 2650.77 | 0.87× | 656.0 → 822.2 |
| Chemistry-like | 16 | DMRG | 205.08 | 160.57 | 1.28× | 153.1 → 112.7 |
| Chemistry-like | 16 | DMRG2 | 282.26 | 246.63 | 1.14× | 159.2 → 147.3 |
| Chemistry-like | 32 | DMRG | 308.46 | 258.49 | 1.19× | 209.9 → 179.9 |
| Chemistry-like | 32 | DMRG2 | 436.98 | 437.67 | 1.00× | 219.8 → 246.0 |

**Setup-inclusive cost.** The two-sweep total is the median of each sample’s setup + first sweep + second sweep, rather than the sum of separate medians. Small differences around 1% should be treated as effectively unchanged.

| Operator | χ | Algorithm | Cached setup (ms) | Ordinary first sweep (ms) | Cached first sweep (ms) | Ordinary → cached two-sweep total (ms) | Total speed ratio |
|---|---:|---|---:|---:|---:|---:|---:|
| Nearest neighbor | 16 | DMRG | 6.10 | 62.13 | 54.32 | 78.54 → 72.27 | 1.09× |
| Nearest neighbor | 16 | DMRG2 | 6.05 | 77.73 | 73.15 | 101.27 → 99.98 | 1.01× |
| Nearest neighbor | 32 | DMRG | 7.91 | 108.72 | 97.00 | 132.49 → 121.24 | 1.09× |
| Nearest neighbor | 32 | DMRG2 | 9.50 | 186.40 | 178.82 | 221.88 → 221.11 | 1.00× |
| Large sparse | 16 | DMRG | 79.24 | 473.47 | 356.83 | 861.46 → 775.93 | 1.11× |
| Large sparse | 16 | DMRG2 | 95.00 | 654.43 | 593.28 | 1184.59 → 1258.37 | 0.94× |
| Large sparse | 32 | DMRG | 144.77 | 1516.09 | 1456.30 | 2698.19 → 2781.87 | 0.97× |
| Large sparse | 32 | DMRG2 | 223.16 | 2830.48 | 2839.97 | 5049.78 → 5724.50 | 0.88× |
| Chemistry-like | 16 | DMRG | 56.90 | 260.62 | 180.94 | 466.69 → 399.68 | 1.17× |
| Chemistry-like | 16 | DMRG2 | 62.68 | 329.91 | 266.01 | 608.68 → 574.96 | 1.06× |
| Chemistry-like | 32 | DMRG | 68.77 | 419.75 | 334.21 | 730.30 → 660.49 | 1.11× |
| Chemistry-like | 32 | DMRG2 | 82.64 | 542.95 | 471.12 | 974.25 → 990.94 | 0.98× |

Ordinary setup medians were 0.06–0.11 ms. At χ=32 the sparse DMRG2 second-sweep samples ranged from 2.20–2.45 s ordinarily and 2.41–3.81 s with the cache. Recorded GC time ranged from 0.035–0.101 s ordinarily and 0.220–1.388 s with the cache. Even subtracting GC time sample by sample, median sweep time was higher with the cache; GC is only part of the regression.

**Where the two-site cost goes.** The separate instrumented profile measures one warmed second sweep at χ=32 per path; it is diagnostic, not another median benchmark. Ordinary lazy environment work appears inside `AC2_hamiltonian`; cached transfer and record preparation appear inside `advance_env`. The constructor itself becomes almost free, but that does not imply an equally large reduction in total sweep work.

| Operator | Ordinary `AC2_hamiltonian` (ms) | Cached `advance_env` (ms) | Cached `AC2_hamiltonian` (ms) | Ordinary constructor → cached advancement allocations (MiB) |
|---|---:|---:|---:|---:|
| Nearest neighbor | 15.8 | 11.7 | 0.102 | 8.68 → 7.53 |
| Large sparse | 414 | 494 | 0.233 | 517 → 739 |
| Chemistry-like | 215 | 190 | 0.109 | 209 → 235 |

The sparse profile points to record preparation as a concrete next target. Currently DMRG2 prepares both one-site and two-site sides on advancement. The two-site preparation reads the one-site raw blocks and unfused continuing contraction, whereas one-site preparation also builds a prepared copy and a fused continuing representation. Auditing which of those extra one-site products DMRG2 actually needs could reduce the preparation and allocation overhead. These benchmarks do not yet isolate that possible optimization.

**The `remainder` change alone.** This compares the previous deep-copy-then-delete construction against independent containers sharing retained tensor blocks. Each sample measures 20 whole-MPO constructions and divides by 20. Equality of the resulting remainders is checked. This isolates only `remainder`, not the other operator metadata or the full cache.

| Operator | Copy/delete (ms) | Share/filter (ms) | Speed ratio | Copy/delete → share/filter allocations (MiB) |
|---|---:|---:|---:|---:|
| Nearest neighbor | 0.029 | 0.016 | 1.81× | 0.087 → 0.065 |
| Large sparse | 0.736 | 0.062 | 11.77× | 1.064 → 0.130 |
| Chemistry-like | 1.361 | 0.226 | 6.03× | 2.066 → 0.554 |

The sweep ratios above compare the entire cache to ordinary environments; they do not isolate the latest `remainder` change. Its construction savings occur once at initialization, while the full-cache comparison also includes contraction reuse, record preparation, transfers, eigensolves, and gauging. These small and medium bond dimensions and synthetic channel graphs do not establish performance for large molecular calculations.
