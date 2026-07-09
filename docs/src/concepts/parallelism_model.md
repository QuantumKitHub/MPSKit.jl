# [The parallelism model](@id concept_parallelism_model)

Julia has excellent [parallelism infrastructure](https://julialang.org/blog/2019/07/multithreading/),
but there is a caveat that touches every algorithm in MPSKit: Julia's own threads do not
compose cleanly with the threads that BLAS uses internally for linear algebra.
Since `gemm` (general matrix-matrix multiplication) is a core routine throughout MPSKit,
this interaction has a real effect on performance.

This page explains the model behind the settings, so that the recipes on
[Parallelism and GPU support](@ref howto_parallelism_gpu) are more than a list of magic
incantations.

## Julia threads versus BLAS threads

Much of the confusion here comes from the fact that BLAS threading behaviour is not
consistent between vendors, and that performance depends strongly on the hardware, the
specifics of the problem, and the availability of resources such as total memory and memory
bandwidth.
There is no one-size-fits-all setting, which is why the how-to page frames its advice as
starting points to be measured rather than guarantees.

The two vendors most commonly used with Julia treat the BLAS thread count differently.
With OpenBLAS (the default), the configured number of BLAS threads is the **total** size of a
single thread pool that is shared by all Julia threads: 4 Julia threads and 4 BLAS threads
means all 4 Julia threads draw from the same pool of 4 BLAS threads.
Setting the BLAS thread count to `1` instead frees OpenBLAS to run its work on the Julia
threads themselves, so that MPSKit's Julia-level parallelism is the thing that scales.
<!-- REVIEW: description of OpenBLAS sharing a single BLAS thread pool across Julia threads, and that set_num_threads(1) hands work to the Julia threads, inherited from man/parallelism.md; confirm the mechanism. -->

With [MKL.jl](https://github.com/JuliaLinearAlgebra/MKL.jl), which often outperforms OpenBLAS,
the count is instead the number of threads spawned by **each** Julia thread: 4 Julia threads
with 4 BLAS threads each gives 16 BLAS threads in total.
Getting this wrong oversubscribes the physical cores — more software threads than hardware
can run — which degrades rather than improves performance.
<!-- REVIEW: "MKL often outperforms OpenBLAS" is a performance comparison inherited from man/parallelism.md; confirm before shipping. -->

## Where MPSKit parallelizes

When Julia is started with more than one thread, MPSKit uses
[OhMyThreads.jl](https://juliafolds2.github.io/OhMyThreads.jl/stable/) to parallelize its
algorithms wherever possible.
In practice this happens where a unit cell (or a chain of sites) lets local updates run
independently: the work is distributed across the sites of the system, with the tensor at
each site updated in parallel.This is exactly why setting the BLAS thread count to `1` on OpenBLAS tends to help: it keeps
the Julia threads free to work through the sites, rather than contending with a shared BLAS
pool.

The amount of speedup you can expect therefore tracks how much independent per-site work an
algorithm exposes.
<!-- REVIEW: framing scaling as proportional to the amount of independent per-site work is a synthesis, not stated verbatim in man/parallelism.md; confirm or soften. -->

## Parallelism over symmetry sectors

There is a second, orthogonal layer of parallelism for tensors that carry an internal
symmetry.
Such tensors are block-diagonal over their symmetry sectors, and the work can be spread
across those blocks.
This is handled by [TensorKit](https://quantumkithub.github.io/TensorKit.jl/stable/) at the
level of the individual tensor operations, below MPSKit's site-level parallelism, so the two
layers are independent of one another.
<!-- REVIEW: man/parallelism.md defers sector-level parallelism detail to TensorKit ("more information can be found in its documentation (Soon TM)"); the block-diagonal-over-sectors framing is standard but confirm TensorKit indeed parallelizes over sectors as stated. -->

## Why memory pressure arises

The same task-based parallelism that speeds MPSKit up can also drive its memory usage high.
The algorithms spawn tasks in a nested fashion, and each of those tasks allocates and
deallocates a fair amount of memory in a tight loop.
This can produce enough garbage, quickly enough, that the garbage collector cannot keep up;
in the worst case memory is exhausted and an `OutOfMemory` error is thrown before the garbage
can be cleared.
<!-- REVIEW: the nested-task allocation / GC-cannot-keep-up mechanism is described in man/parallelism.md as observed behaviour under investigation; confirm it still holds and is not yet fixed. -->

The most memory-intensive step is reportedly the application of the `derivatives` — the
effective local operators built during the sweeps — which is why the practical mitigation is
to disable MPSKit's multithreading there.
<!-- REVIEW: claim that the `derivatives` dominate memory use, inherited from man/parallelism.md; confirm. -->
The concrete recipe for doing so is on
[Parallelism and GPU support](@ref howto_parallelism_gpu).
