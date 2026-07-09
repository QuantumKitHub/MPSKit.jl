# [Parallelism and GPU support](@id howto_parallelism_gpu)

This page collects the practical knobs for controlling how MPSKit uses the hardware:
how to set BLAS threads, how to pick the MPSKit multithreading scheduler, how to inspect
the resulting thread layout, and what to do when a calculation runs out of memory.
It closes with a short, experimental note on moving states onto a GPU.

For the reasoning behind these settings — why Julia threads and BLAS threads interact the
way they do, and where MPSKit actually parallelizes — see
[The parallelism model](@ref concept_parallelism_model).

!!! note
    Threading performance depends heavily on the hardware, the BLAS vendor, the size of
    the problem, and the availability of memory and memory bandwidth.
    There is no single setting that is optimal everywhere; the recipes below are sensible
    starting points, and you should measure on your own machine.

## Setting the number of BLAS threads

Most of the heavy linear algebra in MPSKit ends up in BLAS routines (in particular `gemm`,
general matrix-matrix multiplication).
The number of BLAS threads is controlled through `LinearAlgebra.BLAS.set_num_threads`:

```julia
using LinearAlgebra: BLAS
BLAS.set_num_threads(1)
```

With OpenBLAS (the default vendor), `set_num_threads` sets the **total** number of BLAS
threads held in a shared pool across all Julia threads.
When Julia is started with multiple threads, setting this to `1` lets MPSKit drive the
parallelism through its own (Julia-thread) machinery instead, which is often the best
option for OpenBLAS.

With [MKL.jl](https://github.com/JuliaLinearAlgebra/MKL.jl) the semantics differ: the BLAS
thread count applies **per Julia thread**, so 4 Julia threads with 4 BLAS threads each spawn
16 BLAS threads in total.
In that case you typically want to lower the BLAS thread count to avoid oversubscribing the
physical cores.

## Setting the MPSKit scheduler

When Julia runs with multiple threads, MPSKit parallelizes parts of its algorithms through
[OhMyThreads.jl](https://juliafolds2.github.io/OhMyThreads.jl/stable/).
The behaviour is controlled by a global scheduler, set with `MPSKit.Defaults.set_scheduler!`:

```julia
MPSKit.Defaults.set_scheduler!(:serial)  # disable multithreading
MPSKit.Defaults.set_scheduler!(:greedy)  # multithreading with greedy load-balancing
MPSKit.Defaults.set_scheduler!(:dynamic) # multithreading with dynamic load-balancing
```

`set_scheduler!` also accepts an `OhMyThreads.Scheduler` directly, or a symbol together with
keyword arguments that are forwarded to the corresponding OhMyThreads scheduler.
When left unset, the default is a serial scheduler if Julia was started with a single thread,
and a dynamic scheduler otherwise.
For the full list of schedulers and their keyword arguments, see the
[OhMyThreads.jl documentation](https://juliafolds2.github.io/OhMyThreads.jl/stable/refs/api/#Schedulers).

## Diagnosing the thread layout

Because the interaction between Julia threads and BLAS threads is easy to get wrong, it helps
to inspect the actual layout.
[ThreadPinning.jl](https://github.com/carstenbauer/ThreadPinning.jl) provides `threadinfo`,
which reports the Julia threads, their CPU mapping, and the BLAS backend and thread count:

```julia-repl
julia> Threads.nthreads()
4

julia> using ThreadPinning; threadinfo(; blas = true, hints = true)

System: 8 cores (2-way SMT), 1 sockets, 1 NUMA domains

| 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 |

# = Julia thread, # = HT, # = Julia thread on HT, | = Socket separator

Julia threads: 4
├ Occupied CPU-threads: 4
└ Mapping (Thread => CPUID): 1 => 8, 2 => 5, 3 => 9, 4 => 2,

BLAS: libopenblas64_.so
└ openblas_get_num_threads: 8

[ Info: jlthreads != 1 && blasthreads < cputhreads. You should either set BLAS.set_num_threads(1) (recommended!) or at least BLAS.set_num_threads(16).
[ Info: jlthreads < cputhreads. Perhaps increase number of Julia threads to 16?
```

Passing `hints = true` makes ThreadPinning emit the advisory messages shown above.
Loading a different BLAS backend changes the report; with MKL, for example, `threadinfo`
reports `libmkl_rt.so` and warns when the per-Julia-thread BLAS thread count exceeds the
available CPU threads per Julia thread:

```julia-repl
julia> using MKL; threadinfo(; blas = true, hints = true)

System: 8 cores (2-way SMT), 1 sockets, 1 NUMA domains

| 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 |

# = Julia thread, # = HT, # = Julia thread on HT, | = Socket separator

Julia threads: 4
├ Occupied CPU-threads: 4
└ Mapping (Thread => CPUID): 1 => 11, 2 => 12, 3 => 1, 4 => 2,

BLAS: libmkl_rt.so
├ mkl_get_num_threads: 8
└ mkl_get_dynamic: true

┌ Warning: blasthreads_per_jlthread > cputhreads_per_jlthread. You should decrease the number of MKL threads, i.e. BLAS.set_num_threads(4).
└ @ ThreadPinning ~/.julia/packages/ThreadPinning/qV2Cd/src/threadinfo.jl:256
[ Info: jlthreads < cputhreads. Perhaps increase number of Julia threads to 16?
```

## Reducing memory usage

MPSKit's multithreading spawns tasks in a nested fashion, each allocating and deallocating
memory in a tight loop.
This can put enough pressure on the garbage collector that memory usage climbs and, in the
worst case, an `OutOfMemory` error occurs before the garbage can be cleared.

If you hit this, the most effective remedy is usually to disable MPSKit's multithreading,
by setting the scheduler to serial:

```julia
MPSKit.Defaults.set_scheduler!(:serial)
```

The `derivatives` (the effective local operators applied during the sweeps) are reported to
be the most memory-intensive part, so this is where switching off multithreading helps most.

For why this pressure arises, see
[Why memory pressure arises](@ref concept_parallelism_model).

## GPU support

!!! warning "Experimental"
    GPU support in MPSKit is **experimental and minimal**.
    There are no GPU-specific algorithms, kernels, or tuning options; the only surface is
    an [Adapt.jl](https://github.com/JuliaGPU/Adapt.jl)-based mechanism for moving a state
    or operator onto a different array type.
    Treat this as preparatory infrastructure rather than a supported workflow.

The package extension `MPSKitAdaptExt` defines `Adapt.adapt_structure` for `FiniteMPS`,
`InfiniteMPS`, `MPO`, and `MPOHamiltonian`.
This lets you convert the underlying tensors to a GPU array type with `Adapt.adapt`, after
which the algorithms dispatch through that array type.
A move onto a CUDA array would look like the following:

```julia
using Adapt, CUDA
ψ_gpu = adapt(CuArray, ψ)
```

