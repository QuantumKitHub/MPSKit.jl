# Parallelism in julia

Julia has great
[parallelism infrastructure](https://julialang.org/blog/2019/07/multithreading/), but there
is a caveat that is relevant for all algorithms implemented in MPSKit. The Julia threads do
not play nicely together with the BLAS threads, which are the threads used for many of the
linear algebra routines, and in particular for `gemm` (general matrix-matrix
multiplication). As this is a core routine in MPSKit, this has a significant impact on the
overall performance.

## Julia threads vs BLAS threads

A lot of the confusion stems from the fact that the BLAS threading behaviour is not
consistent between different vendors. Additionally, performance behaviour is severely
dependent on hardware, the specifics of the problem, and the availability of other resources
such as total memory, or memory bandwith. This means that there is no one size fits all
solution, and that you will have to experiment with the settings to get optimal performance.
Nevertheless, there are some general guidelines that can be followed, which seem to at least
work well in most cases.

The number of threads that are set by `BLAS.set_num_threads()`, in the case of OpenBLAS (the
default vendor), is equal to the **total number** of BLAS threads that is kept in a pool,
which is then shared by all Julia threads. This means that if you have 4 julia threads and 4
BLAS threads, then all julia threads will share the same 4 BLAS threads. On the other hand,
using `BLAS.set_num_threads(1)`, OpenBLAS will now utilize the julia threads to run the BLAS
jobs. Thus, for OpenBLAS, very often setting the number of BLAS threads to 1 is the best
option, which will then maximally utilize the julia threading infrastructure of MPSKit.

In the case of [MKL.jl](), which often outperforms OpenBLAS, the situation is a bit
different. Here, the number of BLAS threads corresponds to the number of threads that are
spawned by **each** julia thread. Thus, if you have 4 julia threads and 4 BLAS threads, then
each julia thread will spawn 4 BLAS threads, for a total of 16 BLAS threads. As such, it
might become necessary to adapt the settings to avoid oversubscription of the cores.

A careful analysis of the different cases and benefits can be inspected by making use of
[`ThreadPinning.jl`](https://github.com/carstenbauer/ThreadPinning.jl)'s tool
`threadinfo(; blas=true, info=true)`. In particular, the following might demonstrate the
difference between OpenBLAS and MKL:

```julia-repl
julia> Threads.nthreads()
4

julia> using ThreadPinning; threadinfo(; blas=true, hints=true)

System: 8 cores (2-way SMT), 1 sockets, 1 NUMA domains

| 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 | 

# = Julia thread, # = HT, # = Julia thread on HT, | = Socket seperator

Julia threads: 4
├ Occupied CPU-threads: 4
└ Mapping (Thread => CPUID): 1 => 8, 2 => 5, 3 => 9, 4 => 2,

BLAS: libopenblas64_.so
└ openblas_get_num_threads: 8

[ Info: jlthreads != 1 && blasthreads < cputhreads. You should either set BLAS.set_num_threads(1) (recommended!) or at least BLAS.set_num_threads(16).
[ Info: jlthreads < cputhreads. Perhaps increase number of Julia threads to 16?
julia> using MKL; threadinfo(; blas=true, hints=true)

System: 8 cores (2-way SMT), 1 sockets, 1 NUMA domains

| 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 | 

# = Julia thread, # = HT, # = Julia thread on HT, | = Socket seperator

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

## MPSKit multithreading

Within MPSKit, when Julia is started with multiple threads, by default the `Threads.@spawn`
machinery will be used to parallelize the code as much as possible. In particular, there are
three main places where this is happening, which can be disabled separately through a preference-based system.

1. During the process of some algorithms (e.g. VUMPS), local updates can take place at each
   site in parallel. This can be controlled by the `parallelize_sites` preference.

2. During the calculation of the environments, when the MPO is block-sparse, it is possible
   to parallelize over these blocks. This can be enabled or disabled by the
   `parallelize_transfers` preference. (Note that left- and right environments will always
   be computed in parallel)

3. During the calculation of the derivatives, when the MPO is block-sparse, it is possible
   to parallelize over these blocks. This can be enabled or disabled by the
   `parallelize_derivatives` preference.

For convenience, these preferences can be set via [`MPSKit.Defaults.set_parallelization`](@ref), which takes as input pairs of preferences and booleans. For example, to disable all parallelization, one can call

```julia
Defaults.set_parallelization("sites" => false, "transfers" => false, "derivatives" => false)
```

!!! warning
    These settings are statically set at compile-time, and for changes to take
    effect the Julia session must be restarted.

## TensorKit multithreading

Finally, when dealing with tensors that have some internal symmetry, it is also possible to
parallelize over the symmetry sectors. This is handled by TensorKit, and more information
can be found in its documentation (Soon TM).

## Memory management

Because of the way julia threads work, it is possible that the total memory usage of your
program becomes rather high. This seems to be because of the fact that MPSKit spawns several
tasks (in a nested way), which each allocate and deallocate quite a bit of memory in a tight
loop. This seems to lead to a situation where the garbage collector is not able to keep up,
and can even fail to clear the garbage before an `OutOfMemory` error occurs. In this case,
often the best thing to do is disable the multithreading of MPSKit, specifically for the
`derivatives`, as this seems to be the most memory intensive part. This is something that is
under investigation, and hopefully will be fixed in the future.
