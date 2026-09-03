"""
    module Defaults

Some default values and settings for MPSKit.
"""
module Defaults

import KrylovKit: GMRES, Arnoldi, Lanczos
using OhMyThreads
using ..MPSKit: DynamicTol, AdaptiveKrylov
using Preferences: @load_preference, @set_preferences!
using TensorKit: TensorKit
using TensorOperations: DefaultBackend
using MatrixAlgebraKit: DefaultAlgorithm, Householder

const VERBOSE_NONE = 0
const VERBOSE_WARN = 1
const VERBOSE_CONV = 2
const VERBOSE_ITER = 3
const VERBOSE_ALL = 4

const eltype = ComplexF64
const maxiter = 200
const tolgauge = 1.0e-13
const tol = 1.0e-10
const verbosity = VERBOSE_ITER
const dynamic_tols = true
const tol_min = 1.0e-14
const tol_max = 1.0e-4
const eigs_tolfactor = 1.0e-3
const gauge_tolfactor = 1.0e-6
const envs_tolfactor = 1.0e-4
const krylovdim = 30

_finalize(iter, state, opp, envs) = (state, envs)

const linearsolver = GMRES(; tol, maxiter)
const eigsolver = Arnoldi(; tol, maxiter, eager = true)

# Default algorithms
# ------------------

alg_svd() = DefaultAlgorithm()
alg_orth() = Householder(; positive = true)

"""
    backend()

The default backend for tensor contractions and index manipulations.

`TensorOperations.DefaultBackend` is a placeholder rather than a CPU backend: the actual
implementation is selected from the types of the tensors involved, so this is already the right
choice for GPU-backed states.
"""
backend() = DefaultBackend()

function alg_gauge(;
        tol = tolgauge, maxiter = maxiter, verbosity = VERBOSE_WARN,
        alg_orth = alg_orth(),
        dynamic_tols = dynamic_tols, tol_min = tol_min, tol_max = tol_max,
        tol_factor = gauge_tolfactor
    )
    alg = (; tol, maxiter, verbosity, alg_orth)
    return dynamic_tols ? DynamicTol(alg, tol_min, tol_max, tol_factor) : alg
end

function alg_eigsolve(;
        ishermitian = true, tol = tol, maxiter = maxiter, verbosity = 0,
        eager = true, krylovdim = krylovdim, dynamic_tols = dynamic_tols, tol_min = tol_min,
        tol_max = tol_max, tol_factor = eigs_tolfactor, adaptive = false, adaptive_kwargs...
    )
    adaptive && return AdaptiveKrylov(; ishermitian, adaptive_kwargs...)
    alg = ishermitian ? Lanczos(; tol, maxiter, eager, krylovdim, verbosity) :
        Arnoldi(; tol, maxiter, eager, krylovdim, verbosity)
    return dynamic_tols ? DynamicTol(alg, tol_min, tol_max, tol_factor) : alg
end

function alg_environments(;
        tol = tol, maxiter = maxiter, verbosity = 0, krylovdim = krylovdim, eager = true,
        dynamic_tols = dynamic_tols, tol_min = tol_min, tol_max = tol_max,
        tol_factor = envs_tolfactor
    )
    alg = DefaultAlgorithm(; tol, maxiter, verbosity, eager, krylovdim)
    return dynamic_tols ? DynamicTol(alg, tol_min, tol_max, tol_factor) : alg
end

function alg_expsolve(;
        tol = tol, maxiter = maxiter, verbosity = 0, ishermitian = true,
        krylovdim = krylovdim
    )
    return ishermitian ? Lanczos(; tol, maxiter, krylovdim, verbosity) :
        Arnoldi(; tol, maxiter, krylovdim, verbosity)
end

"""
   const scheduler

The current settings for multi-threading, typically used to parallelize over unitcells.
This value is best controlled using [`set_scheduler!`](@ref).
"""
const scheduler = Ref{Scheduler}()

"""
    set_scheduler!([scheduler]; kwargs...)

Set the `OhMyThreads` multi-threading scheduler parameters.

The function either accepts a `scheduler` as an `OhMyThreads.Scheduler` or as a symbol where the corresponding parameters are specified as keyword arguments.
For a detailed description of all schedulers and their keyword arguments consult the [`OhMyThreads` documentation](https://juliafolds2.github.io/OhMyThreads.jl/stable/refs/api/#Schedulers).
"""
function set_scheduler!(sc = OhMyThreads.Implementation.NotGiven(); kwargs...)
    if isempty(kwargs) && sc isa OhMyThreads.Implementation.NotGiven
        # default value: Serial if single-threaded, Dynamic otherwise
        scheduler[] = Threads.nthreads() == 1 ? SerialScheduler() : DynamicScheduler()
    else
        scheduler[] = OhMyThreads.Implementation._scheduler_from_userinput(sc; kwargs...)
    end
    return scheduler[]
end

"""
    const buffering

Whether local updates serve their scratch space from a dedicated allocator.

This is a compile-time preference, best controlled using [`set_buffering!`](@ref).
"""
const buffering = @load_preference("buffering", true)::Bool

"""
    set_buffering!(b::Bool)

Enable or disable dedicated scratch space for local updates.

When enabled - the default - the tensor contractions of a local update serve their intermediates
from an allocator that bypasses Julia's memory manager, which cuts both the allocation count and the
garbage-collection time substantially. Disabling it trades that back for lower memory use, which is
worthwhile when memory rather than time is the binding constraint.

This only affects storage types for which MPSKit has such an allocator, i.e. host memory; see
[`MPSKit.default_allocator`](@ref).

!!! note
    This setting is stored in a `LocalPreferences.toml` file next to the active `Project.toml` and is
    read when MPSKit is compiled, so Julia has to be restarted for a change to take effect.
"""
function set_buffering!(b::Bool)
    @set_preferences!("buffering" => b)
    @info "Buffering set to $b; restart Julia for this to take effect."
    return b
end

end
