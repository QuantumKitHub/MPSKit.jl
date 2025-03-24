"""
    module Defaults

Some default values and settings for MPSKit.
"""
module Defaults

import KrylovKit: GMRES, Arnoldi, Lanczos
using OhMyThreads
using ..MPSKit: DynamicTol
using TensorKit: TensorKit

const VERBOSE_NONE = 0
const VERBOSE_WARN = 1
const VERBOSE_CONV = 2
const VERBOSE_ITER = 3
const VERBOSE_ALL = 4

const eltype = ComplexF64
const maxiter = 200
const tolgauge = 1e-13
const tol = 1e-10
const verbosity = VERBOSE_ITER
const dynamic_tols = true
const tol_min = 1e-14
const tol_max = 1e-4
const eigs_tolfactor = 1e-3
const gauge_tolfactor = 1e-6
const envs_tolfactor = 1e-4
const krylovdim = 30

_finalize(iter, state, opp, envs) = (state, envs)

const linearsolver = GMRES(; tol, maxiter)
const eigsolver = Arnoldi(; tol, maxiter, eager=true)

# Default algorithms
# ------------------

function alg_gauge(; tol=tolgauge, maxiter=maxiter, verbosity=VERBOSE_WARN,
                   dynamic_tols=dynamic_tols, tol_min=tol_min, tol_max=tol_max,
                   tol_factor=gauge_tolfactor)
    alg = (; tol, maxiter, verbosity)
    return dynamic_tols ? DynamicTol(alg, tol_min, tol_max, tol_factor) : alg
end

function alg_eigsolve(; ishermitian=true, tol=tol, maxiter=maxiter, verbosity=0,
                      eager=true,
                      krylovdim=krylovdim,
                      dynamic_tols=dynamic_tols, tol_min=tol_min, tol_max=tol_max,
                      tol_factor=eigs_tolfactor)
    alg = ishermitian ? Lanczos(; tol, maxiter, eager, krylovdim, verbosity) :
          Arnoldi(; tol, maxiter, eager, krylovdim, verbosity)
    return dynamic_tols ? DynamicTol(alg, tol_min, tol_max, tol_factor) : alg
end

alg_svd() = TensorKit.SDD()

# TODO: make verbosity and maxiter actually do something
function alg_environments(; tol=tol, maxiter=maxiter, verbosity=0,
                          dynamic_tols=dynamic_tols, tol_min=tol_min, tol_max=tol_max,
                          tol_factor=envs_tolfactor)
    alg = (; tol, maxiter, verbosity)
    return dynamic_tols ? DynamicTol(alg, tol_min, tol_max, tol_factor) : alg
end

function alg_expsolve(; tol=tol, maxiter=maxiter, verbosity=0,
                      ishermitian=true, krylovdim=krylovdim)
    return ishermitian ? Lanczos(; tol, maxiter, krylovdim, verbosity) :
           Arnoldi(; tol, maxiter, krylovdim, verbosity)
end

"""
   const scheduler

A scoped value that controls the current settings for multi-threading, typically used to parallelize over unitcells.
This value is best controlled using [`set_scheduler!`](@ref).
"""
const scheduler = Ref{Scheduler}()

"""
    set_scheduler!([scheduler]; kwargs...)

Set the `OhMyThreads` multi-threading scheduler parameters.

The function either accepts a `scheduler` as an `OhMyThreads.Scheduler` or as a symbol where the corresponding parameters are specificed as keyword arguments.
For a detailed description of all schedulers and their keyword arguments consult the [`OhMyThreads` documentation](https://juliafolds2.github.io/OhMyThreads.jl/stable/refs/api/#Schedulers).
"""
function set_scheduler!(sc=OhMyThreads.Implementation.NotGiven(); kwargs...)
    if isempty(kwargs) && sc isa OhMyThreads.Implementation.NotGiven
        # default value: Serial if single-threaded, Dynamic otherwise
        scheduler[] = Threads.nthreads() == 1 ? SerialScheduler() : DynamicScheduler()
    else
        scheduler[] = OhMyThreads.Implementation._scheduler_from_userinput(sc; kwargs...)
    end
    return scheduler[]
end

end
