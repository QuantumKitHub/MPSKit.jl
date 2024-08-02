"""
    module Defaults

Some default values and settings for MPSKit.
"""
module Defaults

using Preferences
import KrylovKit: GMRES, Arnoldi, Lanczos
using ..MPSKit: DynamicTol

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
    return dynamic_tols ? DynamicTol(alg, tol, tol_max, tol_factor) : alg
end

function alg_eigsolve(; ishermitian=true, tol=tol, maxiter=maxiter, verbosity=0,
                      eager=true,
                      krylovdim=krylovdim,
                      dynamic_tols=dynamic_tols, tol_min=tol_min, tol_max=tol_max,
                      tol_factor=eigs_tolfactor)
    alg = ishermitian ? Lanczos(; tol, maxiter, eager, krylovdim, verbosity) :
          Arnoldi(; tol, maxiter, eager, krylovdim, verbosity)
    return dynamic_tols ? DynamicTol(alg, tol, tol_max, tol_factor) : alg
end

# TODO: make verbosity and maxiter actually do something
function alg_environments(; tol=tol, maxiter=maxiter, verbosity=0,
                          dynamic_tols=dynamic_tols, tol_min=tol_min, tol_max=tol_max,
                          tol_factor=envs_tolfactor)
    alg = (; tol, maxiter, verbosity)
    return dynamic_tols ? DynamicTol(alg, tol, tol_max, tol_factor) : alg
end
function alg_expsolve(; tol=tol, maxiter=maxiter, verbosity=0,
                      ishermitian=true, krylovdim=krylovdim)
    return ishermitian ? Lanczos(; tol, maxiter, krylovdim, verbosity) :
           Arnoldi(; tol, maxiter, krylovdim, verbosity)
end

# Preferences
# -----------

function set_parallelization(options::Pair{String,Bool}...)
    for (key, val) in options
        if !(key in ("sites", "derivatives", "transfers"))
            throw(ArgumentError("Invalid option: \"$(key)\""))
        end

        @set_preferences!("parallelize_$key" => val)
    end

    sites = @load_preference("parallelize_sites", nothing)
    derivatives = @load_preference("parallelize_derivatives", nothing)
    transfers = @load_preference("parallelize_transfers", nothing)
    @info "Parallelization changed; restart your Julia session for this change to take effect!" sites derivatives transfers
    return nothing
end

const parallelize_sites = @load_preference("parallelize_sites", Threads.nthreads() > 1)
const parallelize_derivatives = @load_preference("parallelize_derivatives",
                                                 Threads.nthreads() > 1)
const parallelize_transfers = @load_preference("parallelize_transfers",
                                               Threads.nthreads() > 1)

end
