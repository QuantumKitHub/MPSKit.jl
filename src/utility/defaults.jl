"""
    module Defaults

Some default values and settings for MPSKit.
"""
module Defaults

using Preferences
import KrylovKit: GMRES, Arnoldi
using ..MPSKit: ThrottledTol

const eltype = ComplexF64
const maxiter = 100
const tolgauge = 1e-14
const tol = 1e-12
const verbose = true
const dynamical_tols = true
const tol_min = 1e-14
const tol_max = 1e-5
const eigs_tolfactor = 1e-5
const gauge_tolfactor = 1e-8
const envs_tolfactor = 1e-5

_finalize(iter, state, opp, envs) = (state, envs)

const linearsolver = GMRES(; tol, maxiter)
const eigsolver = Arnoldi(; tol, maxiter, eager=true)

# Default algorithms
# ------------------

function alg_gauge(; tol=tolgauge, maxiter=maxiter,
                   dynamical_tols=dynamical_tols, tol_min=tol_min, tol_max=tol_max,
                   tol_factor=gauge_tolfactor)
    alg = (; tol, maxiter)
    return dynamical_tols ? ThrottledTol(alg, tol, tol_max, tol_factor) : alg
end

function alg_eigsolve(; tol=tol, maxiter=maxiter, eager=true,
                      dynamical_tols=dynamical_tols, tol_min=tol_min, tol_max=tol_max,
                      tol_factor=eigs_tolfactor)
    alg = Arnoldi(; tol, maxiter, eager)
    return dynamical_tols ? ThrottledTol(alg, tol, tol_max, tol_factor) : alg
end

function alg_environments(; tol=tol, maxiter=maxiter,
                          dynamical_tols=dynamical_tols, tol_min=tol_min, tol_max=tol_max,
                          tol_factor=envs_tolfactor)
    alg = (; tol, maxiter)
    return dynamical_tols ? ThrottledTol(alg, tol, tol_max, tol_factor) : alg
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
