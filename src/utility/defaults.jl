"""
    module Defaults

Some default values and settings for MPSKit.
"""
module Defaults

using Preferences
import KrylovKit: GMRES, Arnoldi

const eltype    = ComplexF64
const maxiter   = 100
const tolgauge  = 1e-14
const tol       = 1e-12
const verbose   = true

_finalize(iter, state, opp, envs) = (state, envs)

const linearsolver = GMRES(; tol, maxiter)
const eigsolver = Arnoldi(; tol, maxiter)

# Preferences
# -----------

function set_parallelization(options::Pair{String, Bool}...)
    for (key, val) in options
        if !(key in ("sites", "derivatives", "transfers"))
            throw(ArgumentError("Invalid option: \"$(key)\""))
        end
        
        @set_preferences!("parallelize_$key" => val)
    end
    
    sites = @load_preference("parallelize_sites", nothing)
    derivatives = @load_preference("parallelize_derivatives", nothing)
    transfers = @load_preference("parallelize_derivatives", nothing)
    @info "Parallelization changed; restart your Julia session for this change to take effect!" sites derivatives transfers
    return nothing
end

const parallelize_sites = @load_preference("parallelize_sites", Threads.nthreads() > 1)
const parallelize_derivatives = @load_preference("parallelize_derivatives", Threads.nthreads() > 1)
const parallelize_transfers = @load_preference("parallelize_transfers", Threads.nthreads() > 1)

end
