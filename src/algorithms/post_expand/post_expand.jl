_update_alg_gauge(alg, iter, ϵ) = alg

set_alg_gauge(::Nothing, inner_gauge) = inner_gauge
alg_gauge(alg) = alg

# A plain (non-expanding) gauge step does not touch `H`/`envs`, and thus has no contraction to
# route a `backend`/`allocator` to: accept and drop them, so that callers can pass them
# unconditionally and only the expanding gauges (e.g. [`DMRG3S`](@ref)) pick them up.
gauge!(
    ψ::AbstractFiniteMPS, pos::Int, direction, H, envs, AC, alg_gauge;
    normalize::Bool = false,
    backend::AbstractBackend = DefaultBackend(), allocator = DefaultAllocator()
) = gauge!(ψ, pos, direction, AC, alg_gauge; normalize)
gauge2!(ψ::AbstractFiniteMPS, pos::Int, direction, H, envs, AC2, alg_gauge; normalize::Bool = false) =
    gauge2!(ψ, pos, direction, AC2, alg_gauge; normalize)
