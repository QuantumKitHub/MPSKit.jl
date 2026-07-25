_update_alg_gauge(alg, iter, ϵ) = alg

set_alg_gauge(::Nothing, inner_gauge) = inner_gauge
alg_gauge(alg) = alg

gauge!(ψ::AbstractFiniteMPS, pos::Int, direction, H, envs, AC, alg_gauge; normalize::Bool = false) =
    gauge!(ψ, pos, direction, AC, alg_gauge; normalize)
gauge2!(ψ::AbstractFiniteMPS, pos::Int, direction, H, envs, AC2, alg_gauge; normalize::Bool = false) =
    gauge2!(ψ, pos, direction, AC2, alg_gauge; normalize)
