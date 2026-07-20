struct NoExpand{A} <: Algorithm
    alg_gauge::A
end

_update_alg_gauge(alg::NoExpand, iter, ϵ) = alg

gauge!(pos::Int, direction, ψ::AbstractFiniteMPS, H, envs, AC, alg::NoExpand; normalize::Bool = false) = 
    gauge!(ψ, pos, direction, AC, alg.alg_gauge; normalize)
gauge2!(pos::Int, direction, ψ::AbstractFiniteMPS, H, envs, AC2, alg::NoExpand; normalize::Bool = false) = 
    gauge2!(ψ, pos, direction, AC2, alg.alg_gauge; normalize)