struct NoExpand <: Algorithm end

_update_post_expand(::NoExpand, iter, ϵ) = NoExpand()

function post_expand!(pos::Int, ::Val{:right}, ψ::AbstractFiniteMPS, H, ::NoExpand, envs, AC, alg_gauge; normalize::Bool = false)
    return left_gauge!(ψ, pos, AC, alg_gauge; normalize)
end

function post_expand!(pos::Int, ::Val{:left}, ψ::AbstractFiniteMPS, H, ::NoExpand, envs, AC, alg_gauge; normalize::Bool = false)
    return right_gauge!(ψ, pos, AC, alg_gauge; normalize)
end
