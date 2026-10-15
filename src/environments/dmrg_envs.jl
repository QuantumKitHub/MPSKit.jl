"""
    struct DMRGEnvironments <: AbstractMPSEnvironments

Environment manager for a finite DMRG solve with a fixed operator. It wraps the solve's
`FiniteEnvironments`, which keeps the environments up to date, and stores alongside each
environment its prepared half of the `nsites`-site effective Hamiltonian. A half is rebuilt
when it is queried and the environment it was prepared from has been replaced.
"""
struct DMRGEnvironments{E <: FiniteEnvironments, P, L, R, G, B, A} <: AbstractMPSEnvironments
    parent::E
    "operator-only data for each pair of adjacent sites, or `nothing` if unused"
    pairs::P
    "prepared left halves, `nothing` until first queried"
    lhalves::Vector{Union{Nothing, Missing, L}}
    "prepared right halves, `nothing` until first queried"
    rhalves::Vector{Union{Nothing, Missing, R}}
    "the environments the halves were prepared from"
    lsources::Vector{Union{Nothing, G}}
    rsources::Vector{Union{Nothing, G}}
    backend::B
    allocator::A
    "number of sites of the effective Hamiltonians whose halves are prepared"
    nsites::Int
end

Base.parent(envs::DMRGEnvironments) = envs.parent

leftenv(envs::DMRGEnvironments, i, ψ; kwargs...) =
    leftenv(parent(envs), i, ψ; envs.backend, envs.allocator)
rightenv(envs::DMRGEnvironments, i, ψ; kwargs...) =
    rightenv(parent(envs), i, ψ; envs.backend, envs.allocator)

"""
    lefthalf(envs, i, ψ)
    righthalf(envs, i, ψ)

The prepared half of the effective Hamiltonian at site `i` that depends on the left (right)
environment of `i` for the state `ψ`, rebuilt if that environment has changed.
"""
function lefthalf(envs::DMRGEnvironments, i, ψ)
    GL = leftenv(envs, i, ψ)
    if envs.lsources[i] !== GL
        O = parent(envs).operator
        envs.lhalves[i] = prepare_lefthalf(GL, O, envs.pairs, i, envs.nsites, envs.backend, envs.allocator)
        envs.lsources[i] = GL
    end
    return something(envs.lhalves[i])
end
function righthalf(envs::DMRGEnvironments, i, ψ)
    GR = rightenv(envs, i, ψ)
    if envs.rsources[i] !== GR
        O = parent(envs).operator
        envs.rhalves[i] = prepare_righthalf(GR, O, envs.pairs, i, envs.nsites, envs.backend, envs.allocator)
        envs.rsources[i] = GR
    end
    return something(envs.rhalves[i])
end

"""
    dmrg_environments(ψ, O, envs; backend, allocator, nsites)

Wrap `envs` into [`DMRGEnvironments`](@ref) whose halves are prepared for `nsites`-site
effective Hamiltonians (1 for DMRG, 2 for DMRG2). Unprepared operators, or those for the
other window size, are constructed from the environments.
Unsupported environments are returned unchanged.
"""
dmrg_environments(ψ, O, envs; kwargs...) = envs
function dmrg_environments(
        ψ::Union{FiniteMPS{<:MPSTensor}, WindowMPS{<:MPSTensor}}, O::AbstractMPO, envs::FiniteEnvironments;
        backend, allocator, nsites::Int,
    )
    @assert nsites in (1, 2) "effective Hamiltonians act on one or two sites"
    # Mixed bra/ket environments are not supported.
    isnothing(envs.above) || return envs
    N = length(ψ)
    GL, GR = envs.GLs[1], envs.GRs[end]
    pairs = nsites == 2 ? pair_data(O, GL) : nothing
    # The halves at the boundaries fix the element types.
    L = prepare_lefthalf(GL, O, pairs, 1, nsites, backend, allocator)
    R = prepare_righthalf(GR, O, pairs, N, nsites, backend, allocator)
    lhalves = fill!(Vector{Union{Nothing, Missing, typeof(L)}}(undef, N), nothing)
    rhalves = fill!(Vector{Union{Nothing, Missing, typeof(R)}}(undef, N), nothing)
    lsources = fill!(Vector{Union{Nothing, eltype(envs.GLs)}}(undef, N), nothing)
    rsources = fill!(Vector{Union{Nothing, eltype(envs.GRs)}}(undef, N), nothing)
    lhalves[1], lsources[1] = L, GL
    rhalves[N], rsources[N] = R, GR
    return DMRGEnvironments(envs, pairs, lhalves, rhalves, lsources, rsources, backend, allocator, nsites)
end

function dmrg_environments(ψ::WindowMPS{<:MPSTensor}, O::WindowMPOHamiltonian, envs::FiniteEnvironments; kwargs...)
    return dmrg_environments(ψ, O.finite_ham, envs; kwargs...)
end
function dmrg_environments(ψ, O::LazySum, envs::MultipleEnvironments; kwargs...)
    return MultipleEnvironments(map((op, env) -> dmrg_environments(ψ, op, env; kwargs...), O.ops, envs.envs))
end
dmrg_environments(ψ, O::MultipliedOperator, envs; kwargs...) = dmrg_environments(ψ, O.op, envs; kwargs...)
function dmrg_environments(ψ, O::LinearCombination, envs::LazyLincoCache; kwargs...)
    return LazyLincoCache(O, map((op, env) -> dmrg_environments(ψ, op, env; kwargs...), O.opps, envs.envs))
end

unwrap_environments(envs) = envs
unwrap_environments(envs::DMRGEnvironments) = parent(envs)
unwrap_environments(envs::MultipleEnvironments) = MultipleEnvironments(map(unwrap_environments, envs.envs))
unwrap_environments(envs::LazyLincoCache) = LazyLincoCache(envs.operator, map(unwrap_environments, envs.envs))
