"""
    struct DMRGEnvironments <: AbstractMPSEnvironments

Environment manager for a finite DMRG solve with a fixed operator. It wraps the solve's
`FiniteEnvironments`, whose `GLs` and `GRs` it updates explicitly, and stores alongside each
environment its prepared half of the `nsites`-site effective Hamiltonian. Environments are
only required to be current outside the active window, and are advanced by
[`center_window!`](@ref) rather than by checking dependencies on every query.
"""
struct DMRGEnvironments{E <: FiniteEnvironments, P, L, R, D, B, A} <: AbstractMPSEnvironments
    "the solve's environments; read-only callbacks may refresh them"
    parent::E
    "operator-only data for each pair of adjacent sites, or `nothing` if unused"
    pairs::P
    "prepared left halves, `nothing` until the sweep first reaches the site"
    lhalves::Vector{Union{Nothing, Missing, L}}
    "prepared right halves, `nothing` until the sweep first reaches the site"
    rhalves::Vector{Union{Nothing, Missing, R}}
    "`AL[i]` last transferred into the left half at `i + 1`"
    ldependencies::Vector{D}
    "`AR[i]` last transferred into the right half at `i - 1`"
    rdependencies::Vector{D}
    backend::B
    allocator::A
    "number of sites of the effective Hamiltonians whose halves are prepared"
    nsites::Int
end

Base.parent(envs::DMRGEnvironments) = envs.parent

function leftenv(envs::DMRGEnvironments, i, ψ; kwargs...)
    isnothing(envs.lhalves[i]) && throw(ArgumentError("left environment $i has not been reached"))
    return envs.parent.GLs[i]
end
function rightenv(envs::DMRGEnvironments, i, ψ; kwargs...)
    isnothing(envs.rhalves[i]) && throw(ArgumentError("right environment $i has not been reached"))
    return envs.parent.GRs[i + 1]
end
lefthalf(envs::DMRGEnvironments, i) = something(envs.lhalves[i])
righthalf(envs::DMRGEnvironments, i) = something(envs.rhalves[i])

"""
    dmrg_environments(ψ, O, envs; backend, allocator, nsites)

Wrap `envs` into [`DMRGEnvironments`](@ref) whose halves are prepared for `nsites`-site
effective Hamiltonians (1 for DMRG, 2 for DMRG2), transferring the right environments from
the right boundary to set up the first local window. Unprepared operators, or those for the
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
    L = prepare_lefthalf(GL, O, pairs, 1, nsites, backend, allocator)
    R = prepare_righthalf(GR, O, pairs, N, nsites, backend, allocator)
    lhalves = fill!(Vector{Union{Nothing, Missing, typeof(L)}}(undef, N), nothing)
    rhalves = fill!(Vector{Union{Nothing, Missing, typeof(R)}}(undef, N), nothing)
    lhalves[1], rhalves[N] = L, R
    dmrg_envs = DMRGEnvironments(
        envs, pairs, lhalves, rhalves, copy(envs.ldependencies), copy(envs.rdependencies),
        backend, allocator, nsites,
    )
    for site in (N - 1):-1:1
        transfer_rightenv!(dmrg_envs, ψ, O, ψ, site)
    end
    return dmrg_envs
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

uses_dmrg_environments(envs) = false
uses_dmrg_environments(::DMRGEnvironments) = true
uses_dmrg_environments(envs::Union{MultipleEnvironments, LazyLincoCache}) = any(uses_dmrg_environments, envs.envs)

function transfer_leftenv!(
        envs::DMRGEnvironments, below, operator, above, site::Int;
        backend::AbstractBackend = envs.backend, allocator = envs.allocator
    )
    AL = below.AL[site - 1]
    GL = leftenv(envs, site - 1, below) * TransferMatrix(AL, operator[site - 1], AL; backend, allocator)
    envs.parent.GLs[site] = GL
    envs.lhalves[site] = prepare_lefthalf(GL, operator, envs.pairs, site, envs.nsites, backend, allocator)
    envs.ldependencies[site - 1] = envs.parent.ldependencies[site - 1] = AL
    return envs
end
function transfer_rightenv!(
        envs::DMRGEnvironments, below, operator, above, site::Int;
        backend::AbstractBackend = envs.backend, allocator = envs.allocator
    )
    AR = below.AR[site + 1]
    GR = TransferMatrix(AR, operator[site + 1], AR; backend, allocator) * rightenv(envs, site + 1, below)
    envs.parent.GRs[site + 1] = GR
    envs.rhalves[site] = prepare_righthalf(GR, operator, envs.pairs, site, envs.nsites, backend, allocator)
    envs.rdependencies[site + 1] = envs.parent.rdependencies[site + 1] = AR
    return envs
end

"""
    center_window!(envs, ψ, first, last)

Bring the environments around the window `first:last` up to date: transfer `AL[first - 1]`
and `AR[last + 1]` if they are not the tensors those environments were last built from.
"""
center_window!(envs, ψ, first, last) = envs
function center_window!(envs::DMRGEnvironments, ψ, first, last)
    O = envs.parent.operator
    if first > 1 && (isnothing(envs.lhalves[first]) || envs.ldependencies[first - 1] !== ψ.AL[first - 1])
        transfer_leftenv!(envs, ψ, O, ψ, first)
    end
    if last < length(ψ) && (isnothing(envs.rhalves[last]) || envs.rdependencies[last + 1] !== ψ.AR[last + 1])
        transfer_rightenv!(envs, ψ, O, ψ, last)
    end
    return envs
end
function center_window!(envs::Union{MultipleEnvironments, LazyLincoCache}, ψ, first, last)
    foreach(env -> center_window!(env, ψ, first, last), envs.envs)
    return envs
end

"""
    dmrg_environments_current(envs, ψ, finalized_envs)

Whether `envs` still describe `ψ` after a finalizer returned `finalized_envs`, assuming a
completed backward sweep. Environments that the next window refreshes itself are not checked.
"""
dmrg_environments_current(envs, ψ, finalized_envs) = envs === finalized_envs
function dmrg_environments_current(envs::DMRGEnvironments, ψ, finalized_envs)
    parent(envs) === finalized_envs || return false
    return all(i -> envs.rdependencies[i] === ψ.AR[i], 3:length(ψ))
end
function dmrg_environments_current(envs::Union{MultipleEnvironments, LazyLincoCache}, ψ, finalized_envs)
    return all(splat((env, fin) -> dmrg_environments_current(env, ψ, fin)), zip(envs.envs, finalized_envs.envs))
end
