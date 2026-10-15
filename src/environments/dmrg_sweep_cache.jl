"""
    DMRGSweepCache

Solve-owned environments for finite DMRG. Records are snapshots for a particular sweep
position: only the environments outside the active window are required to be current.
Center moves explicitly replace records; querying a record never scans MPS dependencies.
The operator is fixed; a finalizer that modifies the state or environments forces a rebuild.
"""
struct DMRGSweepCache{E, P, L, R, D, B, A} <: AbstractMPSEnvironments
    "ordinary manager kept compatible with the final state and read-only callbacks"
    environments::E
    "operator-only pair data shared by all records, or `nothing` if unused"
    pairs::P
    "left snapshots before each site; `nothing` until first reached by the forward sweep"
    left::Vector{Union{Nothing, L}}
    "right snapshots after each site, initialized by a backward transfer pass"
    right::Vector{Union{Nothing, R}}
    "`AL[i]` last absorbed into the left record at `i + 1`; the ordinary manager's own
    dependencies can be refreshed by read-only callbacks without updating these records"
    ldependencies::Vector{D}
    "`AR[i]` last absorbed into the right record at `i - 1`"
    rdependencies::Vector{D}
    "backend used to construct persistent contractions"
    backend::B
    "allocator for temporary work; persistent records own their tensor storage"
    allocator::A
    "number of sites of the effective Hamiltonians whose contributions are prepared"
    nsites::Int
end

"""
    DMRGEnvironmentRecord

Directional environment snapshot and its contribution to the effective Hamiltonian.
A left record at `i` represents sites before `i`; a right record represents sites after
`i`. Replacing a record publishes new tensors without mutating earlier snapshots.
Transfers only read `environment`.
"""
struct DMRGEnvironmentRecord{G, P}
    "ordinary `GL` or `GR` tensor"
    environment::G
    "directional contribution; `missing` at a boundary without a two-site window"
    contribution::Union{Missing, P}

    function DMRGEnvironmentRecord{G, P}(environment, contribution) where {G, P}
        return new{G, P}(environment, contribution)
    end
end
DMRGEnvironmentRecord(env, contribution) =
    DMRGEnvironmentRecord{typeof(env), typeof(contribution)}(env, contribution)

leftenv(cache::DMRGSweepCache, i, ψ; kwargs...) = something(cache.left[i]).environment
rightenv(cache::DMRGSweepCache, i, ψ; kwargs...) = something(cache.right[i]).environment
left_contribution(cache::DMRGSweepCache, i) = something(cache.left[i]).contribution
right_contribution(cache::DMRGSweepCache, i) = something(cache.right[i]).contribution

"""
    initialize_sweep_cache(ψ, O, envs; backend, allocator, nsites)

Build solve-owned snapshots for a fixed operator: prepare the left boundary and absorb
right-canonical tensors from the right boundary to initialize the first local window.
Records carry prepared contributions to `nsites`-site effective Hamiltonians (1 for DMRG,
2 for DMRG2). Queries for unprepared operators, or for the other window size, construct
them from the snapshot environments.
Unsupported environment representations retain their existing update behavior.
"""
initialize_sweep_cache(ψ, O, envs; kwargs...) = envs
function initialize_sweep_cache(
        ψ::Union{FiniteMPS{<:MPSTensor}, WindowMPS{<:MPSTensor}}, O::AbstractMPO, envs::FiniteEnvironments;
        backend, allocator, nsites::Int,
    )
    @assert nsites in (1, 2) "effective Hamiltonians act on one or two sites"
    # Mixed bra/ket environments are outside the DMRG cache contract.
    isnothing(envs.above) || return envs
    pairs = nsites == 2 ? cache_pair_data(O, envs.GLs[1]) : nothing
    N = length(ψ)
    GL, GR = envs.GLs[1], envs.GRs[end]
    L = DMRGEnvironmentRecord(GL, prepare_left_environment(GL, O, pairs, 1, nsites, backend, allocator))
    R = DMRGEnvironmentRecord(GR, prepare_right_environment(GR, O, pairs, N, nsites, backend, allocator))
    cache = DMRGSweepCache(
        envs, pairs, fill!(Vector{Union{Nothing, typeof(L)}}(undef, N), nothing),
        fill!(Vector{Union{Nothing, typeof(R)}}(undef, N), nothing),
        copy(envs.ldependencies), copy(envs.rdependencies), backend, allocator, nsites,
    )
    cache.left[1] = L
    cache.right[N] = R
    for i in N:-1:2
        absorb_site!(cache, ψ, i, Val(:left))
    end
    return cache
end

function initialize_sweep_cache(ψ::WindowMPS{<:MPSTensor}, O::WindowMPOHamiltonian, envs::FiniteEnvironments; kwargs...)
    return initialize_sweep_cache(ψ, O.finite_ham, envs; kwargs...)
end
function initialize_sweep_cache(ψ, O::LazySum, envs::MultipleEnvironments; kwargs...)
    return MultipleEnvironments(
        map(O.ops, envs.envs) do op, env
            initialize_sweep_cache(ψ, op, env; kwargs...)
        end
    )
end
initialize_sweep_cache(ψ, O::MultipliedOperator, envs; kwargs...) =
    initialize_sweep_cache(ψ, O.op, envs; kwargs...)
function initialize_sweep_cache(ψ, O::LinearCombination, envs::LazyLincoCache; kwargs...)
    return LazyLincoCache(
        O, map(O.opps, envs.envs) do op, env
            initialize_sweep_cache(ψ, op, env; kwargs...)
        end
    )
end

unwrap_environments(envs) = envs
unwrap_environments(cache::DMRGSweepCache) = cache.environments
unwrap_environments(envs::MultipleEnvironments) = MultipleEnvironments(map(unwrap_environments, envs.envs))
unwrap_environments(envs::LazyLincoCache) = LazyLincoCache(envs.operator, map(unwrap_environments, envs.envs))

"""
    absorb_site!(envs, ψ, i, direction)

Contract the finalized tensor at `i` into an environment and publish its prepared record.
`Val(:right)` absorbs `AL[i]` into `GL`, producing the left record at `i + 1`;
`Val(:left)` absorbs `AR[i]` into `GR`, producing the right record at `i - 1`.
The underlying ordinary manager's tensor and dependency are updated at the same time.
"""
absorb_site!(envs, ψ, i, direction) = envs
function absorb_site!(cache::DMRGSweepCache{E, P, L}, ψ, i, ::Val{:right}) where {E, P, L}
    i < length(ψ) || return cache
    (; backend, allocator, pairs, nsites) = cache
    O, AL = cache.environments.operator, ψ.AL[i]
    GL = leftenv(cache, i, ψ) * TransferMatrix(AL, O[i], AL; backend, allocator)
    cache.left[i + 1] = L(GL, prepare_left_environment(GL, O, pairs, i + 1, nsites, backend, allocator))
    cache.environments.GLs[i + 1] = GL
    cache.ldependencies[i] = cache.environments.ldependencies[i] = AL
    return cache
end
function absorb_site!(cache::DMRGSweepCache{E, P, L, R}, ψ, i, ::Val{:left}) where {E, P, L, R}
    i > 1 || return cache
    (; backend, allocator, pairs, nsites) = cache
    O, AR = cache.environments.operator, ψ.AR[i]
    GR = TransferMatrix(AR, O[i], AR; backend, allocator) * rightenv(cache, i, ψ)
    cache.right[i - 1] = R(GR, prepare_right_environment(GR, O, pairs, i - 1, nsites, backend, allocator))
    cache.environments.GRs[i] = GR
    cache.rdependencies[i] = cache.environments.rdependencies[i] = AR
    return cache
end
function absorb_site!(envs::Union{MultipleEnvironments, LazyLincoCache}, ψ, i, direction)
    foreach(env -> absorb_site!(env, ψ, i, direction), envs.envs)
    return envs
end

"""
    center_window!(envs, ψ, first, last)

Bring the records around the window `first:last` up to date: absorb `AL[first - 1]` and
`AR[last + 1]` if they are not the tensors those records were last built from.
"""
center_window!(envs, ψ, first, last) = envs
function center_window!(cache::DMRGSweepCache, ψ, first, last)
    (; ldependencies, rdependencies) = cache
    if first > 1 && (isnothing(cache.left[first]) || ldependencies[first - 1] !== ψ.AL[first - 1])
        absorb_site!(cache, ψ, first - 1, Val(:right))
    end
    if last < length(ψ) && (isnothing(cache.right[last]) || rdependencies[last + 1] !== ψ.AR[last + 1])
        absorb_site!(cache, ψ, last + 1, Val(:left))
    end
    return cache
end
function center_window!(envs::Union{MultipleEnvironments, LazyLincoCache}, ψ, first, last)
    foreach(env -> center_window!(env, ψ, first, last), envs.envs)
    return envs
end

"""
    sweep_cache_current(envs, ψ, finalized_envs)

Whether the sweep environments `envs` still describe `ψ` after a finalizer returned
`finalized_envs`, assuming a completed backward sweep. Records that the next window
refreshes itself are not checked.
"""
sweep_cache_current(envs, ψ, finalized_envs) = envs === finalized_envs
function sweep_cache_current(cache::DMRGSweepCache, ψ, finalized_envs)
    cache.environments === finalized_envs || return false
    return all(i -> cache.rdependencies[i] === ψ.AR[i], 3:length(ψ))
end
function sweep_cache_current(envs::Union{MultipleEnvironments, LazyLincoCache}, ψ, finalized_envs)
    return all(splat((env, fin) -> sweep_cache_current(env, ψ, fin)), zip(envs.envs, finalized_envs.envs))
end

uses_sweep_cache(envs) = false
uses_sweep_cache(::DMRGSweepCache) = true
uses_sweep_cache(envs::Union{MultipleEnvironments, LazyLincoCache}) = any(uses_sweep_cache, envs.envs)
