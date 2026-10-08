"""
    struct FiniteEnvironments <: AbstractMPSEnvironments

Environment manager for `FiniteMPS` and `WindowMPS`. This structure is responsible for automatically checking
if the queried environment is still correctly cached and if not recalculates.
"""
struct FiniteEnvironments{A, B, C, D} <: AbstractMPSEnvironments
    above::A

    operator::B #the operator

    ldependencies::Vector{C} #the data we used to calculate leftenvs/rightenvs
    rdependencies::Vector{C}

    GLs::Vector{D}
    GRs::Vector{D}
end

function initialize_environments(below::AbstractFiniteMPS, operator, above, leftstart, rightstart)
    above = above === below ? nothing : above
    N = length(below)
    leftenvs = [i == 0 ? leftstart : similar(leftstart) for i in 0:N]
    rightenvs = [i == N ? rightstart : similar(rightstart) for i in 0:N]

    t = similar(below.AL[1])
    return FiniteEnvironments(
        above, operator, fill(t, N), fill(t, N), leftenvs, rightenvs
    )
end

function environments(
        below::FiniteMPS{S}, operator::Union{FiniteMPO, FiniteMPOHamiltonian}, above;
        leftstart = nothing, rightstart = nothing
    ) where {S}
    N = length(below)
    if isnothing(leftstart)
        Vl_bot = left_virtualspace(below, 1)
        Vl_mid = left_virtualspace(operator, 1)
        Vl_top = isnothing(above) ? left_virtualspace(below, 1) : left_virtualspace(above, 1)
        leftstart = isomorphism(storagetype(S), Vl_bot ⊗ Vl_mid' ← Vl_top)
    end
    if isnothing(rightstart)
        Vr_bot = right_virtualspace(below, N)
        Vr_mid = right_virtualspace(operator, N)
        Vr_top = isnothing(above) ? right_virtualspace(below, N) : right_virtualspace(above, N)
        rightstart = isomorphism(storagetype(S), Vr_top ⊗ Vr_mid ← Vr_bot)
    end
    return initialize_environments(below, operator, above, leftstart, rightstart)
end
function environments(
        below::WindowMPS, operator::Union{InfiniteMPOHamiltonian, InfiniteMPO}, above;
        lenvs = environments(below.left_gs, operator, below.left_gs),
        renvs = environments(below.right_gs, operator, below.right_gs),
        leftstart = copy(lenvs.GLs[1]),
        rightstart = copy(renvs.GRs[end])
    )
    return initialize_environments(below, operator, above, leftstart, rightstart)
end
function environments(
        below::WindowMPS, O::WindowMPOHamiltonian, above = nothing;
        # NOTE: the defaults are deliberately inlined rather than shared via `lenvs`/`renvs`
        # kwargs: when explicit boundaries are passed (e.g. the squared environments in
        # `squaredenvs`), we must not trigger the infinite fixed-point solve, which need not
        # converge for e.g. `conj(H) * H`.
        leftstart = copy(environments(below.left_gs, O.left_ham, below.left_gs).GLs[1]),
        rightstart = copy(environments(below.right_gs, O.right_ham, below.right_gs).GRs[end])
    )
    return initialize_environments(below, O.finite_ham, above, leftstart, rightstart)
end

environments(below::S, above::S) where {S <: Union{FiniteMPS, WindowMPS}} =
    environments(below, nothing, above)
function environments(below::S, ::Nothing, above::S) where {S <: Union{FiniteMPS, WindowMPS}}
    S isa WindowMPS &&
        (above.left_gs == below.left_gs || throw(ArgumentError("left gs differs")))
    S isa WindowMPS &&
        (above.right_gs == below.right_gs || throw(ArgumentError("right gs differs")))

    ops = fill(nothing, length(below))
    return initialize_environments(below, ops, above, l_LL(above), r_RR(above))
end

function environments(
        state::Union{FiniteMPS, WindowMPS}, operator::ProjectionOperator, above = nothing
    )
    @plansor leftstart[-1; -2 -3 -4] := l_LL(operator.ket)[-3; -4] * l_LL(operator.ket)[-1; -2]
    @plansor rightstart[-1; -2 -3 -4] := r_RR(operator.ket)[-1; -2] * r_RR(operator.ket)[-3; -4]
    return initialize_environments(
        state, fill(nothing, length(state)), operator.ket, leftstart, rightstart
    )
end

#notify the cache that we updated in-place, so it should invalidate the dependencies
function poison!(ca::FiniteEnvironments, ind)
    ca.ldependencies[ind] = similar(ca.ldependencies[ind])
    return ca.rdependencies[ind] = similar(ca.rdependencies[ind])
end

#rightenv[ind] will be contracteable with the tensor on site [ind]
"""
    rightenv(envs, site, state)

Return the right environment stored in `envs` at `site` for the given `state`: the contraction
of everything to the right of `site` in the network the environments were built for.
The result is gauge-compatible with the tensor of `state` at `site` and can be contracted onto
it directly.

See also [`leftenv`](@ref) and [`environments`](@ref).
"""
function rightenv(
        ca::FiniteEnvironments, ind, state;
        backend::AbstractBackend = DefaultBackend(), allocator = DefaultAllocator()
    )
    a = findfirst(i -> !(state.AR[i] === ca.rdependencies[i]), length(state):-1:(ind + 1))
    a = isnothing(a) ? nothing : length(state) - a + 1

    if !isnothing(a)
        #we need to recalculate
        for j in a:-1:(ind + 1)
            above = isnothing(ca.above) ? state.AR[j] : ca.above.AR[j]
            ca.GRs[j] = TransferMatrix(
                above, ca.operator[j], state.AR[j]; backend, allocator
            ) * ca.GRs[j + 1]
            ca.rdependencies[j] = state.AR[j]
        end
    end

    return ca.GRs[ind + 1]
end

"""
    leftenv(envs, site, state)

Return the left environment stored in `envs` at `site` for the given `state`: the contraction
of everything to the left of `site` in the network the environments were built for.
The result is gauge-compatible with the tensor of `state` at `site` and can be contracted onto
it directly.

See also [`rightenv`](@ref) and [`environments`](@ref).
"""
function leftenv(
        ca::FiniteEnvironments, ind, state;
        backend::AbstractBackend = DefaultBackend(), allocator = DefaultAllocator()
    )
    a = findfirst(i -> !(state.AL[i] === ca.ldependencies[i]), 1:(ind - 1))

    if !isnothing(a)
        #we need to recalculate
        for j in a:(ind - 1)
            above = isnothing(ca.above) ? state.AL[j] : ca.above.AL[j]
            ca.GLs[j + 1] = ca.GLs[j] * TransferMatrix(
                above, ca.operator[j], state.AL[j]; backend, allocator
            )
            ca.ldependencies[j] = state.AL[j]
        end
    end

    return ca.GLs[ind]
end

# Explicit environment snapshots used during finite DMRG sweeps.
"""
    DMRGSweepCache

Solve-owned environments for finite DMRG. Records are snapshots for a particular sweep
position: only the environments outside the active window are required to be current.
Center moves explicitly replace records; querying a record never scans MPS dependencies.
The operator is fixed and finalizers must not modify the state or cache.
"""
struct DMRGSweepCache{E, O, L, R, B, A} <: AbstractMPSEnvironments
    "ordinary manager kept compatible with the final state and read-only callbacks"
    environments::E
    "fixed operator or operator-only metadata shared by all records"
    operator_data::O
    "left snapshots before each site; `nothing` until first reached by the forward sweep"
    left::Vector{Union{Nothing, L}}
    "right snapshots after each site, initialized by a backward transfer pass"
    right::Vector{Union{Nothing, R}}
    "backend used to construct persistent contractions"
    backend::B
    "allocator for temporary work; persistent records own their tensor storage"
    allocator::A
    "whether local updates require prepared one-site contributions"
    one_site::Bool
    "whether local updates require prepared two-site contributions"
    two_site::Bool
end

"""
    DMRGEnvironmentRecord

Directional environment snapshot and the associated effective-operator contributions.
A left record at `i` represents sites before `i`; a right record represents sites after
`i`. Replacing a record publishes new tensors without mutating earlier snapshots.
Transfers only read `environment`.
"""
struct DMRGEnvironmentRecord{G, P, Q}
    "ordinary `GL` or `GR` tensor"
    environment::G
    "directional one-site contribution; `missing` when only pairs are prepared"
    one_site::P
    "directional two-site contribution; `missing` if unused"
    two_site::Union{Missing, Q}

    function DMRGEnvironmentRecord{G, P, Q}(environment, one_site, two_site) where {G, P, Q}
        return new{G, P, Q}(environment, one_site, two_site)
    end
end
DMRGEnvironmentRecord(env, one_site, two_site) =
    DMRGEnvironmentRecord{typeof(env), typeof(one_site), typeof(two_site)}(env, one_site, two_site)

function leftenv(cache::DMRGSweepCache{E, O, L}, i, ψ; kwargs...) where {E, O, L}
    record = cache.left[i]
    isnothing(record) && error("left DMRG environment at site $i has not been advanced")
    return (record::L).environment
end
function rightenv(cache::DMRGSweepCache{E, O, L, R}, i, ψ; kwargs...) where {E, O, L, R}
    record = cache.right[i]
    isnothing(record) && error("right DMRG environment at site $i has not been advanced")
    return (record::R).environment
end
