"""
    InfiniteEnvironments <: AbstractMPSEnvironments

Environments for an infinite MPS-MPO-MPS combination. These solve the corresponding fixedpoint equations:
```math
GLs[i] * T_LL[i] = λ GLs[i + 1]
T_RR[i] * GRs[i] = λ GRs[i - 1]
```
where `T_LL` and `T_RR` are the (regularized) transfer matrix operators on a give site for `AL-O-AL` and `AR-O-AR` respectively.
"""
struct InfiniteEnvironments{V <: GenericMPSTensor} <: AbstractMPSEnvironments
    GLs::PeriodicVector{V}
    GRs::PeriodicVector{V}
end

Base.length(envs::InfiniteEnvironments) = length(envs.GLs)

leftenv(envs::InfiniteEnvironments, site::Int, state) = envs.GLs[site]
rightenv(envs::InfiniteEnvironments, site::Int, state) = envs.GRs[site]

function environments(
        below::InfiniteMPS, operator::Union{InfiniteMPO, InfiniteMPOHamiltonian},
        above::InfiniteMPS = below; kwargs...
    )
    GLs, GRs = initialize_environments(below, operator, above)
    envs = InfiniteEnvironments(GLs, GRs)
    return recalculate!(envs, below, operator, above; kwargs...)
end

function issamespace(
        envs::InfiniteEnvironments, below::InfiniteMPS,
        operator::Union{InfiniteMPO, InfiniteMPOHamiltonian}, above::InfiniteMPS
    )
    L = check_length(below, operator, above)
    for i in 1:L
        space(envs.GLs[i]) ==
            (
            left_virtualspace(below, i) ⊗ left_virtualspace(operator, i)' ←
                left_virtualspace(above, i)
        ) || return false
        space(envs.GRs[i]) ==
            (
            right_virtualspace(above, i) ⊗ right_virtualspace(operator, i) ←
                right_virtualspace(below, i)
        ) || return false
    end
    return true
end

function recalculate!(
        envs::InfiniteEnvironments, below::InfiniteMPS,
        operator::Union{InfiniteMPO, InfiniteMPOHamiltonian},
        above::InfiniteMPS = below;
        kwargs...
    )
    if !issamespace(envs, below, operator, above)
        # TODO: in-place initialization?
        GLs, GRs = initialize_environments(below, operator, above)
        copy!(envs.GLs, GLs)
        copy!(envs.GRs, GRs)
    end

    alg = environment_alg(below, operator, above; kwargs...)

    @sync begin
        @spawn compute_leftenvs!(envs, below, operator, above, alg)
        @spawn compute_rightenvs!(envs, below, operator, above, alg)
    end
    normalize!(envs, below, operator, above)

    return envs
end

# InfiniteMPO environments
# ------------------------
function initialize_environments(
        below::InfiniteMPS, operator::InfiniteMPO, above::InfiniteMPS = below
    )
    L = check_length(below, operator, above)
    GLs = PeriodicVector([randomize!(allocate_GL(below, operator, above, i)) for i in 1:L])
    GRs = PeriodicVector([randomize!(allocate_GR(below, operator, above, i)) for i in 1:L])
    return GLs, GRs
end

function compute_leftenvs!(
        envs::InfiniteEnvironments, below::InfiniteMPS,
        operator::InfiniteMPO, above::InfiniteMPS, alg
    )
    # compute eigenvector
    T = TransferMatrix(above.AL, operator, below.AL)
    λ, envs.GLs[1] = fixedpoint(flip(T), envs.GLs[1], :LM, alg)
    # push through unitcell
    for i in 2:length(operator)
        envs.GLs[i] = envs.GLs[i - 1] *
            TransferMatrix(above.AL[i - 1], operator[i - 1], below.AL[i - 1])
    end
    return λ, envs
end

function compute_rightenvs!(
        envs::InfiniteEnvironments, below::InfiniteMPS, operator::InfiniteMPO,
        above::InfiniteMPS, alg
    )
    # compute eigenvector
    T = TransferMatrix(above.AR, operator, below.AR)
    λ, envs.GRs[end] = fixedpoint(T, envs.GRs[end], :LM, alg)
    # push through unitcell
    for i in reverse(1:(length(operator) - 1))
        envs.GRs[i] = TransferMatrix(
            above.AR[i + 1], operator[i + 1],
            below.AR[i + 1]
        ) * envs.GRs[i + 1]
    end
    return λ, envs
end

# normalization convention of the environments:
# - normalize the right environment to have norm 1
# - normalize the left environment to have overlap 1
# this avoids catastrophic blow-up of norms, while keeping the total normalized
# and does not lead to issues for negative overlaps and real entries.
function TensorKit.normalize!(
        envs::InfiniteEnvironments, below::InfiniteMPS, operator::InfiniteMPO,
        above::InfiniteMPS
    )
    for i in 1:length(operator)
        normalize!(envs.GRs[i])
        Hc = C_hamiltonian(i, below, operator, above, envs)
        λ = dot(below.C[i], Hc * above.C[i])
        scale!(envs.GLs[i + 1], inv(λ))
    end
    return envs
end

# InfiniteMPOHamiltonian environments
# -----------------------------------
function initialize_environments(
        below::InfiniteMPS, operator::InfiniteMPOHamiltonian,
        above::InfiniteMPS = below
    )
    L = check_length(above, operator, below)
    GLs = PeriodicVector([allocate_GL(below, operator, above, i) for i in 1:L])
    GRs = PeriodicVector([allocate_GR(below, operator, above, i) for i in 1:L])

    # GL = (1, 0, 0)
    GL = first(GLs)
    for i in 1:length(GL)
        if i == 1
            GL[i] = isomorphism(storagetype(GL), space(GL[i]))
        else
            fill!(GL[i], zero(scalartype(GL)))
        end
    end

    # GR = (0, 0, 1)^T
    GR = last(GRs)
    for i in 1:length(GR)
        if i == length(GR)
            GR[i] = isomorphism(storagetype(GR), space(GR[i]))
        else
            fill!(GR[i], zero(scalartype(GR)))
        end
    end

    return GLs, GRs
end

function compute_leftenvs!(
        envs::InfiniteEnvironments, below::InfiniteMPS,
        operator::InfiniteMPOHamiltonian, above::InfiniteMPS, alg
    )
    L = check_length(below, above, operator)
    GLs = envs.GLs
    vsize = length(first(GLs))

    @assert above === below "not implemented"

    ρ_left = l_LL(above)
    ρ_right = r_LL(above)

    # the start element
    # TODO: check if this is necessary
    # leftutil = similar(above.AL[1], space(GL[1], 2)[1])
    # fill_data!(leftutil, one)
    # @plansor GL[1][1][-1 -2; -3] = ρ_left[-1; -3] * leftutil[-2]

    (L > 1) && left_cyclethrough!(1, GLs, below, operator, above)

    for i in 2:vsize
        prev = copy(GLs[1][i])
        zerovector!(GLs[1][i])
        left_cyclethrough!(i, GLs, below, operator, above)

        if isidentitylevel(operator, i) # identity matrices; do the hacky renormalization
            T = regularize(TransferMatrix(above.AL, below.AL), ρ_left, ρ_right)
            GLs[1][i], convhist = linsolve(flip(T), GLs[1][i], prev, alg, 1, -1)
            convhist.converged == 0 &&
                @warn "GL$i failed to converge: normres = $(convhist.normres)"

            (L > 1) && left_cyclethrough!(i, GLs, below, operator, above)

            # go through the unitcell, again subtracting fixpoints
            for site in 1:L
                @plansor GLs[site][i][-1 -2; -3] -= GLs[site][i][1 -2; 2] *
                    r_LL(above, site - 1)[2; 1] * l_LL(above, site)[-1; -3]
            end

        else
            if !isemptylevel(operator, i)
                diag = map(h -> h[i, 1, 1, i], operator[:])
                T = TransferMatrix(above.AL, diag, below.AL)
                GLs[1][i], convhist = linsolve(flip(T), GLs[1][i], prev, alg, 1, -1)
                convhist.converged == 0 &&
                    @warn "GL$i failed to converge: normres = $(convhist.normres)"
            end
            (L > 1) && left_cyclethrough!(i, GLs, below, operator, above)
        end
    end

    return GLs
end

function left_cyclethrough!(
        index::Int, GL, below::InfiniteMPS, H::InfiniteMPOHamiltonian,
        above::InfiniteMPS = below
    )
    # TODO: efficient transfer matrix slicing for large unitcells
    leftinds = 1:index
    for site in eachindex(GL)
        GL[site + 1][index] = GL[site][leftinds] * TransferMatrix(
            above.AL[site], H[site][leftinds, 1, 1, index], below.AL[site]
        )
    end
    return GL
end

function compute_rightenvs!(
        envs::InfiniteEnvironments, below::InfiniteMPS,
        operator::InfiniteMPOHamiltonian, above::InfiniteMPS, alg
    )
    L = check_length(above, operator, below)
    GRs = envs.GRs
    vsize = length(last(GRs))

    @assert above === below "not implemented"

    ρ_left = l_RR(above)
    ρ_right = r_RR(above)

    # the start element
    # TODO: check if this is necessary
    # rightutil = similar(state.AL[1], space(GR[end], 2)[end])
    # fill_data!(rightutil, one)
    # @plansor GR[end][end][-1 -2; -3] = r_RR(state)[-1; -3] * rightutil[-2]

    (L > 1) && right_cyclethrough!(vsize, GRs, below, operator, above) # populate other sites

    for i in (vsize - 1):-1:1
        prev = copy(GRs[end][i])
        zerovector!(GRs[end][i])
        right_cyclethrough!(i, GRs, below, operator, above)

        if isidentitylevel(operator, i) # identity matrices; do the hacky renormalization
            # subtract fixpoints
            T = regularize(TransferMatrix(above.AR, below.AR), ρ_left, ρ_right)
            GRs[end][i], convhist = linsolve(T, GRs[end][i], prev, alg, 1, -1)
            convhist.converged == 0 &&
                @warn "GR$i failed to converge: normres = $(convhist.normres)"

            L > 1 && right_cyclethrough!(i, GRs, below, operator, above)

            # go through the unitcell, again subtracting fixpoints
            for site in 1:L
                @plansor GRs[site][i][-1 -2; -3] -= GRs[site][i][1 -2; 2] *
                    l_RR(above, site + 1)[2; 1] * r_RR(above, site)[-1; -3]
            end
        else
            if !isemptylevel(operator, i)
                diag = map(b -> b[i, 1, 1, i], operator[:])
                T = TransferMatrix(above.AR, diag, below.AR)
                GRs[end][i], convhist = linsolve(T, GRs[end][i], prev, alg, 1, -1)
                convhist.converged == 0 &&
                    @warn "GR$i failed to converge: normres = $(convhist.normres)"
            end

            (L > 1) && right_cyclethrough!(i, GRs, below, operator, above)
        end
    end

    return GRs
end

function right_cyclethrough!(
        index::Int, GR, below::InfiniteMPS, operator::InfiniteMPOHamiltonian,
        above::InfiniteMPS = below
    )
    # TODO: efficient transfer matrix slicing for large unitcells
    for site in reverse(eachindex(GR))
        rightinds = index:length(GR[site])
        GR[site - 1][index] = TransferMatrix(
            above.AR[site], operator[site][index, 1, 1, rightinds], below.AR[site]
        ) * GR[site][rightinds]
    end
    return GR
end

# no normalization necessary -- for consistant interface
function TensorKit.normalize!(
        envs::InfiniteEnvironments, below::InfiniteMPS,
        operator::InfiniteMPOHamiltonian, above::InfiniteMPS
    )
    return envs
end

# Transfer operations
# -------------------

function transfer_leftenv!(envs::InfiniteEnvironments, below, operator, above, site::Int)
    T = TransferMatrix(above.AL[site - 1], operator[site - 1], below.AL[site - 1])
    envs.GLs[site] = envs.GLs[site - 1] * T
    return envs
end

function transfer_rightenv!(envs::InfiniteEnvironments, below, operator, above, site::Int)
    T = TransferMatrix(above.AR[site + 1], operator[site + 1], below.AR[site + 1])
    envs.GRs[site] = T * envs.GRs[site + 1]
    return envs
end
