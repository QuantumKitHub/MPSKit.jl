"""
    entropy(state, [site::Int])

Calculate the Von Neumann entanglement entropy of a given MPS. If an integer `site` is
given, the entropy is across the entanglement cut to the right of site `site`. Otherwise, a
vector of entropies is returned, one for each site.
"""
entropy(state::InfiniteMPS) = map(c -> -tr(safe_xlogx(c * c')), state.CR);
function entropy(state::Union{FiniteMPS,WindowMPS,InfiniteMPS}, loc::Int)
    return -tr(safe_xlogx(state.CR[loc] * state.CR[loc]'))
end;

# function infinite_temperature(H::MPOHamiltonian)
#     return [permute(isomorphism(storagetype(H[1, 1, 1]), oneunit(sp) * sp,
#                                 oneunit(sp) * sp), (1, 2, 4), (3,)) for sp in H.pspaces]
# end

"""
    calc_galerkin(state, envs)

Calculate the galerkin error.
"""
function calc_galerkin(state::Union{InfiniteMPS,FiniteMPS,WindowMPS}, loc, envs)::Float64
    AC´ = ∂∂AC(loc, state, envs.operator, envs) * state.AC[loc]
    normalize!(AC´)
    out = add!(AC´, state.AL[loc] * state.AL[loc]' * AC´, -1)
    return norm(out)
end
function calc_galerkin(state::Union{InfiniteMPS,FiniteMPS,WindowMPS}, envs)::Float64
    return maximum([calc_galerkin(state, loc, envs) for loc in 1:length(state)])
end
function calc_galerkin(state::MPSMultiline, envs::InfiniteMPOEnvironments)::Float64
    above = isnothing(envs.above) ? state : envs.above

    εs = zeros(Float64, size(state, 1), size(state, 2))
    for (row, col) in product(1:size(state, 1), 1:size(state, 2))
        AC´ = ∂∂AC(row, col, state, envs.operator, envs) * above.AC[row, col]
        normalize!(AC´)
        out = add!(AC´, state.AL[row + 1, col] * state.AL[row + 1, col]' * AC´, -1)
        εs[row, col] = norm(out)
    end

    return maximum(εs[:])
end
function calc_galerkin(state::InfiniteMPS, site::Int, envs::InfiniteEnvironments)
    AC´ = ∂∂AC(site, state, envs.operator, envs) * state.AC[site]
    normalize!(AC´)
    out = add!(AC´, state.AL[site] * (state.AL[site]' * AC´), -1)
    return norm(out)
end

"""
    transfer_spectrum(above::InfiniteMPS; below=above, tol=Defaults.tol, num_vals=20,
                           sector=first(sectors(oneunit(left_virtualspace(above, 1)))))

Calculate the partial spectrum of the left mixed transfer matrix corresponding to the
overlap of a given `above` state and a `below` state. The `sector` keyword argument can be
used to specify a non-trivial total charge for the transfer matrix eigenvectors.
Specifically, an auxiliary space `ℂ[typeof(sector)](sector => 1)'` will be added to the
domain of each eigenvector. The `tol` and `num_vals` keyword arguments are passed to
`KrylovKit.eigolve`
"""
function transfer_spectrum(above::InfiniteMPS; below=above, tol=Defaults.tol, num_vals=20,
                           sector=first(sectors(oneunit(left_virtualspace(above, 1)))))
    init = randomize!(similar(above.AL[1], left_virtualspace(below, 0),
                              ℂ[typeof(sector)](sector => 1)' * left_virtualspace(above, 0)))

    transferspace = fuse(left_virtualspace(above, 0) * left_virtualspace(below, 0)')
    num_vals = min(dim(transferspace, sector), num_vals) # we can ask at most this many values
    eigenvals, eigenvecs, convhist = eigsolve(flip(TransferMatrix(above.AL, below.AL)),
                                              init, num_vals, :LM; tol=tol)
    convhist.converged < num_vals &&
        @warn "correlation length failed to converge: normres = $(convhist.normres)"

    return eigenvals
end

"""
    entanglement_spectrum(ψ, [site::Int=0]) -> SectorDict{sectortype(ψ),Vector{<:Real}}

Compute the entanglement spectrum at a given site, i.e. the singular values of the gauge
matrix to the right of a given site. This is a dictionary mapping the charge to the singular
values.
"""
function entanglement_spectrum(st::Union{InfiniteMPS,FiniteMPS,WindowMPS}, site::Int=0)
    @assert site <= length(st)
    _, S, = tsvd(st.CR[site])
    return TensorKit.SectorDict(c => real(diag(b)) for (c, b) in blocks(S))
end

"""
Find the closest fractions of π, differing at most ```tol_angle```
"""
function approx_angles(spectrum; tol_angle=0.1)
    angles = angle.(spectrum) ./ π                          # ∈ ]-1, 1]
    angles_approx = rationalize.(angles, tol=tol_angle)     # ∈ [-1, 1]

    # Remove the effects of the branchcut.
    angles_approx[findall(angles_approx .== -1)] .= 1       # ∈ ]-1, 1]

    return angles_approx .* π                               # ∈ ]-π, π]
end

"""
Given an InfiniteMPS, compute the gap ```ϵ``` for the asymptotics of the transfer matrix, as
well as the Marek gap ```δ``` as a scaling measure of the bond dimension.
"""
function marek_gap(above::InfiniteMPS; tol_angle=0.1, kwargs...)
    spectrum = transfer_spectrum(above; kwargs...)
    return marek_gap(spectrum; tol_angle)
end

function marek_gap(spectrum; tol_angle=0.1)
    # Remove 1s from the spectrum
    inds = findall(abs.(spectrum) .< 1 - 1e-12)
    length(spectrum) - length(inds) < 2 || @warn "Non-injective mps?"

    spectrum = spectrum[inds]

    angles = approx_angles(spectrum; tol_angle=tol_angle)
    θ = first(angles)

    spectrum_at_angle = spectrum[findall(angles .== θ)]

    lambdas = -log.(abs.(spectrum_at_angle))

    ϵ = first(lambdas)

    δ = Inf
    if length(lambdas) > 2
        δ = lambdas[2] - lambdas[1]
    end

    return ϵ, δ, θ
end

"""
    correlation_length(above::InfiniteMPS; kwargs...)

Compute the correlation length of a given InfiniteMPS based on the next-to-leading
eigenvalue of the transfer matrix. The `kwargs` are passed to [`transfer_spectrum`](@ref),
and can for example be used to target the correlation length in a specific sector. 
"""
function correlation_length(above::InfiniteMPS; kwargs...)
    ϵ, = marek_gap(above; kwargs...)
    return 1 / ϵ
end

function correlation_length(spectrum; kwargs...)
    ϵ, = marek_gap(spectrum; kwargs...)
    return 1 / ϵ
end

"""
    variance(state, hamiltonian, [envs=environments(state, hamiltonian)])

Compute the variance of the energy of the state with respect to the hamiltonian.
"""
function variance end

# function variance(state::InfiniteMPS, H::MPOHamiltonian,
#                   envs=environments(state, H))
#     # first rescale, such that the ground state energy is zero
#     # this needs to be done in a way that is consistent with the computation of the environments
#     # TODO: actually figure out why/if this is correct
#     e_local = map(1:length(state)) do i
#         return sum(1:(H.odim)) do j
#             @plansor (leftenv(envs, i, state)[j] *
#                       TransferMatrix(state.AC[i], H[i][j, H.odim], state.AC[i]))[1 2; 3] *
#                      rightenv(envs, i, state)[H.odim][3 2; 1]
#         end
#     end
#     rescaled_H = H - e_local
#
#     return real(expectation_value(state, rescaled_H * rescaled_H))
# end
function variance(state::InfiniteMPS, H::InfiniteMPOHamiltonian,
                  envs=environments(state, H))
    e_local = map(1:length(state)) do i
        return @plansor state.AC[i][3 7; 5] *
                        leftenv(envs, i, state)[1 2; 3] *
                        H[i][:, :, :, end][2 4; 7 8] *
                        rightenv(envs, i, state)[end][5 8; 6] *
                        conj(state.AC[i][1 4; 6])
    end
    lattice = physicalspace.(Ref(state), 1:length(state))
    H_renormalized = InfiniteMPOHamiltonian(lattice,
                                            i => e *
                                                 id(Matrix{scalartype(state)}, lattice[i])
                                            for (i, e) in enumerate(e_local))
    return real(expectation_value(state, (H - H_renormalized)^2))
end

# function variance(state::FiniteMPS, H::MPOHamiltonian, envs=environments(state, H))
#     H2 = H * H
#     return real(expectation_value(state, H2) -
#                 expectation_value(state, H, envs)^2)
# end
function variance(state::FiniteMPS, H::FiniteMPOHamiltonian, envs=environments(state, H))
    H2 = H * H
    return real(expectation_value(state, H2) -
                expectation_value(state, H, envs)^2)
end

function variance(state::FiniteQP, H::FiniteMPOHamiltonian, args...)
    return variance(convert(FiniteMPS, state), H)
end

# function variance(state::InfiniteQP, H::MPOHamiltonian, envs=environments(state, H))
#     # I remember there being an issue here @gertian?
#     state.trivial ||
#         throw(ArgumentError("variance of domain wall excitations is not implemented"))
#     gs = state.left_gs
#
#     rescaled_H = H - expectation_value(gs, H)
#
#     #I don't remember where the formula came from
#     # TODO: this is probably broken
#     E_ex = dot(state, effective_excitation_hamiltonian(H, state, envs))
#
#     rescaled_envs = environments(gs, rescaled_H)
#     GL = leftenv(rescaled_envs, 1, gs)
#     GR = rightenv(rescaled_envs, 0, gs)
#     E_f = sum(zip(GL, GR)) do (gl, gr)
#         @plansor gl[5 3; 1] * gs.CR[0][1; 4] * conj(gs.CR[0][5; 2]) * gr[4 3; 2]
#     end
#
#     H2 = rescaled_H * rescaled_H
#
#     return real(dot(state, effective_excitation_hamiltonian(H2, state)) -
#                 2 * (E_f + E_ex) * E_ex + E_ex^2)
# end
function variance(state::InfiniteQP, H::InfiniteMPOHamiltonian, envs=environments(state, H))
    # I remember there being an issue here @gertian?
    state.trivial ||
        throw(ArgumentError("variance of domain wall excitations is not implemented"))
    gs = state.left_gs

    e_local = map(1:length(state)) do i
        return @plansor gs.AC[i][3 7; 5] *
                        leftenv(envs.leftenvs, i, gs)[1 2; 3] *
                        H[i][:, :, :, end][2 4; 7 8] *
                        rightenv(envs.rightenvs, i, gs)[end][5 8; 6] *
                        conj(gs.AC[i][1 4; 6])
    end
    lattice = physicalspace.(Ref(gs), 1:length(state))
    H_regularized = H - InfiniteMPOHamiltonian(lattice,
                                               i => e *
                                                    id(Matrix{scalartype(state)}, lattice[i])
                                               for (i, e) in enumerate(e_local))

    # I don't remember where the formula came from
    # TODO: this is probably broken
    E_ex = dot(state, effective_excitation_hamiltonian(H, state, envs))

    rescaled_envs = environments(gs, H_regularized)
    GL = leftenv(rescaled_envs, 1, gs)
    GR = rightenv(rescaled_envs, 0, gs)
    E_f = @plansor GL[5 3; 1] * gs.CR[0][1; 4] * conj(gs.CR[0][5; 2]) * GR[4 3; 2]

    H2 = H_regularized^2

    return real(dot(state, effective_excitation_hamiltonian(H2, state)) -
                2 * (E_f + E_ex) * E_ex + E_ex^2)
end

function variance(ψ, H::LazySum, envs=environments(ψ, sum(H)))
    # TODO: avoid throwing error and just compute correct environments 
    envs isa MultipleEnvironments &&
        throw(ArgumentError("The environment cannot be Lazy i.e. use environments of sum(H)"))
    return variance(ψ, sum(H), envs)
end

"""
You can impose periodic boundary conditions on an mpo-hamiltonian (for a given size)
That creates a new mpo-hamiltonian with larger bond dimension
The interaction never wraps around multiple times
"""
# function periodic_boundary_conditions(H::MPOHamiltonian{S,T,E},
#                                       len=H.period) where {S,T,E}
#     sanitycheck(H) || throw(ArgumentError("invalid hamiltonian"))
#     mod(len, H.period) == 0 ||
#         throw(ArgumentError("$(len) is not a multiple of unitcell"))
#
#     fusers = PeriodicArray(map(1:len) do loc
#                                map(Iterators.product(H.domspaces[loc, :],
#                                                      H.domspaces[len + 1, :],
#                                                      H.domspaces[loc, :])) do (v1, v2,
#                                                                                v3)
#                                    return isomorphism(storagetype(T), fuse(v1 * v2' * v3),
#                                                       v1 * v2' * v3)
#                                end
#                            end)
#
#     #a -> what progress have I made in the upper layer?
#     #b -> what virtual space did I "lend" in the beginning?
#     #c -> what progress have I made in the lower layer?
#     χ = H.odim
#
#     indmap = zeros(Int, χ, χ, χ)
#     χ´ = 0
#     for b in χ:-1:2, c in b:χ
#         χ´ += 1
#         indmap[1, b, c] = χ´
#     end
#
#     for a in 2:χ, b in χ:-1:a
#         χ´ += 1
#         indmap[a, b, χ] = χ´
#     end
#
#     #do the bulk
#     bulk = PeriodicArray(convert(Array{Union{T,E},3}, fill(zero(E), H.period, χ´, χ´)))
#
#     for loc in 1:(H.period), (j, k) in keys(H[loc])
#
#         #apply (j,k) above
#         l = H.odim
#         for i in 2:(H.odim)
#             k <= i && i <= l || continue
#
#             f1 = fusers[loc][j, i, l]
#             f2 = fusers[loc + 1][k, i, l]
#             j′ = indmap[j, i, l]
#             k′ = indmap[k, i, l]
#             @plansor bulk[loc, j′, k′][-1 -2; -3 -4] := H[loc][j, k][1 2; -3 6] *
#                                                         f1[-1; 1 3 5] *
#                                                         conj(f2[-4; 6 7 8]) * τ[2 3; 7 4] *
#                                                         τ[4 5; 8 -2]
#         end
#
#         #apply (j,k) below
#         i = 1
#         for l in 2:(H.odim - 1)
#             l > 1 && l >= i && l <= j || continue
#
#             f1 = fusers[loc][i, l, j]
#             f2 = fusers[loc + 1][i, l, k]
#             j′ = indmap[i, l, j]
#             k′ = indmap[i, l, k]
#             @plansor bulk[loc, j′, k′][-1 -2; -3 -4] := H[loc][j, k][1 -2; 3 6] *
#                                                         f1[-1; 4 2 1] *
#                                                         conj(f2[-4; 8 7 6]) * τ[5 2; 7 3] *
#                                                         τ[-3 4; 8 5]
#         end
#     end
#
#     # make the starter
#     starter = convert(Array{Union{T,E},2}, fill(zero(E), χ´, χ´))
#     for (j, k) in keys(H[1])
#
#         #apply (j,k) above
#         if j == 1
#             f1 = fusers[1][1, end, end]
#             f2 = fusers[2][k, end, end]
#             j′ = indmap[k, H.odim, H.odim]
#             @plansor starter[1, j′][-1 -2; -3 -4] := H[1][j, k][-1 -2; -3 2] *
#                                                      conj(f2[-4; 2 3 3])
#         end
#
#         #apply (j,k) below
#         if j > 1 && j < H.odim
#             f1 = fusers[1][1, j, j]
#             f2 = fusers[2][1, j, k]
#
#             @plansor starter[1, indmap[1, j, k]][-1 -2; -3 -4] := H[1][j, k][4 -2; 3 1] *
#                                                                   conj(f2[-4; 6 2 1]) *
#                                                                   τ[5 4; 2 3] *
#                                                                   τ[-3 -1; 6 5]
#         end
#     end
#     starter[1, 1] = one(E)
#     starter[end, end] = one(E)
#
#     # make the ender
#     ender = convert(Array{Union{T,E},2}, fill(zero(E), χ´, χ´))
#     for (j, k) in keys(H[H.period])
#         if k > 1
#             f1 = fusers[end][j, k, H.odim]
#             k′ = indmap[j, k, H.odim]
#             @plansor ender[k′, end][-1 -2; -3 -4] := f1[-1; 1 2 6] *
#                                                      H[H.period][j, k][1 3; -3 4] *
#                                                      τ[3 2; 4 5] * τ[5 6; -4 -2]
#         end
#     end
#     ender[1, 1] = one(E)
#     ender[end, end] = one(E)
#
#     # fill in the entire H
#     nos = convert(Array{Union{T,E},3}, fill(zero(E), len, χ´, χ´))
#     nos[1, :, :] = starter[:, :]
#     nos[end, :, :] = ender[:, :]
#
#     for i in 2:(len - 1)
#         nos[i, :, :] = bulk[i, :, :]
#     end
#
#     return MPOHamiltonian(nos)
# end

#impose periodic boundary conditions on a normal mpo
"""
    periodic_boundary_conditions(mpo::AbstractInfiniteMPO, L::Int)

Convert an infinite MPO into a finite MPO of length `L`, by mapping periodic boundary conditions onto an open system.
"""
function periodic_boundary_conditions(mpo::Union{InfiniteMPO{O},DenseMPO{O}},
                                      L=length(mpo)) where {O}
    mod(L, length(mpo)) == 0 ||
        throw(ArgumentError("length $L is not a multiple of the infinite unitcell"))

    # allocate output
    output = Vector{O}(undef, L)
    V_wrap = left_virtualspace(mpo, 1)'
    ST = storagetype(O)

    util = fill!(Tensor{scalartype(O)}(undef, oneunit(V_wrap)), one(scalartype(O)))
    @plansor cup[-1; -2 -3] := id(ST, V_wrap)[-3; -2] * util[-1]

    local F_right
    for i in 1:L
        V_left = i == 1 ? oneunit(V_wrap) : fuse(V_wrap ⊗ left_virtualspace(mpo, i))
        V_right = i == L ? oneunit(V_wrap) : fuse(V_wrap' ⊗ right_virtualspace(mpo, i)')
        output[i] = similar(mpo[i],
                            V_left * physicalspace(mpo, i) ←
                            physicalspace(mpo, i) * V_right)
        F_left = i == 1 ? cup : F_right
        F_right = i == L ? cup :
                  isomorphism(ST, V_right ← V_wrap * right_virtualspace(mpo, i)')
        @plansor output[i][-1 -2; -3 -4] = F_left[-1; 1 2] *
                                           τ[-3 1; 4 3] *
                                           mpo[i][2 -2; 3 5] *
                                           conj(F_right[-4; 4 5])
    end

    return mpo isa DenseMPO ? DenseMPO(FiniteMPO(output)) : FiniteMPO(output)
end

function periodic_boundary_conditions(H::InfiniteMPOHamiltonian, L=length(H))
    Hmpo = periodic_boundary_conditions(InfiniteMPO(H), L)
    return FiniteMPOHamiltonian(Hmpo.data)
end

"""
    open_boundary_conditions(mpo::AbstractInfiniteMPO, L::Int)

Convert an infinite MPO into a finite MPO of length `L`, by applying open boundary conditions.
"""
function open_boundary_conditions(mpo::Union{InfiniteMPO,DenseMPO},
                                  L=length(mpo))
    mod(L, length(mpo)) == 0 ||
        throw(ArgumentError("length $L is not a multiple of the infinite unitcell"))

    # Make a FiniteMPO, filling it up with the tensors of H
    # Only keep top row of the first and last column of the last MPO tensor

    # allocate output
    output = Vector(repeat(copy(mpo.data), L ÷ length(mpo)))
    output[1] = output[1][1, 1, 1, :]
    output[end] = output[end][:, 1, 1, 1]

    return mpo isa DenseMPO ? DenseMPO(output) : FiniteMPO(output)
end

function open_boundary_conditions(H::InfiniteMPOHamiltonian, L=length(H))
    Hmpo = open_boundary_conditions(InfiniteMPO(H), L)
    return FiniteMPOHamiltonian(Hmpo.data)
end
