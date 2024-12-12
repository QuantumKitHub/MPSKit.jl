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
function calc_galerkin(state::InfiniteMPS, site::Int,
                       envs::InfiniteMPOHamiltonianEnvironments)
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
                                                 id(storagetype(eltype(H)), lattice[i])
                                            for (i, e) in enumerate(e_local))
    return real(expectation_value(state, (H - H_renormalized)^2))
end

function variance(state::FiniteMPS, H::FiniteMPOHamiltonian, envs=environments(state, H))
    H2 = H * H
    return real(expectation_value(state, H2) -
                expectation_value(state, H, envs)^2)
end

function variance(state::FiniteQP, H::FiniteMPOHamiltonian, args...)
    return variance(convert(FiniteMPS, state), H)
end

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
                                                    id(storagetype(eltype(H)), lattice[i])
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
    periodic_boundary_conditions(mpo::AbstractInfiniteMPO, L::Int)

Convert an infinite MPO into a finite MPO of length `L`, by mapping periodic boundary conditions onto an open system.
"""
function periodic_boundary_conditions(mpo::InfiniteMPO{O},
                                      L=length(mpo)) where {O}
    mod(L, length(mpo)) == 0 ||
        throw(ArgumentError("length $L is not a multiple of the infinite unitcell"))

    # allocate output
    output = Vector{O}(undef, L)
    V_wrap = left_virtualspace(mpo, 1)'
    ST = storagetype(O)

    util = fill!(similar(mpo[1], oneunit(V_wrap)), one(scalartype(O)))
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

    mpo isa SparseMPO && dropzeros!.(output) # the above process fills sparse mpos with zeros.

    return FiniteMPO(output)
end

# TODO: check if this is correct?
function periodic_boundary_conditions(H::InfiniteMPOHamiltonian, L=length(H))
    Hmpo = periodic_boundary_conditions(InfiniteMPO(H), L)
    return FiniteMPOHamiltonian(parent(Hmpo))
end

"""
    open_boundary_conditions(mpo::InfiniteMPOHamiltonian, L::Int) -> FiniteMPOHamiltonian

Convert an infinite MPO into a finite MPO of length `L`, by applying open boundary conditions.
"""
function open_boundary_conditions(mpo::InfiniteMPOHamiltonian,
                                  L=length(mpo))
    mod(L, length(mpo)) == 0 ||
        throw(ArgumentError("length $L is not a multiple of the infinite unitcell"))

    # Make a FiniteMPO, filling it up with the tensors of H
    # Only keep top row of the first and last column of the last MPO tensor

    # allocate output
    output = Vector(repeat(copy(parent(mpo)), L ÷ length(mpo)))
    output[1] = output[1][1, :, :, :]
    output[end] = output[end][:, :, :, 1]

    return FiniteMPOHamiltonian(output)
end
