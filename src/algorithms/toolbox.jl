"""
    entropy(state, [site::Int])

Calculate the Von Neumann entanglement entropy of a given MPS. If an integer `site` is
given, the entropy is across the entanglement cut to the right of site `site`. Otherwise, a
vector of entropies is returned, one for each site.
"""
entropy(state::InfiniteMPS) = map(Base.Fix1(entropy, state), 1:length(state))
function entropy(state::Union{FiniteMPS, WindowMPS, InfiniteMPS}, loc::Int)
    S = zero(real(scalartype(state)))
    tol = eps(typeof(S))
    for (c, b) in entanglement_spectrum(state, loc)
        s = zero(S)
        for x in b
            x < tol && break
            x² = x^2
            s += x² * log(x²)
        end
        S += oftype(S, dim(c) * s)
    end
    return -S
end

"""
    infinite_temperature_density_matrix(H::MPOHamiltonian) -> MPO

Return the density matrix of the infinite temperature state for a given Hamiltonian.
This is the identity matrix in the physical space, and the identity in the auxiliary space.
"""
function infinite_temperature_density_matrix(H::MPOHamiltonian)
    V = first(left_virtualspace(H[1]))
    W = map(1:length(H)) do site
        return BraidingTensor{scalartype(H)}(physicalspace(H, site), V)
    end
    return isfinite(H) ? FiniteMPO(W) : InfiniteMPO(W)
end

"""
    calc_galerkin(below, operator, above, envs)
    calc_galerkin(pos, below, operator, above, envs)

Calculate the Galerkin error, which is the error between the solution of the original problem, and the solution of the problem projected on the tangent space.
Concretely, this is the overlap of the current state with the single-site derivative, projected onto the nullspace of the current state:

```math
\\epsilon = |VL * (VL' * \\frac{above}{\\partial AC_{pos}})|
```
"""
function calc_galerkin(
        pos::Int, below::Union{InfiniteMPS, FiniteMPS, WindowMPS}, operator, above, envs
    )
    AC´ = AC_hamiltonian(pos, below, operator, above, envs) * above.AC[pos]
    normalize!(AC´)
    out = add!(AC´, below.AL[pos] * below.AL[pos]' * AC´, -1)
    return norm(out)
end
function calc_galerkin(
        pos::CartesianIndex{2}, below::MultilineMPS, operator::MultilineMPO,
        above::MultilineMPS, envs::MultilineEnvironments
    )
    row, col = pos.I
    return calc_galerkin(col, below[row + 1], operator[row], above[row], envs[row])
end
function calc_galerkin(
        below::Union{InfiniteMPS, FiniteMPS, WindowMPS}, operator, above, envs
    )
    return maximum(pos -> calc_galerkin(pos, below, operator, above, envs), 1:length(above))
end
function calc_galerkin(
        below::MultilineMPS, operator::MultilineMPO, above::MultilineMPS,
        envs::MultilineEnvironments
    )
    return maximum(
        pos -> calc_galerkin(pos, below, operator, above, envs),
        CartesianIndices(size(above))
    )
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
function transfer_spectrum(
        above::InfiniteMPS; below = above, tol = Defaults.tol, num_vals = 20,
        sector = first(sectors(oneunit(left_virtualspace(above, 1))))
    )
    init = randomize!(
        similar(
            above.AL[1], left_virtualspace(below, 1),
            ℂ[typeof(sector)](sector => 1)' * left_virtualspace(above, 1)
        )
    )

    transferspace = fuse(left_virtualspace(above, 1) * left_virtualspace(below, 1)')
    num_vals = min(dim(transferspace, sector), num_vals) # we can ask at most this many values
    eigenvals, eigenvecs, convhist = eigsolve(
        flip(TransferMatrix(above.AL, below.AL)), init, num_vals, :LM; tol = tol
    )
    convhist.converged < num_vals &&
        @warn "correlation length failed to converge: normres = $(convhist.normres)"

    return eigenvals
end

"""
    entanglement_spectrum(ψ, site::Int) -> SectorDict{sectortype(ψ),Vector{<:Real}}

Compute the entanglement spectrum at a given site, i.e. the singular values of the gauge
matrix to the right of a given site. This is a dictionary mapping the charge to the singular
values.

For `InfiniteMPS` and `WindowMPS` the default value for `site` is 0.

For `FiniteMPS` no default value for `site` is given, it is up to the user to specify.
"""
function entanglement_spectrum(st::Union{InfiniteMPS, WindowMPS}, site::Int = 0)
    checkbounds(st, site)
    return LinearAlgebra.svdvals(st.C[site])
end
function entanglement_spectrum(st::FiniteMPS, site::Int)
    checkbounds(st, site)
    return LinearAlgebra.svdvals(st.C[site])
end

"""
Find the closest fractions of π, differing at most ```tol_angle```
"""
function approx_angles(spectrum; tol_angle = 0.1)
    angles = angle.(spectrum) ./ π                          # ∈ ]-1, 1]
    angles_approx = rationalize.(angles, tol = tol_angle)     # ∈ [-1, 1]

    # Remove the effects of the branchcut.
    angles_approx[findall(angles_approx .== -1)] .= 1       # ∈ ]-1, 1]

    return angles_approx .* π                               # ∈ ]-π, π]
end

"""
Given an InfiniteMPS, compute the gap ```ϵ``` for the asymptotics of the transfer matrix, as
well as the Marek gap ```δ``` as a scaling measure of the bond dimension.
"""
function marek_gap(above::InfiniteMPS; tol_angle = 0.1, kwargs...)
    spectrum = transfer_spectrum(above; kwargs...)
    return marek_gap(spectrum; tol_angle)
end

function marek_gap(spectrum; tol_angle = 0.1)
    # Remove 1s from the spectrum
    inds = findall(abs.(spectrum) .< 1 - 1.0e-12)
    length(spectrum) - length(inds) < 2 || @warn "Non-injective mps?"

    spectrum = spectrum[inds]

    angles = approx_angles(spectrum; tol_angle = tol_angle)
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

function variance(
        state::InfiniteMPS, H::InfiniteMPOHamiltonian, envs = environments(state, H)
    )
    e_local = map(1:length(state)) do i
        return contract_mpo_expval(
            state.AC[i], envs.GLs[i], H[i][:, :, :, end], envs.GRs[i][end]
        )
    end
    lattice = physicalspace(state)
    H_renormalized = InfiniteMPOHamiltonian(
        lattice, i => e * id(storagetype(eltype(H)), lattice[i]) for (i, e) in enumerate(e_local)
    )
    return real(expectation_value(state, (H - H_renormalized)^2))
end

function variance(state::FiniteMPS, H::FiniteMPOHamiltonian, envs = environments(state, H))
    H2 = H * H
    return real(expectation_value(state, H2) - expectation_value(state, H, envs)^2)
end

function variance(state::FiniteQP, H::FiniteMPOHamiltonian, args...)
    return variance(convert(FiniteMPS, state), H)
end

function variance(state::InfiniteQP, H::InfiniteMPOHamiltonian, envs = environments(state, H))
    # I remember there being an issue here @gertian?
    state.trivial ||
        throw(ArgumentError("variance of domain wall excitations is not implemented"))
    gs = state.left_gs

    e_local = map(1:length(state)) do i
        GL = leftenv(envs, i, gs)
        GR = rightenv(envs, i, gs)
        return contract_mpo_expval(gs.AC[i], GL, H[i][:, :, :, end], GR[end])
    end
    lattice = physicalspace(gs)
    H_regularized = H - InfiniteMPOHamiltonian(
        lattice, i => e * id(storagetype(eltype(H)), lattice[i]) for (i, e) in enumerate(e_local)
    )

    # I don't remember where the formula came from
    # TODO: this is probably broken
    E_ex = dot(state, effective_excitation_hamiltonian(H, state, envs))

    rescaled_envs = environments(gs, H_regularized)
    GL = leftenv(rescaled_envs, 1, gs)
    GR = rightenv(rescaled_envs, 0, gs)
    E_f = @plansor GL[5 3; 1] * gs.C[0][1; 4] * conj(gs.C[0][5; 2]) * GR[4 3; 2]

    H2 = H_regularized^2

    return real(
        dot(state, effective_excitation_hamiltonian(H2, state)) - 2 * (E_f + E_ex) * E_ex + E_ex^2
    )
end

function variance(ψ, H::LazySum, envs = environments(ψ, sum(H)))
    # TODO: avoid throwing error and just compute correct environments
    envs isa MultipleEnvironments &&
        throw(ArgumentError("The environment cannot be Lazy i.e. use environments of sum(H)"))
    return variance(ψ, sum(H), envs)
end

"""
    periodic_boundary_conditions(mpo::AbstractInfiniteMPO, L::Int)

Convert an infinite MPO into a finite MPO of length `L`, by mapping periodic boundary conditions onto an open system.
"""
function periodic_boundary_conditions(mpo::InfiniteMPO{O}, L = length(mpo)) where {O}
    mod(L, length(mpo)) == 0 ||
        throw(ArgumentError("length $L is not a multiple of the infinite unitcell"))
    # allocate output
    output = Vector{O}(undef, L)
    V_wrap = left_virtualspace(mpo, 1)
    ST = storagetype(O)

    util = isometry(storagetype(O), oneunit(V_wrap) ← one(V_wrap))
    @plansor cup[-1; -2 -3] := id(ST, V_wrap)[-2; -3] * util[-1]

    local F_right
    for i in 1:L
        V_left = i == 1 ? oneunit(V_wrap) : fuse(V_wrap ⊗ left_virtualspace(mpo, i))
        V_right = i == L ? oneunit(V_wrap) : fuse(V_wrap ⊗ right_virtualspace(mpo, i))
        output[i] = similar(
            mpo[i], V_left * physicalspace(mpo, i) ← physicalspace(mpo, i) * V_right
        )
        F_left = i == 1 ? cup : F_right
        F_right = i == L ? cup : isomorphism(ST, V_right ← V_wrap' * right_virtualspace(mpo, i))
        @plansor output[i][-1 -2; -3 -4] = F_left[-1; 1 2] * τ[-3 1; 4 3] *
            mpo[i][2 -2; 3 5] * conj(F_right[-4; 4 5])
    end

    return remove_orphans!(FiniteMPO(output))
end

function _indmap_pbc(chi)
    indmap = Dict{NTuple{3, Int}, Int}()
    chi_ = 0
    for b in reverse(2:chi), c in b:chi
        chi_ += 1
        indmap[1, b, c] = chi_
    end
    for a in 2:chi, b in reverse(a:chi)
        chi_ += 1
        indmap[a, b, chi] = chi_
    end
    return indmap, chi_
end

function periodic_boundary_conditions(H::InfiniteMPOHamiltonian, L = length(H))
    mod(L, length(H)) == 0 ||
        throw(ArgumentError("length $L is not a multiple of the infinite unitcell"))
    # @assert size(H[1], 1) > 2 "Not implemented"
    O = eltype(H)
    A = storagetype(O)
    S = spacetype(O)
    chi = size(H[1], 1)

    # linearize indices:
    indmap, chi_ = _indmap_pbc(chi)

    # compute all fusers
    V_wrap = left_virtualspace(H, 1)
    fusers = PeriodicVector(
        map(1:L) do i
            V_top = left_virtualspace(H, i)
            V_bot = left_virtualspace(H, i)
            return map(Iterators.product(V_top.spaces, V_wrap.spaces, V_bot.spaces)) do (v_top, v_wrap, v_bot)
                return isomorphism(A, fuse(v_top ⊗ v_wrap' ⊗ v_bot), v_top ⊗ v_wrap' ⊗ v_bot)
            end
        end
    )

    # allocate output
    output = Vector{O}(undef, L)
    for site in 1:L
        V_left = if site == 1
            oneunit(V_wrap)
        else
            vs = Vector{S}(undef, chi_)
            for (k, v) in indmap
                vs[v] = _firstspace(fusers[site][k...])
            end
            SumSpace(vs)
        end
        V_right = if site == L
            oneunit(V_wrap)
        else
            vs = Vector{S}(undef, chi_)
            for (k, v) in indmap
                vs[v] = _firstspace(fusers[site + 1][k...])
            end
            SumSpace(vs)
        end
        output[site] = similar(
            H[site], V_left ⊗ physicalspace(H, site) ← physicalspace(H, site) ⊗ V_right
        )
    end

    # bulk changes
    for site in 2:(L - 1)
        V_left = Vector{S}(undef, chi_)
        V_right = Vector{S}(undef, chi_)

        for (I, h) in nonzero_pairs(H[site])
            j, _, _, k = I.I
            # apply [j, k] above
            l = chi
            for i in max(2, k):min(l, chi)
                F_left = fusers[site][j, i, l]
                F_right = fusers[site + 1][k, i, l]
                j′ = indmap[j, i, l]
                k′ = indmap[k, i, l]
                ((j′ == 1 && k′ == 1) || (j′ == size(output[site], 1) && k′ == size(output[site], 4))) && continue
                @plansor o[-1 -2; -3 -4] := h[1 2; -3 6] * F_left[-1; 1 3 5] *
                    conj(F_right[-4; 6 7 8]) * τ[2 3; 7 4] * τ[4 5; 8 -2]
                output[site][j′, 1, 1, k′] = o
            end

            # apply [j, k] below
            i = 1
            for l in max(2, i):min(chi - 1, j)
                F_left = fusers[site][i, l, j]
                F_right = fusers[site + 1][i, l, k]
                j′ = indmap[i, l, j]
                k′ = indmap[i, l, k]
                ((j′ == 1 && k′ == 1) || (j′ == size(output[site], 1) && k′ == size(output[site], 4))) && continue
                @plansor o[-1 -2; -3 -4] := h[1 -2; 3 6] * F_left[-1; 4 2 1] *
                    conj(F_right[-4; 8 7 6]) * τ[5 2; 7 3] * τ[-3 4; 8 5]
                output[site][j′, 1, 1, k′] = o
            end
        end
    end

    # starter
    for (I, h) in nonzero_pairs(H[1])
        j, _, _, k = I.I
        # apply [j, k] above
        if j == 1
            F_right = fusers[2][k, end, end]
            j′ = indmap[k, chi, chi]
            j′ == 1 && continue
            @plansor o[-1 -2; -3 -4] := h[-1 -2; -3 2] * conj(F_right[-4; 2 3 3])
            output[1][1, 1, 1, j′] = o
        end

        # apply [j, k] below
        if 1 < j < chi
            F_right = fusers[2][1, j, k]
            j′ = indmap[1, j, k]
            j′ == 1 && continue
            @plansor o[-1 -2; -3 -4] := h[4 -2; 3 1] * conj(F_right[-4; 6 2 1]) *
                τ[5 4; 2 3] * τ[-3 -1; 6 5]
            output[1][1, 1, 1, j′] = o
        end
    end

    # ender
    for (I, h) in nonzero_pairs(H[end])
        j, _, _, k = I.I
        if k > 1
            F_left = fusers[end][j, k, chi]
            k′ = indmap[j, k, chi]
            k′ == size(output[end], 1) && continue
            @plansor o[-1 -2; -3 -4] := F_left[-1; 1 2 6] * h[1 3; -3 4] * τ[3 2; 4 5] *
                τ[5 6; -4 -2]
            output[end][k′, 1, 1, 1] = o
        end
    end

    return remove_orphans!(FiniteMPOHamiltonian(output))
end

"""
    open_boundary_conditions(mpo::InfiniteMPO, L::Int) -> FiniteMPO

Convert an infinite MPO into a finite MPO of length `L`, by applying open boundary conditions.
"""
function open_boundary_conditions(mpo::InfiniteMPO{O}, L = length(mpo)) where {O <: SparseBlockTensorMap}
    mod(L, length(mpo)) == 0 ||
        throw(ArgumentError("length $L is not a multiple of the infinite unitcell"))

    # Make a FiniteMPO, filling it up with the tensors of H
    # Only keep top row of the first and last column of the last MPO tensor

    # allocate output
    output = Vector(repeat(copy(parent(mpo)), L ÷ length(mpo)))
    output[1] = output[1][1, :, :, :]
    output[end] = output[end][:, :, :, 1]

    return FiniteMPO(output)
end

"""
    open_boundary_conditions(mpo::InfiniteMPOHamiltonian, L::Int) -> FiniteMPOHamiltonian

Convert an infinite MPO into a finite MPO of length `L`, by applying open boundary conditions.
"""
function open_boundary_conditions(mpo::InfiniteMPOHamiltonian, L = length(mpo))
    mod(L, length(mpo)) == 0 ||
        throw(ArgumentError("length $L is not a multiple of the infinite unitcell"))

    # Make a FiniteMPO, filling it up with the tensors of H
    # Only keep top row of the first and last column of the last MPO tensor

    # allocate output
    output = Vector(repeat(copy(parent(mpo)), L ÷ length(mpo)))
    output[1] = output[1][1, :, :, :]
    output[end] = output[end][:, :, :, end]

    return FiniteMPOHamiltonian(output)
end
