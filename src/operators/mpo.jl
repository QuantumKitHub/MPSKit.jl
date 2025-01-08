"""
    struct MPO{O<:MPOTensor,V<:AbstractVector{O}} <: AbstractMPO{O}

Matrix Product Operator (MPO) acting on a tensor product space with a linear order.

See also: [`FiniteMPO`](@ref), [`InfiniteMPO`](@ref)
"""
struct MPO{TO<:MPOTensor,V<:AbstractVector{TO}} <: AbstractMPO{TO}
    O::V
end

"""
    FiniteMPO(Os::Vector{<:MPOTensor}) -> FiniteMPO
    FiniteMPO(O::AbstractTensorMap{S,N,N}) where {S,N} -> FiniteMPO

Matrix Product Operator (MPO) acting on a finite tensor product space with a linear order.
"""
const FiniteMPO{O<:MPOTensor} = MPO{O,Vector{O}}

function FiniteMPO(Os::AbstractVector{O}) where {O<:MPOTensor}
    for i in eachindex(Os)[1:(end - 1)]
        right_virtualspace(Os[i]) == left_virtualspace(Os[i + 1]) ||
            throw(SpaceMismatch("unmatching virtual spaces at site $i"))
    end
    return FiniteMPO{O}(Os)
end

function FiniteMPO(O::AbstractTensorMap{T,S,N,N}) where {T,S,N}
    return FiniteMPO(decompose_localmpo(add_util_leg(O)))
end

"""
    InfiniteMPO(Os::PeriodicVector{<:MPOTensor}) -> InfiniteMPO

Matrix Product Operator (MPO) acting on an infinite tensor product space with a linear order.
"""
const InfiniteMPO{O<:MPOTensor} = MPO{O,PeriodicVector{O}}

function InfiniteMPO(Os::AbstractVector{O}) where {O<:MPOTensor}
    for i in eachindex(Os)
        right_virtualspace(Os[i]) == left_virtualspace(Os[mod1(i + 1, end)]) ||
            throw(SpaceMismatch("umatching virtual spaces at site $i"))
    end
    return InfiniteMPO{O}(Os)
end

const DenseMPO = MPO{<:TensorMap}

DenseMPO(mpo::MPO) = mpo isa DenseMPO ? copy(mpo) : MPO(map(TensorMap, parent(mpo)))

# Utility
# -------
Base.parent(mpo::MPO) = mpo.O
Base.copy(mpo::MPO) = MPO(map(copy, mpo))

function Base.similar(mpo::MPO, ::Type{O}, L::Int) where {O<:MPOTensor}
    return MPO(similar(parent(mpo), O, L))
end

Base.repeat(mpo::MPO, n::Int) = MPO(repeat(parent(mpo), n))

function remove_orphans!(mpo::InfiniteMPO; tol=eps(real(scalartype(mpo)))^(3 / 4))
    droptol!.(mpo, tol)

    # drop dead starts/ends
    changed = true
    while changed
        changed = false
        for i in 1:length(mpo)
            # slice empty columns on right or empty rows on left
            mask = filter(1:size(mpo[i], 4)) do j
                return j ∈ getindex.(nonzero_keys(mpo[i]), 1) ||
                       j ∈ getindex.(nonzero_keys(mpo[i + 1]), 4)
            end
            changed |= length(mask) == size(mpo[i], 4)
            mpo[i] = mpo[i][:, :, :, mask]
            mpo[i + 1] = mpo[i + 1][mask, :, :, :]
        end
    end

    return mpo
end

function remove_orphans!(mpo::FiniteMPO; tol=eps(real(scalartype(mpo)))^(3 / 4))
    droptol!.(mpo, tol)

    # Forward sweep
    # col j on site i empty -> remove row j on site i + 1
    for i in 1:(length(mpo) - 1)
        mask = filter(1:size(mpo[i], 4)) do j
            return j ∈ getindex.(nonzero_keys(mpo[i]), 4)
        end
        mpo[i] = mpo[i][:, :, :, mask]
        mpo[i + 1] = mpo[i + 1][mask, :, :, :]
    end

    # Backward sweep
    # row j on site i empty -> remove col j on site i - 1
    for i in length(mpo):-1:2
        mask = filter(1:size(mpo[i], 1)) do j
            return j ∈ getindex.(nonzero_keys(mpo[i]), 1)
        end
        mpo[i] = mpo[i][mask, :, :, :]
        mpo[i - 1] = mpo[i - 1][:, :, :, mask]
    end

    return mpo
end

# Converters
# ----------
function Base.convert(::Type{<:FiniteMPS}, mpo::FiniteMPO)
    return FiniteMPS(map(_mpo_to_mps, parent(mpo)))
end
function Base.convert(::Type{<:InfiniteMPS}, mpo::InfiniteMPO)
    return InfiniteMPS(map(_mpo_to_mps, parent(mpo)))
end
function _mpo_to_mps(O::MPOTensor)
    @plansor A[-1 -2 -3; -4] := O[-1 -2; 1 2] * τ[1 2; -4 -3]
    return A isa AbstractBlockTensorMap ? TensorMap(A) : A
end

function Base.convert(::Type{<:FiniteMPO}, mps::FiniteMPS)
    return FiniteMPO(map(_mps_to_mpo, [mps.AC[1]; mps.AR[2:end]]))
end
function Base.convert(::Type{<:InfiniteMPO}, mps::InfiniteMPS)
    return InfiniteMPO(map(_mps_to_mpo, mps.AL))
end
function _mps_to_mpo(A::GenericMPSTensor{S,3}) where {S}
    @plansor O[-1 -2; -3 -4] := A[-1 -2 1; 2] * τ[-3 2; -4 1]
    return O
end

function Base.convert(::Type{TensorMap}, mpo::FiniteMPO)
    N = length(mpo)
    # add trivial tensors to remove left and right trivial leg.
    V_left = left_virtualspace(mpo, 1)
    @assert V_left == oneunit(V_left)
    U_left = ones(scalartype(mpo), V_left)'

    V_right = right_virtualspace(mpo, length(mpo))
    @assert V_right == oneunit(V_right)
    U_right = ones(scalartype(mpo), V_right)

    tensors = vcat(U_left, parent(mpo), U_right)
    indices = [[i, -i, -(2N - i + 1), i + 1] for i in 1:length(mpo)]
    pushfirst!(indices, [1])
    push!(indices, [N + 1])
    O = ncon(tensors, indices)

    return repartition(O, N, N)
end

# Linear Algebra
# --------------
# VectorInterface.scalartype(::Type{FiniteMPO{O}}) where {O} = scalartype(O)

Base.:+(mpo::MPO) = MPO(map(+, parent(mpo)))
function Base.:+(mpo1::FiniteMPO{TO}, mpo2::FiniteMPO{TO}) where {TO}
    (N = length(mpo1)) == length(mpo2) || throw(ArgumentError("dimension mismatch"))
    @assert left_virtualspace(mpo1, 1) == left_virtualspace(mpo2, 1) &&
            right_virtualspace(mpo1, N) == right_virtualspace(mpo2, N)

    mpo = similar(parent(mpo1))
    halfN = N ÷ 2
    A = storagetype(TO)

    # left half
    F₁ = isometry(A, (right_virtualspace(mpo1, 1) ⊕ right_virtualspace(mpo2, 1)),
                  right_virtualspace(mpo1, 1))
    F₂ = leftnull(F₁)
    @assert _lastspace(F₂) == right_virtualspace(mpo2, 1)'

    @plansor O[-3 -1 -2; -4] := mpo1[1][-1 -2; -3 1] * conj(F₁[-4; 1]) +
                                mpo2[1][-1 -2; -3 1] * conj(F₂[-4; 1])

    # making sure that the new operator is "full rank"
    O, R = leftorth!(O)
    mpo[1] = transpose(O, ((2, 3), (1, 4)))

    for i in 2:halfN
        # incorporate fusers from left side
        @plansor O₁[-1 -2; -3 -4] := R[-1; 1] * F₁[1; 2] * mpo1[i][2 -2; -3 -4]
        @plansor O₂[-1 -2; -3 -4] := R[-1; 1] * F₂[1; 2] * mpo2[i][2 -2; -3 -4]

        # incorporate fusers from right side
        F₁ = isometry(A, (right_virtualspace(mpo1, i) ⊕ right_virtualspace(mpo2, i)),
                      right_virtualspace(mpo1, i))
        F₂ = leftnull(F₁)
        @assert _lastspace(F₂) == right_virtualspace(mpo2, i)'
        @plansor O[-3 -1 -2; -4] := O₁[-1 -2; -3 1] * conj(F₁[-4; 1]) +
                                    O₂[-1 -2; -3 1] * conj(F₂[-4; 1])

        # making sure that the new operator is "full rank"
        O, R = leftorth!(O)
        mpo[i] = transpose(O, ((2, 3), (1, 4)))
    end

    C₁, C₂ = F₁, F₂

    # right half
    F₁ = isometry(A, left_virtualspace(mpo1, N) ⊕ left_virtualspace(mpo2, N),
                  left_virtualspace(mpo1, N))
    F₂ = leftnull(F₁)
    @assert _lastspace(F₂) == left_virtualspace(mpo2, N)'

    @plansor O[-1; -3 -4 -2] := F₁[-1; 1] * mpo1[N][1 -2; -3 -4] +
                                F₂[-1; 1] * mpo2[N][1 -2; -3 -4]

    # making sure that the new operator is "full rank"
    L, O = rightorth!(O)
    mpo[end] = transpose(O, ((1, 4), (2, 3)))

    for i in (N - 1):-1:(halfN + 1)
        # incorporate fusers from right side
        @plansor O₁[-1 -2; -3 -4] := mpo1[i][-1 -2; -3 2] * conj(F₁[1; 2]) * L[1; -4]
        @plansor O₂[-1 -2; -3 -4] := mpo2[i][-1 -2; -3 2] * conj(F₂[1; 2]) * L[1; -4]

        # incorporate fusers from left side
        F₁ = isometry(A, left_virtualspace(mpo1, i) ⊕ left_virtualspace(mpo2, i),
                      left_virtualspace(mpo1, i))
        F₂ = leftnull(F₁)
        @assert _lastspace(F₂) == left_virtualspace(mpo2, i)'
        @plansor O[-1; -3 -4 -2] := F₁[-1; 1] * O₁[1 -2; -3 -4] +
                                    F₂[-1; 1] * O₂[1 -2; -3 -4]

        # making sure that the new operator is "full rank"
        L, O = rightorth!(O)
        mpo[i] = transpose(O, ((1, 4), (2, 3)))
    end

    # create center gauge and absorb to the right
    C₁ = C₁ * F₁'
    C₂ = C₂ * F₂'
    C = R * (C₁ + C₂) * L
    @plansor mpo[halfN + 1][-1 -2; -3 -4] := mpo[halfN + 1][1 -2; -3 -4] * C[-1; 1]

    return FiniteMPO(mpo)
end

function VectorInterface.scale!(mpo::MPO, α::Number)
    scale!(first(mpo), α)
    return mpo
end

function Base.:*(mpo1::FiniteMPO{TO}, mpo2::FiniteMPO{TO}) where {TO}
    (N = length(mpo1)) == length(mpo2) || throw(ArgumentError("dimension mismatch"))
    S = spacetype(TO)
    if (left_virtualspace(mpo1, 1) != oneunit(S) ||
        left_virtualspace(mpo2, 1) != oneunit(S)) ||
       (right_virtualspace(mpo1, N) != oneunit(S) ||
        right_virtualspace(mpo2, N) != oneunit(S))
        @warn "left/right virtual space is not trivial, fusion may not be unique"
        # this is a warning because technically any isomorphism that fuses the left/right
        # would work and for now I dont feel like figuring out if this is important
    end

    O = similar(parent(mpo1))
    A = storagetype(TO)

    # note order of mpos: mpo1 * mpo2 * state -> mpo2 on top of mpo1
    local Fᵣ # trick to make Fᵣ defined in the loop
    for i in 1:N
        Fₗ = i != 1 ? Fᵣ : fuser(A, left_virtualspace(mpo2, i), left_virtualspace(mpo1, i))
        Fᵣ = fuser(A, right_virtualspace(mpo2, i), right_virtualspace(mpo1, i))
        @plansor O[i][-1 -2; -3 -4] := Fₗ[-1; 1 4] * mpo2[i][1 2; -3 3] *
                                       mpo1[i][4 -2; 2 5] *
                                       conj(Fᵣ[-4; 3 5])
    end

    return changebonds!(FiniteMPO(O), SvdCut(; trscheme=notrunc()))
end

function Base.:*(mpo::FiniteMPO, mps::FiniteMPS)
    length(mpo) == length(mps) || throw(ArgumentError("dimension mismatch"))

    A = [mps.AC[1]; mps.AR[2:end]]
    TT = storagetype(eltype(A))

    local Fᵣ # trick to make Fᵣ defined in the loop
    for i in 1:length(mps)
        Fₗ = i != 1 ? Fᵣ : fuser(TT, left_virtualspace(mps, i), left_virtualspace(mpo, i))
        Fᵣ = fuser(TT, right_virtualspace(mps, i), right_virtualspace(mpo, i))
        A[i] = _fuse_mpo_mps(mpo[i], A[i], Fₗ, Fᵣ)
    end

    return changebonds!(FiniteMPS(A),
                        SvdCut(; trscheme=truncbelow(eps(real(scalartype(TT)))));
                        normalize=false)
end

function Base.:*(mpo::InfiniteMPO, mps::InfiniteMPS)
    L = check_length(mpo, mps)
    T = promote_type(scalartype(mpo), scalartype(mps))
    fusers = PeriodicArray(fuser.(T, left_virtualspace.(Ref(mps), 1:L),
                                  left_virtualspace.(Ref(mpo), 1:L)))
    As = map(1:L) do i
        return _fuse_mpo_mps(mpo[i], mps.AL[i], fusers[i], fusers[i + 1])
    end
    return changebonds(InfiniteMPS(As), SvdCut(; trscheme=notrunc()))
end

function _fuse_mpo_mps(O::MPOTensor, A::MPSTensor, Fₗ, Fᵣ)
    @plansor A′[-1 -2; -3] := Fₗ[-1; 1 3] *
                              A[1 2; 4] *
                              O[3 -2; 2 5] *
                              conj(Fᵣ[-3; 4 5])
    return A′ isa AbstractBlockTensorMap ? TensorMap(A′) : A′
end

function Base.:*(mpo1::InfiniteMPO, mpo2::InfiniteMPO)
    L = check_length(mpo1, mpo2)

    T = promote_type(scalartype(mpo1), scalartype(mpo2))
    make_fuser(i) = fuser(T, left_virtualspace(mpo2, i), left_virtualspace(mpo1, i))
    fusers = PeriodicArray(map(make_fuser, 1:L))

    Os = map(1:L) do i
        return _fuse_mpo_mpo(mpo1[i], mpo2[i], fusers[i], fusers[i + 1])
    end
    return InfiniteMPO(Os)
end

function _fuse_mpo_mpo(O1::MPOTensor, O2::MPOTensor, Fₗ, Fᵣ)
    return @plansor O′[-1 -2; -3 -4] := Fₗ[-1; 1 4] *
                                        O2[1 2; -3 3] *
                                        O1[4 -2; 2 5] *
                                        conj(Fᵣ[-4; 3 5])
end

# TODO: I think the fastest order is to start from both ends, and take the overlap at the
# largest virtual space cut, but it might be better to just multithread both sides and meet
# in the middle
function TensorKit.dot(bra::FiniteMPS{T}, mpo::FiniteMPO, ket::FiniteMPS{T}) where {T}
    (N = length(bra)) == length(mpo) == length(ket) ||
        throw(ArgumentError("dimension mismatch"))
    Nhalf = N ÷ 2
    # left half
    ρ_left = isomorphism(storagetype(T),
                         left_virtualspace(bra, 1) ⊗ left_virtualspace(mpo, 1)',
                         left_virtualspace(ket, 1))
    T_left = TransferMatrix(ket.AL[1:Nhalf], mpo[1:Nhalf], bra.AL[1:Nhalf])
    ρ_left = ρ_left * T_left

    # right half
    ρ_right = isomorphism(storagetype(T),
                          right_virtualspace(ket, N) ⊗ right_virtualspace(mpo, N),
                          right_virtualspace(ket, length(ket)))
    T_right = TransferMatrix(ket.AR[(Nhalf + 1):end], mpo[(Nhalf + 1):end],
                             bra.AR[(Nhalf + 1):end])
    ρ_right = T_right * ρ_right

    # center
    return @plansor ρ_left[3 4; 1] * ket.C[Nhalf][1; 5] * ρ_right[5 4; 2] *
                    conj(ket.C[Nhalf][3; 2])
end
function TensorKit.dot(bra::InfiniteMPS, mpo::InfiniteMPO, ket::InfiniteMPS;
                       ishermitian=false, krylovdim=30, kwargs...)
    ρ₀ = similar(bra.AL[1],
                 left_virtualspace(ket, 1) * left_virtualspace(mpo, 1) ←
                 left_virtualspace(bra, 1))
    randomize!(ρ₀)

    val, = fixedpoint(TransferMatrix(ket.AL, parent(mpo), bra.AL), ρ₀, :LM; ishermitian,
                      krylovdim, kwargs...)
    return val
end

function TensorKit.dot(mpo₁::FiniteMPO, mpo₂::FiniteMPO)
    length(mpo₁) == length(mpo₂) || throw(ArgumentError("dimension mismatch"))
    N = length(mpo₁)
    Nhalf = N ÷ 2
    # left half
    @plansor ρ_left[-1; -2] := conj(mpo₁[1][1 2; 3 -1]) * mpo₂[1][1 2; 3 -2]
    for i in 2:Nhalf
        @plansor ρ_left[-1; -2] := ρ_left[1; 2] * conj(mpo₁[i][1 3; 4 -1]) *
                                   mpo₂[i][2 3; 4 -2]
    end
    # right half
    @plansor ρ_right[-1; -2] := conj(mpo₁[N][-2 1; 2 3]) * mpo₂[N][-1 1; 2 3]
    for i in (N - 1):-1:(Nhalf + 1)
        @plansor ρ_right[-1; -2] := ρ_right[1; 2] * conj(mpo₁[i][-2 4; 3 2]) *
                                    mpo₂[i][-1 4; 3 1]
    end
    return @plansor ρ_left[1; 2] * ρ_right[2; 1]
end

function Base.isapprox(mpo₁::MPO, mpo₂::MPO;
                       atol::Real=0, rtol::Real=atol > 0 ? 0 : √eps(real(scalartype(mpo₁))))
    check_length(mpo₁, mpo₂)
    # computing ||mpo₁ - mpo₂|| without constructing mpo₁ - mpo₂
    # ||mpo₁ - mpo₂||² = ||mpo₁||² + ||mpo₂||² - 2 ⟨mpo₁, mpo₂⟩
    norm₁² = abs(dot(mpo₁, mpo₁))
    norm₂² = abs(dot(mpo₂, mpo₂))
    norm₁₂² = norm₁² + norm₂² - 2 * real(dot(mpo₁, mpo₂))

    # don't take square roots to avoid precision loss
    return norm₁₂² ≤ max(atol^2, rtol^2 * max(norm₁², norm₂²))
end
