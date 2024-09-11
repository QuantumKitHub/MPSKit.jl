"""
    FiniteMPO(Os::Vector{<:MPOTensor}) -> FiniteMPO
    FiniteMPO(O::AbstractTensorMap{S,N,N}) where {S,N} -> FiniteMPO

Matrix Product Operator (MPO) acting on a finite tensor product space with a linear order.
"""
struct FiniteMPO{O<:MPOTensor} <: AbstractMPO{O}
    data::Vector{O}
    function FiniteMPO{O}(::UndefInitializer, dims) where {O<:MPOTensor}
        return FiniteMPO{O}(Vector{O}(undef, dims))
    end
    function FiniteMPO{O}(Os::Vector{O}) where {O<:MPOTensor}
        return new{O}(Os)
    end
end
function FiniteMPO(Os::Vector{O}) where {O<:MPOTensor}
    for i in eachindex(Os)[1:(end - 1)]
        dual(right_virtualspace(Os[i])) == left_virtualspace(Os[i + 1]) ||
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
struct InfiniteMPO{O<:MPOTensor} <: AbstractMPO{O}
    data::PeriodicVector{O}
    function InfiniteMPO{O}(::UndefInitializer, dims) where {O<:MPOTensor}
        return InfiniteMPO{O}(PeriodicVector{O}(undef, dims))
    end
    function InfiniteMPO{O}(Os::PeriodicVector{O}) where {O<:MPOTensor}
        return new{O}(Os)
    end
end
function InfiniteMPO(Os::PeriodicVector{O}) where {O<:MPOTensor}
    for i in eachindex(Os)
        dual(right_virtualspace(Os[i])) == left_virtualspace(Os[1]) ||
            throw(SpaceMismatch("umatching virtual spaces at site $i"))
    end
    return InfiniteMPO{O}(Os)
end
InfiniteMPO(Os::AbstractVector{<:MPOTensor}) = InfiniteMPO(PeriodicVector(Os))
function InfiniteMPO(O::AbstractTensorMap{T,S,N,N}) where {T,S,N}
    return InfiniteMPO(decompose_localmpo(add_util_leg(O)))
end

const InfOrFinMPO{O} = Union{FiniteMPO{O},InfiniteMPO{O}}
const DenseMPO{O<:TensorMap} = InfOrFinMPO{O}

function DenseMPO(mpo::FiniteMPO)
    return FiniteMPO(map(TensorMap, mpo))
end
function DenseMPO(mpo::InfiniteMPO)
    return InfiniteMPO(map(TensorMap, mpo))
end

# Utility
# -------
Base.parent(mpo::InfOrFinMPO) = mpo.data
Base.copy(mpo::FiniteMPO) = FiniteMPO(map(copy, mpo))
Base.copy(mpo::InfiniteMPO) = InfiniteMPO(map(copy, mpo))

function Base.similar(::FiniteMPO, ::Type{O}, L::Int) where {O<:MPOTensor}
    return FiniteMPO{O}(undef, L)
end
function Base.similar(::InfiniteMPO, ::Type{O}, L::Int) where {O<:MPOTensor}
    return InfiniteMPO{O}(undef, L)
end

Base.repeat(mpo::FiniteMPO, n::Int) = FiniteMPO(repeat(parent(mpo), n))
Base.repeat(mpo::InfiniteMPO, n::Int) = InfiniteMPO(repeat(parent(mpo), n))

function remove_orphans!(mpo::SparseMPO; tol=eps(real(scalartype(mpo)))^(3 / 4))
    # drop zeros
    for slice in parent(mpo)
        for (k, v) in nonzero_pairs(slice)
            norm(v) < tol && delete!(slice, k)
        end
    end

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

# Converters
# ----------
function Base.convert(::Type{<:FiniteMPS}, mpo::FiniteMPO)
    return FiniteMPS(map(parent(mpo)) do O
                         @plansor A[-1 -2 -3; -4] := O[-1 -2; 1 2] * τ[1 2; -4 -3]
                     end)
end
function Base.convert(::Type{<:FiniteMPO}, mps::FiniteMPS)
    mpo_tensors = map([mps.AC[1]; mps.AR[2:end]]) do A
        @plansor O[-1 -2; -3 -4] := A[-1 -2 1; 2] * τ[-3 2; -4 1]
    end
    return FiniteMPO(mpo_tensors)
end
function Base.convert(::Type{TensorMap}, mpo::FiniteMPO)
    N = length(mpo)
    # add trivial tensors to remove left and right trivial leg.
    V_left = left_virtualspace(mpo, 1)
    @assert V_left == oneunit(V_left)
    U_left = ones(scalartype(mpo), V_left)'

    V_right = right_virtualspace(mpo, length(mpo))
    @assert V_right == oneunit(V_right)'
    U_right = ones(scalartype(mpo), V_right')

    tensors = vcat(U_left, parent(mpo), U_right)
    indices = [[i, -i, -(i + N), i + 1] for i in 1:length(mpo)]
    pushfirst!(indices, [1])
    push!(indices, [N + 1])
    O = ncon(tensors, indices)

    return transpose(O, (ntuple(identity, N), ntuple(i -> i + N, N)))
end

# Linear Algebra
# --------------
VectorInterface.scalartype(::Type{FiniteMPO{O}}) where {O} = scalartype(O)

Base.:+(mpo::FiniteMPO) = FiniteMPO(map(+, mpo))
function Base.:+(mpo1::FiniteMPO{TO}, mpo2::FiniteMPO{TO}) where {TO}
    (N = length(mpo1)) == length(mpo2) || throw(ArgumentError("dimension mismatch"))
    @assert left_virtualspace(mpo1, 1) == left_virtualspace(mpo2, 1) &&
            right_virtualspace(mpo1, N) == right_virtualspace(mpo2, N)

    mpo = similar(parent(mpo1))
    halfN = N ÷ 2
    A = storagetype(TO)

    # left half
    F₁ = isometry(A, (right_virtualspace(mpo1, 1) ⊕ right_virtualspace(mpo2, 1))',
                  right_virtualspace(mpo1, 1)')
    F₂ = leftnull(F₁)
    @assert _lastspace(F₂) == right_virtualspace(mpo2, 1)

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
        F₁ = isometry(A, (right_virtualspace(mpo1, i) ⊕ right_virtualspace(mpo2, i))',
                      right_virtualspace(mpo1, i)')
        F₂ = leftnull(F₁)
        @assert _lastspace(F₂) == right_virtualspace(mpo2, i)
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

# TODO: replace `copy` with `+` once this is defined for tensormaps
function Base.:-(mpo::FiniteMPO)
    return FiniteMPO(map(i -> i == 1 ? -mpo[i] : copy(mpo[i]), 1:length(mpo)))
end
Base.:-(mpo₁::FiniteMPO, mpo₂::FiniteMPO) = +(mpo₁, -mpo₂)

function Base.:*(mpo::FiniteMPO, α::Number)
    return FiniteMPO(map(i -> i == 1 ? α * mpo[i] : copy(mpo[i]), 1:length(mpo)))
end
Base.:*(α::Number, mpo::FiniteMPO) = mpo * α

function Base.:*(mpo1::FiniteMPO{TO}, mpo2::FiniteMPO{TO}) where {TO}
    (N = length(mpo1)) == length(mpo2) || throw(ArgumentError("dimension mismatch"))
    S = spacetype(TO)
    if (left_virtualspace(mpo1, 1) != oneunit(S) ||
        left_virtualspace(mpo2, 1) != oneunit(S)) ||
       (right_virtualspace(mpo1, N)' != oneunit(S) ||
        right_virtualspace(mpo2, N)' != oneunit(S))
        @warn "left/right virtual space is not trivial, fusion may not be unique"
        # this is a warning because technically any isomorphism that fuses the left/right
        # would work and for now I dont feel like figuring out if this is important
    end

    O = similar(parent(mpo1))
    A = storagetype(TO)

    # note order of mpos: mpo1 * mpo2 * state -> mpo2 on top of mpo1
    local Fᵣ # trick to make Fᵣ defined in the loop
    for i in 1:N
        Fₗ = i != 1 ? Fᵣ :
             isomorphism(A, fuse(left_virtualspace(mpo2, i), left_virtualspace(mpo1, i)),
                         left_virtualspace(mpo2, i) * left_virtualspace(mpo1, i))
        Fᵣ = isomorphism(A, fuse(right_virtualspace(mpo2, i), right_virtualspace(mpo1, i)),
                         right_virtualspace(mpo2, i)' * right_virtualspace(mpo1, i)')

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
        Fₗ = i != 1 ? Fᵣ :
             isomorphism(TT, fuse(left_virtualspace(A[i]), left_virtualspace(mpo, i)),
                         left_virtualspace(A[i]) * left_virtualspace(mpo, i))
        Fᵣ = isomorphism(TT, fuse(right_virtualspace(A[i]), right_virtualspace(mpo, i)),
                         right_virtualspace(A[i])' * right_virtualspace(mpo, i)')
        A[i] = _fuse_mpo_mps(mpo[i], A[i], Fₗ, Fᵣ)
    end

    return changebonds!(FiniteMPS(A), SvdCut(; trscheme=notrunc()); normalize=false)
end

function Base.:*(mpo::InfiniteMPO, st::InfiniteMPS)
    length(st) == length(mpo) || throw(ArgumentError("dimension mismatch"))
    T = promote_type(scalartype(mpo), scalartype(st))
    fusers = PeriodicArray(map(zip(st.AL, mpo)) do (al, mp)
                               return fuser(T, _firstspace(al), _firstspace(mp))
                           end)
    As = map(1:length(st)) do i
        return _fuse_mpo_mps(mpo[i], st.AL[i], fusers[i], fusers[i + 1])
    end

    return changebonds(InfiniteMPS(As), SvdCut(; trscheme=notrunc()))
end

function _fuse_mpo_mps(O::MPOTensor, A::MPSTensor, Fₗ, Fᵣ)
    @plansor A′[-1 -2; -3] := Fₗ[-1; 1 3] *
                              A[1 2; 4] *
                              O[3 -2; 2 5] *
                              conj(Fᵣ[-3; 4 5])
    return A′ isa AbstractBlockTensorMap ? only(A′) : A′
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
                         left_virtualspace(bra, 0) ⊗ left_virtualspace(mpo, 1)',
                         left_virtualspace(ket, 0))
    T_left = TransferMatrix(ket.AL[1:Nhalf], mpo[1:Nhalf], bra.AL[1:Nhalf])
    ρ_left = ρ_left * T_left

    # right half
    ρ_right = isomorphism(storagetype(T),
                          right_virtualspace(ket, N) ⊗ right_virtualspace(mpo, N)',
                          right_virtualspace(ket, length(ket)))
    T_right = TransferMatrix(ket.AR[(Nhalf + 1):end], mpo[(Nhalf + 1):end],
                             bra.AR[(Nhalf + 1):end])
    ρ_right = T_right * ρ_right

    # center
    return @plansor ρ_left[3 4; 1] * ket.CR[Nhalf][1; 5] * ρ_right[5 4; 2] *
                    conj(ket.CR[Nhalf][3; 2])
end
function TensorKit.dot(bra::InfiniteMPS, mpo::InfiniteMPO, ket::InfiniteMPS;
                       ishermitian=false, krylovdim=30, kwargs...)
    ρ₀ = similar(bra.AL[1],
                 left_virtualspace(bra, 1) * left_virtualspace(mpo, 1) ←
                 left_virtualspace(ket, 1))
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

function Base.isapprox(mpo₁::FiniteMPO, mpo₂::FiniteMPO;
                       atol::Real=0, rtol::Real=atol > 0 ? 0 : √eps(real(scalartype(mpo₁))))
    length(mpo₁) == length(mpo₂) || throw(ArgumentError("dimension mismatch"))
    # computing ||mpo₁ - mpo₂|| without constructing mpo₁ - mpo₂
    # ||mpo₁ - mpo₂||² = ||mpo₁||² + ||mpo₂||² - 2 ⟨mpo₁, mpo₂⟩
    norm₁² = abs(dot(mpo₁, mpo₁))
    norm₂² = abs(dot(mpo₂, mpo₂))
    norm₁₂² = norm₁² + norm₂² - 2 * real(dot(mpo₁, mpo₂))

    # don't take square roots to avoid precision loss
    return norm₁₂² ≤ max(atol^2, rtol^2 * max(norm₁², norm₂²))
end

#==========================================================================================#

"
    Represents a dense periodic mpo
"
# struct DenseMPO{O<:MPOTensor}
#     opp::PeriodicArray{O,1}
# end

# DenseMPO(t::AbstractTensorMap) = DenseMPO(fill(t, 1));
# DenseMPO(t::AbstractArray{T,1}) where {T<:MPOTensor} = DenseMPO(PeriodicArray(t));
# Base.length(t::DenseMPO) = length(t.opp)
# Base.size(t::DenseMPO) = (length(t),)
# Base.repeat(t::DenseMPO, n) = DenseMPO(repeat(t.opp, n));
# Base.getindex(t::DenseMPO, i) = getindex(t.opp, i);
# Base.eltype(::DenseMPO{O}) where {O} = O
# VectorInterface.scalartype(::DenseMPO{O}) where {O} = scalartype(O)
# Base.iterate(t::DenseMPO, i=1) = (i > length(t.opp)) ? nothing : (t[i], i + 1);
# TensorKit.space(t::DenseMPO, i) = space(t.opp[i], 2)
# function Base.convert(::Type{InfiniteMPS}, mpo::DenseMPO)
#     return InfiniteMPS(map(mpo.opp) do t
#                            @plansor tt[-1 -2 -3; -4] := t[-1 -2; 1 2] * τ[1 2; -4 -3]
#                        end)
# end
#
# function Base.convert(::Type{DenseMPO}, mps::InfiniteMPS)
#     return DenseMPO(map(mps.AL) do t
#                         @plansor tt[-1 -2; -3 -4] := t[-1 -2 1; 2] * τ[-3 2; -4 1]
#                     end)
# end
#
# #naively apply the mpo to the mps
# function Base.:*(mpo::DenseMPO, st::InfiniteMPS)
#     length(st) == length(mpo) || throw(ArgumentError("dimension mismatch"))
#
#     fusers = PeriodicArray(map(zip(st.AL, mpo)) do (al, mp)
#                                return isometry(fuse(_firstspace(al), _firstspace(mp)),
#                                                _firstspace(al) * _firstspace(mp))
#                            end)
#
#     return InfiniteMPS(map(1:length(st)) do i
#                            @plansor t[-1 -2; -3] := st.AL[i][1 2; 3] *
#                                                     mpo[i][4 -2; 2 5] *
#                                                     fusers[i][-1; 1 4] *
#                                                     conj(fusers[i + 1][-3; 3 5])
#                        end)
# end
# function Base.:*(mpo::DenseMPO, st::FiniteMPS)
#     mod(length(mpo), length(st)) == 0 || throw(ArgumentError("dimension mismatch"))
#
#     tensors = [st.AC[1]; st.AR[2:end]]
#     mpot = mpo[1:length(st)]
#
#     fusers = map(zip(tensors, mpot)) do (al, mp)
#         return isometry(fuse(_firstspace(al), _firstspace(mp)),
#                         _firstspace(al) * _firstspace(mp))
#     end
#
#     push!(fusers,
#           isometry(fuse(_lastspace(tensors[end])', _lastspace(mpot[end])'),
#                    _lastspace(tensors[end])' * _lastspace(mpot[end])'))
#
#     (_firstspace(mpot[1]) == oneunit(_firstspace(mpot[1])) &&
#      _lastspace(mpot[end])' == _firstspace(mpot[1])) ||
#         @warn "mpo does not start/end with a trivial leg"
#
#     return FiniteMPS(map(1:length(st)) do i
#                          @plansor t[-1 -2; -3] := tensors[i][1 2; 3] *
#                                                   mpot[i][4 -2; 2 5] *
#                                                   fusers[i][-1; 1 4] *
#                                                   conj(fusers[i + 1][-3; 3 5])
#                      end)
# end
#
# function Base.:*(mpo1::DenseMPO, mpo2::DenseMPO)
#     length(mpo1) == length(mpo2) || throw(ArgumentError("dimension mismatch"))
#
#     fusers = PeriodicArray(map(zip(mpo2.opp, mpo1.opp)) do (mp1, mp2)
#                                return isometry(fuse(_firstspace(mp1), _firstspace(mp2)),
#                                                _firstspace(mp1) * _firstspace(mp2))
#                            end)
#
#     return DenseMPO(map(1:length(mpo1)) do i
#                         @plansor t[-1 -2; -3 -4] := mpo2[i][1 2; -3 3] *
#                                                     mpo1[i][4 -2; 2 5] *
#                                                     fusers[i][-1; 1 4] *
#                                                     conj(fusers[i + 1][-4; 3 5])
#                     end)
# end
#
# function TensorKit.dot(a::InfiniteMPS, mpo::DenseMPO, b::InfiniteMPS; krylovdim=30)
#     init = similar(a.AL[1],
#                    _firstspace(b.AL[1]) * _firstspace(mpo.opp[1]) ← _firstspace(a.AL[1]))
#     randomize!(init)
#
#     val, = fixedpoint(TransferMatrix(b.AL, mpo.opp, a.AL), init, :LM,
#                       Arnoldi(; krylovdim=krylovdim))
#     return val
# end
