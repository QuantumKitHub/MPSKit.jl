"
    Represents a dense periodic mpo
"
struct DenseMPO{O<:MPOTensor}
    opp::PeriodicArray{O,1}
end

DenseMPO(t::AbstractTensorMap) = DenseMPO(fill(t, 1));
DenseMPO(t::AbstractArray{T,1}) where {T<:MPOTensor} = DenseMPO(PeriodicArray(t));
Base.length(t::DenseMPO) = length(t.opp);
Base.size(t::DenseMPO) = (length(t),)
Base.repeat(t::DenseMPO, n) = DenseMPO(repeat(t.opp, n));
Base.getindex(t::DenseMPO, i) = getindex(t.opp, i);
Base.eltype(::DenseMPO{O}) where {O} = O
VectorInterface.scalartype(::DenseMPO{O}) where {O} = scalartype(O)
Base.iterate(t::DenseMPO, i=1) = (i > length(t.opp)) ? nothing : (t[i], i + 1);
TensorKit.space(t::DenseMPO, i) = space(t.opp[i], 2)
function Base.convert(::Type{InfiniteMPS}, mpo::DenseMPO)
    return InfiniteMPS(map(mpo.opp) do t
                           @plansor tt[-1 -2 -3; -4] := t[-1 -2; 1 2] * τ[1 2; -4 -3]
                       end)
end

function Base.convert(::Type{DenseMPO}, mps::InfiniteMPS)
    return DenseMPO(map(mps.AL) do t
                        @plansor tt[-1 -2; -3 -4] := t[-1 -2 1; 2] * τ[-3 2; -4 1]
                    end)
end

#naively apply the mpo to the mps
function Base.:*(mpo::DenseMPO, st::InfiniteMPS)
    length(st) == length(mpo) || throw(ArgumentError("dimension mismatch"))

    fusers = PeriodicArray(map(zip(st.AL, mpo)) do (al, mp)
                               return isometry(fuse(_firstspace(al), _firstspace(mp)),
                                               _firstspace(al) * _firstspace(mp))
                           end)

    return InfiniteMPS(map(1:length(st)) do i
                           @plansor t[-1 -2; -3] := st.AL[i][1 2; 3] *
                                                    mpo[i][4 -2; 2 5] *
                                                    fusers[i][-1; 1 4] *
                                                    conj(fusers[i + 1][-3; 3 5])
                       end)
end
function Base.:*(mpo::DenseMPO, st::FiniteMPS)
    mod(length(mpo), length(st)) == 0 || throw(ArgumentError("dimension mismatch"))

    tensors = [st.AC[1]; st.AR[2:end]]
    mpot = mpo[1:length(st)]

    fusers = map(zip(tensors, mpot)) do (al, mp)
        return isometry(fuse(_firstspace(al), _firstspace(mp)),
                        _firstspace(al) * _firstspace(mp))
    end

    push!(fusers,
          isometry(fuse(_lastspace(tensors[end])', _lastspace(mpot[end])'),
                   _lastspace(tensors[end])' * _lastspace(mpot[end])'))

    (_firstspace(mpot[1]) == oneunit(_firstspace(mpot[1])) &&
     _lastspace(mpot[end])' == _firstspace(mpot[1])) ||
        @warn "mpo does not start/end with a trivial leg"

    return FiniteMPS(map(1:length(st)) do i
                         @plansor t[-1 -2; -3] := tensors[i][1 2; 3] *
                                                  mpot[i][4 -2; 2 5] *
                                                  fusers[i][-1; 1 4] *
                                                  conj(fusers[i + 1][-3; 3 5])
                     end)
end

function Base.:*(mpo1::DenseMPO, mpo2::DenseMPO)
    length(mpo1) == length(mpo2) || throw(ArgumentError("dimension mismatch"))

    fusers = PeriodicArray(map(zip(mpo2.opp, mpo1.opp)) do (mp1, mp2)
                               return isometry(fuse(_firstspace(mp1), _firstspace(mp2)),
                                               _firstspace(mp1) * _firstspace(mp2))
                           end)

    return DenseMPO(map(1:length(mpo1)) do i
                        @plansor t[-1 -2; -3 -4] := mpo2[i][1 2; -3 3] *
                                                    mpo1[i][4 -2; 2 5] *
                                                    fusers[i][-1; 1 4] *
                                                    conj(fusers[i + 1][-4; 3 5])
                    end)
end

function TensorKit.dot(a::InfiniteMPS, mpo::DenseMPO, b::InfiniteMPS; krylovdim=30)
    init = similar(a.AL[1],
                   _firstspace(b.AL[1]) * _firstspace(mpo.opp[1]) ← _firstspace(a.AL[1]))
    randomize!(init)

    (vals, vecs, convhist) = eigsolve(TransferMatrix(b.AL, mpo.opp, a.AL), init, 1, :LM,
                                      Arnoldi(; krylovdim=krylovdim))
    convhist.converged == 0 && @info "dot mps not converged"
    return vals[1]
end
