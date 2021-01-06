"
    Represents a periodic (in 2 directions) statmech mpo
"
struct PeriodicMPO{O<:MPOTensor} <: Operator
    opp::PeriodicArray{O,2}
end

PeriodicMPO(t::AbstractTensorMap) = PeriodicMPO(fill(t,1,1));
PeriodicMPO(t::AbstractArray{T,2}) where T<:MPOTensor = PeriodicMPO(PeriodicArray(t));

Base.repeat(t::PeriodicMPO,i,j) = PeriodicMPO(repeat(t.opp,i,j));
Base.getindex(o::PeriodicMPO,i,j) = o.opp[i,j]
Base.size(o::PeriodicMPO,i) = size(o.opp,i);
Base.size(o::PeriodicMPO) = size(o.opp);

#naively apply the mpo to the mps
Base.:*(mpo::PeriodicMPO,st::InfiniteMPS) = convert(InfiniteMPS,mpo*convert(MPSMultiline,st))

function Base.:*(mpo::PeriodicMPO,st::MPSMultiline{T}) where T<:MPSTensor #need 3leg tensors for this to make sense
    size(st) == size(mpo) || throw(ArgumentError("dimension mismatch"))

    fusers = PeriodicArray(map(zip(st.AL,mpo.opp)) do (al,mp)
        isometry(fuse(_firstspace(al),_firstspace(mp)),_firstspace(al)*_firstspace(mp))
    end)

    apl = copy(st.AL)

    for i in 1:size(st,1),
        j in 1:size(st,2)
        @tensor apl[i,j][-1 -2;-3] := apl[i,j][1,2,3]*mpo.opp[i,j][4,-2,5,2]*fusers[i,j][-1,1,4]*conj(fusers[i,j+1][-3,3,5])
    end

    MPSMultiline(apl);
end

function Base.:*(mpo1::PeriodicMPO,mpo2::PeriodicMPO)
    size(mpo1) == size(mpo2) || throw(ArgumentError("dimension mismatch"))

    fusers = PeriodicArray(map(zip(mpo1.opp,mpo2.opp)) do (mp1,mp2)
        isometry(fuse(_firstspace(mp1),_firstspace(mp2)),_firstspace(mp1)*_firstspace(mp2))
    end)

    apl = copy(mpo2.opp)

    for i in 1:size(mpo1,1),
        j in 1:size(mpo1,2)
        @tensor apl[i,j][-1 -2;-3 -4] := apl[i,j][1,2,3,-4]*mpo1.opp[i,j][4,-2,5,2]*fusers[i,j][-1,1,4]*conj(fusers[i,j+1][-3,3,5])
    end

    PeriodicMPO(apl)
end

function Base.convert(::Type{MPSMultiline},mpo::PeriodicMPO)
    MPSMultiline(map(mpo.opp) do t
        permute(t,(1,2,4),(3,))
    end)
end

function Base.convert(::Type{PeriodicMPO},mps::MPSMultiline{T}) where T<:GenericMPSTensor{S,3} where S
    PeriodicMPO(map(mps.AL) do t
        permute(t,(1,2),(4,3))
    end)
end

function Base.copyto!(dst::PeriodicMPO,src::MPSMultiline{T}) where T<:GenericMPSTensor{S,3} where S
    for i in 1:size(dst,1),
        j in 1:size(dst,2)
        dst.opp[i,j] = permute(src.AL[i,j],(1,2),(4,3))
    end
    dst
end
