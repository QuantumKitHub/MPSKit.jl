"
It is possible to have matrix product (operators / states) that are also periodic in the vertical direction
For examples, as fix points of statmech problems
These should be represented as respectively MultiLine{<:InfiniteMPO} / Multiline{<:InfiniteMPS}
"
struct Multiline{T}
    data::PeriodicArray{T,1}
end

Base.length(t::Multiline) = prod(size(t));
Base.size(t::Multiline) = (length(t.data),length(t.data[1]));
Base.size(t::Multiline,i) = size(t)[i];
Base.getindex(t::Multiline,i) = t.data[i];
Base.copy(t::Multiline) = Multiline(map(copy,t.data));
Multiline(t::AbstractArray) = Multiline(PeriodicArray(t));

Base.convert(::Vector,t::Multiline) = t.data.data;
Base.convert(::PeriodicArray,t::Multiline) = t.data;

Base.convert(::Multiline,v::AbstractArray) = Multiline(v);

#--- implementation of MPSMultiline
const MPSMultiline = Multiline{<:InfiniteMPS}

function MPSMultiline(pspaces::AbstractArray{S,2},Dspaces::AbstractArray{S,2};kwargs...) where S
    MPSMultiline(map(zip(circshift(Dspaces,(0,-1)),pspaces,Dspaces)) do (D1,p,D2)
        TensorMap(rand,Defaults.eltype,D1*p,D2)
    end; kwargs...)
end

function MPSMultiline(data::AbstractArray{T,2};kwargs...) where T<:GenericMPSTensor
    Multiline(PeriodicArray(map(1:size(data,1)) do row
        InfiniteMPS(data[row,:];kwargs...)
    end))
end
function MPSMultiline(data::AbstractArray{T,2},Cinit::AbstractArray{O,1};kwargs...) where {T<:GenericMPSTensor,O<:MPSBondTensor}
    Multiline(PeriodicArray(map(1:size(data,1)) do row
        InfiniteMPS(data[row,:],Cinit[row];kwargs...);
    end))
end
function Base.getproperty(psi::MPSMultiline,prop::Symbol)
    if prop == :AL
        return ALView(psi)
    elseif prop == :AR
        return ARView(psi)
    elseif prop == :AC
        return ACView(psi)
    elseif prop == :CR
        return CRView(psi)
    else
        return getfield(psi,prop)
    end
end

for f in (:l_RR, :l_RL, :l_LL, :l_LR)
    @eval $f(t::MPSMultiline,i,j = 1) = $f(t[i],j)
end

for f in (:r_RR, :r_RL, :r_LR,:r_LL)
    @eval $f(t::MPSMultiline,i,j = size(t,2)) = $f(t[i],j)
end


site_type(::Type{Multiline{S}}) where S = site_type(S);
bond_type(::Type{Multiline{S}}) where S = bond_type(S);

TensorKit.dot(a::MPSMultiline,b::MPSMultiline;kwargs...) = sum(dot.(a.data,b.data;kwargs...))

Base.convert(::Type{MPSMultiline},st::InfiniteMPS) = Multiline([st]);
Base.convert(::Type{InfiniteMPS},st::MPSMultiline) = st[1];
Base.eltype(t::MPSMultiline) = eltype(t[1]);
virtualspace(t::MPSMultiline,i::Int,j::Int) = virtualspace(t[i],j);
