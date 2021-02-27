#=
I used to define AL/ARview setters (and still can if needed), but the problem is that they're generically not really defined
our tensor decompositions are only unique when some conditions are satisfied
Therefore I've decided to remove them
=#
struct ALView{S}
    parent::S
end

function Base.getindex(v::ALView{S},i::Int)::eltype(S) where S <: Union{FiniteMPS,MPSComoving}
    !(v.parent isa MPSComoving) || i <= length(v.parent) || throw(ArgumentError("out of bounds"))
    ismissing(v.parent.ALs[i]) && v.parent.CR[i] # by getting CL[i+1], we are garantueeing that AL[i] exists
    return v.parent.ALs[i]
end

Base.getindex(v::ALView{<:MPSMultiline},i::Int,j::Int) = v.parent[i].AL[j]
Base.setindex!(v::ALView{<:MPSMultiline},vec,i::Int,j::Int) = setindex!(v.parent[i].AL,vec,j);

struct ARView{S}
    parent::S
end

function Base.getindex(v::ARView{S},i::Int)::eltype(S) where S <: Union{FiniteMPS,MPSComoving}
    !(v.parent isa MPSComoving) || i >=1 || throw(ArgumentError("out of bounds"))
    ismissing(v.parent.ARs[i]) && v.parent.CR[i-1] # by getting CL[i], we are garantueeing that AR[i] exists
    return v.parent.ARs[i]
end

Base.getindex(v::ARView{<:MPSMultiline},i::Int,j::Int) = v.parent[i].AR[j]
Base.setindex!(v::ARView{<:MPSMultiline},vec,i::Int,j::Int) = setindex!(v.parent[i].AR,vec,j);

struct CRView{S}
    parent::S
end

#no MPSComoving boundscheck needed, because they re-use AL/AR/AC getset
function Base.getindex(v::CRView{S},i::Int)::bond_type(S) where S <: Union{FiniteMPS,MPSComoving}
    if ismissing(v.parent.CLs[i+1])
        if i == 0 || !ismissing(v.parent.ALs[i])
            (v.parent.CLs[i+1],temp) = rightorth(_permute_tail(v.parent.AC[i+1]),alg=LQpos())
            v.parent.ARs[i+1] = _permute_front(temp);
        else
            (v.parent.ALs[i],v.parent.CLs[i+1]) = leftorth(v.parent.AC[i],alg=QRpos())
        end
    end
    return v.parent.CLs[i+1]
end

function Base.setindex!(v::CRView{S},vec,i::Int) where S <: Union{FiniteMPS,MPSComoving}
    if ismissing(v.parent.CLs[i+1])
        if !ismissing(v.parent.ALs[i])
            (v.parent.CLs[i+1],temp) = rightorth(_permute_tail(v.parent.AC[i+1]),alg=LQpos())
            v.parent.ARs[i+1] = _permute_front(temp);
        else
            (v.parent.ALs[i],v.parent.CLs[i+1]) = leftorth(v.parent.AC[i],alg=QRpos())
        end
    end

    v.parent.CLs.=missing;
    v.parent.ACs.=missing;
    v.parent.ALs[i+1:end].=missing;
    v.parent.ARs[1:i].=missing;

    v.parent.CLs[i+1] = vec
end

Base.getindex(v::CRView{<:MPSMultiline},i::Int,j::Int) = v.parent[i].CR[j]
Base.setindex!(v::CRView{<:MPSMultiline},vec,i::Int,j::Int) = setindex!(v.parent[i].CR,vec,j);

struct ACView{S}
    parent::S
end

function Base.getindex(v::ACView{S},i::Int)::eltype(S) where S<: Union{FiniteMPS,MPSComoving}
    !(v.parent isa MPSComoving) || (i >= 1 && i <= length(v.parent)) || throw(ArgumentError("out of bounds"))

    if ismissing(v.parent.ACs[i]) && !ismissing(v.parent.ARs[i])
        c = v.parent.CR[i-1];
        ar = v.parent.ARs[i];
        v.parent.ACs[i] = _permute_front(c*_permute_tail(ar))
    elseif ismissing(v.parent.ACs[i]) && !ismissing(v.parent.ALs[i])
        c = v.parent.CR[i];
        al = v.parent.ALs[i];
        v.parent.ACs[i] = al*c;
    end
    return v.parent.ACs[i]
end

function Base.setindex!(v::ACView{S},vec::GenericMPSTensor,i::Int) where S<: Union{FiniteMPS,MPSComoving}
    if ismissing(v.parent.ACs[i])
        i<length(v) && v.parent.AR[i+1]
        i>1 && v.parent.AL[i-1]
    end

    v.parent.ACs.=missing;
    v.parent.CLs.=missing;
    v.parent.ALs[i:end].=missing;
    v.parent.ARs[1:i].=missing;
    v.parent.ACs[i] = vec;
end

function Base.setindex!(v::ACView{S},vec::Tuple{<:GenericMPSTensor,<:GenericMPSTensor},i::Int) where S <: Union{FiniteMPS,MPSComoving}
    if ismissing(v.parent.ACs[i])
        i<length(v) && v.parent.AR[i+1]
        i>1 && v.parent.AL[i-1]
    end

    v.parent.ACs.=missing;
    v.parent.CLs.=missing;
    v.parent.ALs[i:end].=missing;
    v.parent.ARs[1:i].=missing;

    (a,b) = vec
    if isa(a,MPSBondTensor) #c/ar
        v.parent.CLs[i] = a;
        v.parent.ARs[i] = b;
    else #al/c
        @assert isa(b,MPSBondTensor)

        v.parent.CLs[i+1] = b;
        v.parent.ALs[i] = a;
    end

end

Base.getindex(v::ACView{<:MPSMultiline},i::Int,j::Int) = v.parent[i].AC[j]
Base.setindex!(v::ACView{<:MPSMultiline},vec,i::Int,j::Int) = setindex!(v.parent[i].AC,vec,j);

#linear indexing for MPSMultiline
function Base.getindex(v::Union{ACView{S},ALView{S},ARView{S},CRView{S}},i::Int) where S<:MPSMultiline
    inds = CartesianIndices(size(v))[i];
    v[inds[1],inds[2]]
end

Base.firstindex(psi::Union{ACView{S},ALView{S},ARView{S}}, i...) where S <: Union{FiniteMPS,MPSComoving} = firstindex(psi.parent.ALs, i...)
Base.lastindex(psi::Union{ACView{S},ALView{S},ARView{S}}, i...) where S <: Union{FiniteMPS,MPSComoving} = lastindex(psi.parent.ALs, i...)

Base.firstindex(psi::Union{ACView{S},ALView{S},ARView{S},CRView{S}},i...) where S<:MPSMultiline = 1;
Base.lastindex(psi::Union{ACView{S},ALView{S},ARView{S},CRView{S}},i) where S<:MPSMultiline  = i == 1 ? lastindex(psi.parent.data) : lastindex(psi.parent[1].AC);
Base.lastindex(psi::Union{ACView{S},ALView{S},ARView{S},CRView{S}}) where S<:MPSMultiline = prod(size(psi.parent));

Base.length(psi::Union{ACView,ALView,ARView}) = length(psi.parent);
Base.size(psi::Union{ACView,ALView,ARView},args...) = size(psi.parent,args...);
Base.size(psi::Union{CRView{S}},args...) where S <: Union{FiniteMPS,MPSComoving} = size(psi.parent.CLs,args...);
Base.size(psi::Union{CRView{S}},args...) where S <: MPSMultiline = size(psi.parent,args...);

Base.firstindex(psi::Union{CRView{S}}) where S <: Union{FiniteMPS,MPSComoving} = 0;
Base.lastindex(psi::Union{CRView{S}}, i...) where S <: Union{FiniteMPS,MPSComoving} = lastindex(psi.parent.ALs, i...)
Base.length(psi::Union{CRView{S}}) where S <: Union{FiniteMPS,MPSComoving} = length(psi.parent.CLs);

Base.IteratorSize(::Type{<:Union{ACView{S},ALView{S},ARView{S},CRView{S}}}) where S <: Union{FiniteMPS,MPSComoving}  = Base.HasShape{1}()
Base.IteratorSize(::Type{<:Union{ACView{S},ALView{S},ARView{S},CRView{S}}}) where S <: MPSMultiline  = Base.HasShape{2}()

Base.IteratorEltype(::Type{<:Union{ACView,ALView,ARView,CRView}}) = Base.HasEltype()
Base.iterate(view::Union{ACView,ALView,ARView,CRView},istate = firstindex(view)) = istate > lastindex(view) ? nothing : (view[istate],istate+1)

Base.getindex(psi::Union{ACView,ALView,ARView,CRView},r::AbstractRange{Int64}) = [psi[ri] for ri in r]

Base.getindex(psi::Union{ACView,ALView,ARView,CRView},i::Colon,j::Int) = psi[1:end,j];
Base.getindex(psi::Union{ACView,ALView,ARView,CRView},i::Int,j::Colon) = psi[i,1:end];
Base.getindex(psi::Union{ACView,ALView,ARView,CRView},i::Colon,j::Colon) = psi[1:end,1:end];

Base.getindex(psi::Union{ACView,ALView,ARView,CRView},i::AbstractRange{Int64},j::Int) = [psi[ri,j] for ri in i];
Base.getindex(psi::Union{ACView,ALView,ARView,CRView},i::Int,j::AbstractRange{Int64}) = [psi[i,rj] for rj in j];
Base.getindex(psi::Union{ACView,ALView,ARView,CRView},i::AbstractRange{Int64},j::AbstractRange{Int64}) = map((ri,rj)->psi[ri,rj],Iterators.product(i,j));

Base.CartesianIndices(view::Union{ACView,ALView,ARView}) = CartesianIndices(size(view));

#allow calling with cartesian index
Base.getindex(view::Union{ACView,ALView,ARView,CRView},i::CartesianIndex) = view[Tuple(i)...];
Base.setindex!(view::Union{ACView,ALView,ARView,CRView},v,i::CartesianIndex) = setindex!(view,v,Tuple(i)...);
