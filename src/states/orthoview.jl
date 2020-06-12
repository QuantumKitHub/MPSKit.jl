struct ALView{S}
    parent::S
end

function Base.getindex(v::ALView,i::Int)
    leftorth!(v.parent,i,normalize = false);
    return v.parent.site_tensors[i]
end

function Base.setindex!(v::ALView,vec,i::Int)
    leftorth!(v.parent,i,normalize = false);
    v.parent.site_tensors[i] = vec;
end

struct ARView{S}
    parent::S
end

function Base.getindex(v::ARView,i::Int)
    rightorth!(v.parent,i,normalize = false);
    return v.parent.site_tensors[i]
end

function Base.setindex!(v::ARView,vec,i::Int)
    rightorth!(v.parent,i,normalize = false);
    v.parent.site_tensors[i] = vec;
end

struct CRView{S}
    parent::S
end

function Base.getindex(v::CRView,i::Int)
    i != 0 && leftorth!(v.parent,i,normalize=false)
    i != length(v.parent) && rightorth!(v.parent,i+1,normalize=false)
    return v.parent.bond_tensors[i+1]::bond_type(typeof(v.parent))
end

function Base.setindex!(v::CRView,vec,i::Int)
    i != 0 && leftorth!(v.parent,i,normalize=false)
    i != length(v.parent) && rightorth!(v.parent,i+1,normalize=false)
    v.parent.bond_tensors[i+1] = vec
end


struct ACView{S}
    parent::S
end

function Base.getindex(v::ACView,i::Int)
    i != 1 && leftorth!(v.parent,i-1,normalize=false)
    i != length(v.parent) && rightorth!(v.parent,i+1,normalize=false)

    if !ismissing(v.parent.bond_tensors[i])
        v.parent.site_tensors[i] = _permute_front(v.parent.bond_tensors[i]*_permute_tail(v.parent.site_tensors[i]))
        v.parent.bond_tensors[i] = missing;
    end

    if !ismissing(v.parent.bond_tensors[i+1])
        v.parent.site_tensors[i] = v.parent.site_tensors[i]*v.parent.bond_tensors[i+1]
        v.parent.bond_tensors[i+1] = missing;
    end
    v.parent.gaugedpos = (i-1,i+1);

    return v.parent.site_tensors[i]
end

function Base.setindex!(v::ACView,vec::GenericMPSTensor,i::Int)
    i != 1 && leftorth!(v.parent,i-1,normalize=false)
    i != length(v.parent) && rightorth!(v.parent,i+1,normalize=false)

    v.parent.bond_tensors[i] = missing;
    v.parent.bond_tensors[i+1] = missing;

    v.parent.site_tensors[i] = vec;

    v.parent.gaugedpos = (i-1,i+1);
end

# AL * C or C * AR. Note that C has to be a positive semidefinite matrix
function Base.setindex!(v::ACView,vec::Tuple{<:GenericMPSTensor,<:GenericMPSTensor},i::Int)
    i != 1 && leftorth!(v.parent,i-1,normalize=false)
    i != length(v.parent) && rightorth!(v.parent,i+1,normalize=false)

    v.parent.bond_tensors[i] = missing;
    v.parent.bond_tensors[i+1] = missing;

    (a,b) = vec
    if isa(a,MPSBondTensor) #c/ar
        v.parent.bond_tensors[i] = a;
        v.parent.site_tensors[i] = b;
        v.parent.gaugedpos = (i-1,i);
    else #al/c
        @assert isa(b,MPSBondTensor)

        v.parent.site_tensors[i] = a;
        v.parent.bond_tensors[i+1] = b;
        v.parent.gaugedpos = (i,i+1);
    end

end

Base.firstindex(psi::Union{ACView,ALView,ARView}, i...) = firstindex(psi.parent.site_tensors, i...)
Base.lastindex(psi::Union{ACView,ALView,ARView}, i...) = lastindex(psi.parent.site_tensors, i...)
Base.length(psi::Union{ACView,ALView,ARView}) = length(psi.parent);
Base.size(psi::Union{ACView,ALView,ARView},args...) = size(psi.parent,args...);

Base.firstindex(psi::Union{CRView}, i...) = firstindex(psi.parent.bond_tensors, i...)
Base.lastindex(psi::Union{CRView}, i...) = lastindex(psi.parent.site_tensors, i...)
Base.length(psi::Union{CRView}) = length(psi.parent.bond_tensors);
Base.size(psi::Union{CRView},args...) = size(psi.parent.bond_tensors,args...);

Base.IteratorSize(::Type{<:Union{ACView,ALView,ARView,CRView}}) = Base.HasShape{1}()
Base.IteratorEltype(::Type{<:Union{ACView,ALView,ARView,CRView}}) = Base.HasEltype()
Base.iterate(view::Union{ACView,ALView,ARView,CRView},istate = 1) = istate > length(view) ? nothing : (view[istate],istate+1)

Base.getindex(psi::Union{ACView,ALView,ARView,CRView},r::AbstractRange{Int64}) = [psi[ri] for ri in r]
