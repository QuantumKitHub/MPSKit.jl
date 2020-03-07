struct ACView{S}
    parent::S
end

function Base.getindex(v::ACView,i::Int)
    leftorth!(v.parent,i,normalize=false)
    rightorth!(v.parent,i,normalize=false)

    return v.parent[i]
end

function Base.setindex!(v::ACView,vec,i::Int)
    leftorth!(v.parent,i,normalize=false)
    rightorth!(v.parent,i,normalize=false)

    v.parent[i] = vec
end

struct ALView{S}
    parent::S
end

function Base.getindex(v::ALView,i::Int)
    i == length(v.parent) && throw(ArgumentError("out of bounds"))
    leftorth!(v.parent,i+1,normalize = false);

    v.parent[i]
end

function Base.setindex!(v::ALView,vec,i::Int)
    i == length(v.parent) && throw(ArgumentError("out of bounds"))
    leftorth!(v.parent,i+1,normalize = false);

    v.parent[i] = vec
end

struct ARView{S}
    parent::S
end

function Base.getindex(v::ARView,i::Int)
    i == 1 && throw(ArgumentError("out of bounds"))
    rightorth!(v.parent,i-1,normalize = false);

    v.parent[i]
end

function Base.setindex!(v::ARView,vec,i::Int)
    i == 1 && throw(ArgumentError("out of bounds"))
    rightorth!(v.parent,i-1,normalize = false);

    v.parent[i] = vec
end


struct CRView{S}
    parent::S
end

function Base.getindex(v::CRView,i::Int)
    leftorth!(v.parent,i,normalize=false)
    rightorth!(v.parent,i,normalize=false)
    (AL,C) = leftorth(v.parent[i]);
    C
end

function Base.setindex!(v::CRView,vec,i::Int)
    leftorth!(v.parent,i,normalize=false)
    rightorth!(v.parent,i,normalize=false)
    (AL,C) = leftorth(v.parent[i]);

    v.parent[i] = AL*vec
end
