"
    FiniteMps(data::Array)

    finite one dimensional mps
    algorithms usually assume a right-orthormalized input
"
struct FiniteMps{T<:GenMpsType} <: AbstractArray{T,1}
    data::Array{T,1}

    function FiniteMps(data::Array{T,1}) where T<:GenMpsType
        ou = oneunit(space(data[1],1));
        @assert space(data[1],1) == ou
        @assert space(data[end],length(codomain(data[end]))+length(domain(data[end]))) == ou'
        new{T}(data)
    end
end

FiniteMps{T}(init,len) where T = FiniteMps(Array{T,1}(init,len))

Base.length(arr::FiniteMps) = length(arr.data)
Base.size(arr::FiniteMps) = size(arr.data)
Base.getindex(arr::FiniteMps,I::Int) = arr.data[I]
Base.setindex!(arr::FiniteMps{T},v::T,I::Int) where T = setindex!(arr.data,v,I) #should add oneunit checks ...
Base.eltype(arr::FiniteMps{T}) where T = T
Base.iterate(arr::FiniteMps,state=1) = Base.iterate(arr.data,state)
Base.lastindex(arr::FiniteMps) = lastindex(arr.data)
Base.lastindex(arr::FiniteMps,d) = lastindex(arr.data,d)
Base.copy(arr::FiniteMps) where T= FiniteMps(copy(arr.data));
Base.deepcopy(arr::FiniteMps) where T = FiniteMps(deepcopy(arr.data));
Base.similar(arr::FiniteMps) = FiniteMps(similar.(arr))
r_RR(state::FiniteMps{T}) where T = isomorphism(Matrix{eltype(T)},domain(state[end]),domain(state[end]))
l_LL(state::FiniteMps{T}) where T = isomorphism(Matrix{eltype(T)},space(state[1],1),space(state[1],1))

"
    take the sum of 2 finite mpses
"
function Base.:+(v1::FiniteMps{T},v2::FiniteMps{T}) where T #untested and quite horrible code, but not sure how to make it nice
    @assert length(v1)==length(v2)

    ou = oneunit(space(v1[1],1));

    m1 = TensorMap(rand,eltype(T),ou,ou⊕ou);
    (_,m1) = rightorth(m1);
    m2 = rightnull(m1);

    pm1 = m1+m2;

    tot = similar(v1);

    for i = 1:length(v1)
        nm1 = TensorMap(rand,eltype(T),space(v1[i],3)',space(v1[i],3)⊕space(v2[i],3));
        (_,nm1) = rightorth(nm1);
        nm2 = rightnull(nm1);

        @tensor t[-1 -2;-3] := conj(m1[1,-1])*v1[i][1,-2,2]*nm1[2,-3]
        @tensor t[-1 -2;-3] += conj(m2[1,-1])*v2[i][1,-2,2]*nm2[2,-3]

        tot[i] = t;

        m1 = nm1;
        m2 = nm2;
    end

    pm2 = m1+m2;

    @tensor tot[1][-1 -2;-3] := pm1[-1,1]*tot[1][1,-2,-3]
    @tensor tot[end][-1 -2;-3] := tot[end][-1,-2,1]*conj(pm2[-3,1])

    return tot
end


function LinearAlgebra.dot(v1::FiniteMps,v2::FiniteMps)
    @assert length(v1)==length(v2)

    @tensor start[-1;-2]:=v2[1][1,2,-2]*conj(v1[1][1,2,-1])
    for i in 2:length(v1)-1
        start=transfer_left(start,v2[i],v1[i])
    end

    @tensor start[1,2]*v2[end][2,3,4]*conj(v1[end][1,3,4])
end
