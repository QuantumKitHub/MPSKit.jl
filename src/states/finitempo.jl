"
    FiniteMPO(data::Array)

    finite one dimensional mpo
    algorithms usually assume a right-orthormalized input
"
struct FiniteMPO{T<:MPOType} <: AbstractArray{T,1}
    data::Array{T,1}
end

FiniteMPO{T}(init,len) where T = FiniteMPO(Array{T,1}(init,len))
Base.length(arr::FiniteMPO) = length(arr.data)
Base.size(arr::FiniteMPO) = size(arr.data)
Base.getindex(arr::FiniteMPO,I::Int) = arr.data[I]
Base.setindex!(arr::FiniteMPO{T},v::T,I::Int) where T = setindex!(arr.data,v,I)
Base.eltype(arr::FiniteMPO{T}) where T = T
Base.iterate(arr::FiniteMPO,state=1) = Base.iterate(arr.data,state)
Base.lastindex(arr::FiniteMPO) = lastindex(arr.data)
Base.lastindex(arr::FiniteMPO,d) = lastindex(arr.data,d)
Base.copy(arr::FiniteMPO) = FiniteMPO(copy(arr.data));
Base.deepcopy(arr::FiniteMPO) = FiniteMPO(deepcopy(arr.data));
Base.similar(arr::FiniteMPO) = FiniteMPO(similar(arr.data))
r_RR(state::FiniteMPO{T}) where T = isomorphism(Matrix{eltype(T)},space(state[end],3)',space(state[end],3)')
l_LL(state::FiniteMPO{T}) where T = isomorphism(Matrix{eltype(T)},space(state[1],1),space(state[1],1))

@bm function expectation_value(ts::FiniteMPO,opp::TensorMap)
    leftenvs = [Tensor(ones,ComplexF64,space(ts[1],1)')];
    rightenvs = [Tensor(ones,ComplexF64,space(ts[length(ts)],3)')];

    for i in 1:length(ts)
        #we can't trace :(
        lefttracer = isomorphism(space(ts[i],2)',space(ts[i],2)')
        @tensor curl[-1] := leftenvs[end][1]*ts[i][1,2,-1,3]*lefttracer[2,3]
        push!(leftenvs,curl);

        righttracer = isomorphism(space(ts[length(ts)-i+1],2)',space(ts[length(ts)-i+1],2)')
        @tensor curr[-1] := rightenvs[end][1]*ts[length(ts)-i+1][-1,2,1,3]*righttracer[2,3]
        push!(rightenvs,curr);
    end

    tor = []
    for i in 1:length(ts)
        cur = @tensor leftenvs[i][1]*ts[i][1,2,3,4]*rightenvs[end-i][3]*opp[4,2]
        push!(tor,cur)
    end

    tor
end

function max_Ds(f::FiniteMPO)
    Ds = [1 for v in 1:length(f)+1];
    for i in 1:length(f)
        Ds[i+1] = Ds[i]*prod(dim(space(f[i],2))*dim(space(f[i],4)))
    end

    Ds[end] = 1;
    for i in length(f):-1:1
        Ds[i] = min(Ds[i],Ds[i+1]*prod(dim(space(f[i],2))*dim(space(f[i],4))))
    end
    Ds
end
