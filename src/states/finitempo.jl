"
    FiniteMpo(data::Array)

    finite one dimensional mpo
    algorithms usually assume a right-orthormalized input
"
struct FiniteMpo{T<:MpoType} <: AbstractArray{T,1}
    data::Array{T,1}
end

FiniteMpo{T}(init,len) where T = FiniteMpo(Array{T,1}(init,len))
Base.length(arr::FiniteMpo) = length(arr.data)
Base.size(arr::FiniteMpo) = size(arr.data)
Base.getindex(arr::FiniteMpo,I::Int) = arr.data[I]
Base.setindex!(arr::FiniteMpo{T},v::T,I::Int) where T = setindex!(arr.data,v,I)
Base.eltype(arr::FiniteMpo{T}) where T = T
Base.iterate(arr::FiniteMpo,state=1) = Base.iterate(arr.data,state)
Base.lastindex(arr::FiniteMpo) = lastindex(arr.data)
Base.lastindex(arr::FiniteMpo,d) = lastindex(arr.data,d)
Base.copy(arr::FiniteMpo) = FiniteMpo(copy(arr.data));
Base.deepcopy(arr::FiniteMpo) = FiniteMpo(deepcopy(arr.data));
Base.similar(arr::FiniteMpo) = FiniteMpo(similar(arr.data))
r_RR(state::FiniteMpo{T}) where T = isomorphism(Matrix{eltype(T)},space(state[end],3)',space(state[end],3)')
l_LL(state::FiniteMpo{T}) where T = isomorphism(Matrix{eltype(T)},space(state[1],1),space(state[1],1))

function expectation_value(ts::FiniteMpo,opp::TensorMap)
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
