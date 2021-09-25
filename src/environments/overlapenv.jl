#when optimizing over the state psi1 we sometimes have terms like <psi1 | psi2>
#can reuse a lot of environments
#(we asume psi2 remains invariant)

struct OvlEnv{S,C<:AbstractTensorMap,D <: GenericMPSTensor} <: Cache
    above :: S

    ldependencies::Vector{D} #the data we used to calculate leftenvs/rightenvs
    rdependencies::Vector{D}

    leftenvs::Vector{C}
    rightenvs::Vector{C}
end

function environments(below::S,above::S,leftstart::C,rightstart::C) where S <: Union{<:FiniteMPS,<:MPSComoving} where C <: AbstractTensorMap
    leftenvs = [leftstart]
    rightenvs = [rightstart]

    for i in 1:length(above)
        push!(leftenvs,similar(leftstart))
        push!(rightenvs,similar(rightstart))
    end

    t = below.AC[1];
    return OvlEnv{S,C,eltype(below)}(above,fill(t,length(below)),fill(t,length(below)),leftenvs,reverse(rightenvs))
end

function environments(below::S,above::S) where S <:FiniteMPS
    environments(below,above,l_LL(above),r_RR(above))
end
function environments(below::S,above::S) where S <:MPSComoving
    above.left_gs == below.left_gs || throw(ArgumentError("left gs differs"))
    above.right_gs == below.right_gs || throw(ArgumentError("right gs differs"))

    environments(below,above,l_LL(above),r_RR(above))
end
#notify the cache that we updated in-place, so it should invalidate the dependencies
function poison!(ca::OvlEnv,ind)
    ca.ldependencies[ind] = similar(ca.ldependencies[ind])
    ca.rdependencies[ind] = similar(ca.rdependencies[ind])
end

#rightenv[ind] will be contracteable with the tensor on site [ind]
function rightenv(ca::OvlEnv,ind,state)
    a = findfirst(i -> !(state.AR[i] === ca.rdependencies[i]), length(state):-1:(ind+1))
    a = a == nothing ? nothing : length(state)-a+1

    if a != nothing
        #we need to recalculate
        for j = a:-1:ind+1
            ca.rightenvs[j] = transfer_right(ca.rightenvs[j+1],ca.above.AR[j],state.AR[j])
            ca.rdependencies[j] = state.AR[j]
        end
    end

    return ca.rightenvs[ind+1]
end

function leftenv(ca::OvlEnv,ind,state)
    a = findfirst(i -> !(state.AL[i] === ca.ldependencies[i]), 1:(ind-1))

    if a != nothing
        #we need to recalculate
        for j = a:ind-1
            ca.leftenvs[j+1] = transfer_left(ca.leftenvs[j],ca.above.AL[j],state.AL[j])
            ca.ldependencies[j] = state.AL[j]
        end
    end

    return ca.leftenvs[ind]
end
