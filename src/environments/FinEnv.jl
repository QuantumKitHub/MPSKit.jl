"
    FinEnv keeps track of the environments for FiniteMPS / MPSComoving
    It automatically checks if the queried environment is still correctly cached and if not - recalculates
"
struct FinEnv{B <: Operator,C <: MPSTensor,D <: TensorMap} <: Cache
    ldependencies::Array{D,1} #the data we used to calculate leftenvs/rightenvs
    rdependencies::Array{D,1}

    opp::B #the operator

    leftenvs::Array{Array{C,1},1}
    rightenvs::Array{Array{C,1},1}
end

#the constructor used for any state (finitemps or mpscomoving)
#we really should be doing this lazily
function environments(state,opp::Operator,leftstart::Array{C,1},rightstart::Array{C,1}) where C<:MPSTensor
    leftenvs = [leftstart]
    rightenvs = [rightstart]

    for i in 1:length(state)
        push!(leftenvs,similar.(leftstart))#transfer_left(leftenvs[end],opp,i,state[i]))
        push!(rightenvs,similar.(rightstart))#transfer_right(rightenvs[end],opp,length(state)-i+1,state[length(state)-i+1]))
    end
    t = state.AC[1];
    return FinEnv(fill(t,length(state)),fill(t,length(state)),opp,leftenvs,reverse(rightenvs))
end

#automatically construct the correct leftstart/rightstart for a finitemps
function environments(state::FiniteMPS,ham::MPOHamiltonian)
    lll = l_LL(state);rrr = r_RR(state)
    rightstart = Array{eltype(state),1}();leftstart = Array{eltype(state),1}()

    for i in 1:ham.odim
        util_left = Tensor(ones,eltype(eltype(state)),ham.domspaces[1,i]')
        util_right = Tensor(ones,eltype(eltype(state)),ham.imspaces[length(state),i]')

        @tensor ctl[-1 -2; -3]:= lll[-1,-3]*util_left[-2]
        @tensor ctr[-1 -2; -3]:= rrr[-1,-3]*util_right[-2]

        if i != 1
            ctl = zero(ctl)
        end

        if i != ham.odim
            ctr = zero(ctr)
        end

        push!(leftstart,ctl)
        push!(rightstart,ctr)
    end

    return environments(state,ham,leftstart,rightstart)
end

#extract the correct leftstart/rightstart for mpscomoving
function environments(state::MPSComoving,ham::MPOHamiltonian;lenvs=environments(state.left_gs,ham),renvs=environments(state.right_gs,ham))
    environments(state,ham,copy.(leftenv(lenvs,1,state.left_gs)),copy.(rightenv(renvs,length(state),state.right_gs)))
end

#notify the cache that we updated in-place, so it should invalidate the dependencies
function poison!(ca::FinEnv,ind)
    ca.ldependencies[ind] = similar(ca.ldependencies[ind])
    ca.rdependencies[ind] = similar(ca.rdependencies[ind])
end

#rightenv[ind] will be contracteable with the tensor on site [ind]
function rightenv(ca::FinEnv,ind,state)
    a = findfirst(i -> !(state.AR[i] === ca.rdependencies[i]), length(state):-1:(ind+1))
    a = a == nothing ? nothing : length(state)-a+1

    if a != nothing
        #we need to recalculate
        for j = a:-1:ind+1
            ca.rightenvs[j] = transfer_right(ca.rightenvs[j+1],ca.opp,j,state.AR[j])
            ca.rdependencies[j] = state.AR[j]
        end
    end

    return ca.rightenvs[ind+1]
end

function leftenv(ca::FinEnv,ind,state)
    a = findfirst(i -> !(state.AL[i] === ca.ldependencies[i]), 1:(ind-1))

    if a != nothing
        #we need to recalculate
        for j = a:ind-1
            ca.leftenvs[j+1] = transfer_left(ca.leftenvs[j],ca.opp,j,state.AL[j])
            ca.ldependencies[j] = state.AL[j]
        end
    end

    return ca.leftenvs[ind]
end
