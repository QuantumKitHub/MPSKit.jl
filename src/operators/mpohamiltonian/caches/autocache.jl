#=
    when queried for an environment
    will check if it's data is still correct and if not recalculate
    used in finitemps - mpscomoving code (only makes sense for finite chains)
=#
struct AutoCache{B <: MpoHamiltonian,C <: MpsType} <: Cache
    ldependencies::Array{C,1} #the data we used to calculate leftenvs/rightenvs
    rdependencies::Array{C,1}

    ham::B #the hamiltonian used

    #todo : make this type stable
    leftenvs::Array{Array{C,1},1}
    rightenvs::Array{Array{C,1},1}
end

#the constructor used for any state (finitemps or mpscomoving)
#we really should be doing this lazily
function params(state,ham::MpoHamiltonian,leftstart::Array{C,1},rightstart::Array{C,1}) where C<:MpsType
    leftenvs = [leftstart]
    rightenvs = [rightstart]

    for i in 1:length(state)
        push!(leftenvs,mps_apply_transfer_left(leftenvs[end],ham,i,state[i]))
        push!(rightenvs,mps_apply_transfer_right(rightenvs[end],ham,length(state)-i+1,state[length(state)-i+1]))
    end

    return AutoCache([state[i] for i in 1:length(state)],[state[i] for i in 1:length(state)],ham,leftenvs,reverse(rightenvs))
end

#automatically construct the correct leftstart/rightstart for a finitemps
function params(state::FiniteMps,ham::MpoHamiltonian)
    lll = l_LL(state);rrr = r_RR(state)
    rightstart = Array{eltype(state),1}();leftstart = Array{eltype(state),1}()

    for i in 1:ham.odim
        util_left = Tensor(I,ham.domspaces[1][i]')
        util_right = Tensor(I,ham.imspaces[length(state)][i]')

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

    return params(state,ham,leftstart,rightstart)
end

#extract the correct leftstart/rightstart for mpscomoving
params(state::MpsComoving,ham::MpoHamiltonian;lpars=params(state.left_gs,ham),rpars=params(state.right_gs,ham)) = params(state,ham,leftenv(lpars,1,state.left_gs),rightenv(rpars,length(state),state.right_gs))

#notify the cache that we updated in-place, so it should invalidate the dependencies
function poison!(ca::AutoCache,ind)
    ca.ldependencies[ind] = similar(ca.ldependencies[ind])
    ca.rdependencies[ind] = similar(ca.rdependencies[ind])
end

#rightenv[ind] will be contracteable with the tensor on site [ind]
function rightenv(ca::AutoCache,ind,state)
    a = findfirst(i -> !(state[i] === ca.rdependencies[i]), length(state):-1:1)
    a = a == nothing ? nothing : length(state)-a+1

    if a != nothing && a > ind
        #we need to recalculate
        for j = a:-1:ind+1
            ca.rightenvs[j] = mps_apply_transfer_right(ca.rightenvs[j+1],ca.ham,j,state[j])
            ca.rdependencies[j] = state[j]
        end
    end

    return ca.rightenvs[ind+1]
end

function leftenv(ca::AutoCache,ind,state)
    a = findfirst(i -> !(state[i] === ca.ldependencies[i]), 1:length(state))

    if a != nothing && a < ind
        #we need to recalculate
        for j = a:ind-1
            ca.leftenvs[j+1] = mps_apply_transfer_left(ca.leftenvs[j],ca.ham,j,state[j])
            ca.ldependencies[j] = state[j]
        end
    end

    return ca.leftenvs[ind]
end
