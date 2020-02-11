"
    FinEnv keeps track of the environments for FiniteMps / MpsComoving / FiniteMpo
    It automatically checks if the queried environment is still correctly cached and if not - recalculates
"
struct FinEnv{B <: Operator,C <: MpsType,D <: TensorMap} <: Cache
    ldependencies::Array{D,1} #the data we used to calculate leftenvs/rightenvs
    rdependencies::Array{D,1}

    opp::B #the operator

    leftenvs::Array{Array{C,1},1}
    rightenvs::Array{Array{C,1},1}
end

#the constructor used for any state (finitemps or mpscomoving)
#we really should be doing this lazily
function params(state,opp::Operator,leftstart::Array{C,1},rightstart::Array{C,1}) where C<:MpsType
    leftenvs = [leftstart]
    rightenvs = [rightstart]

    for i in 1:length(state)
        push!(leftenvs,transfer_left(leftenvs[end],opp,i,state[i]))
        push!(rightenvs,transfer_right(rightenvs[end],opp,length(state)-i+1,state[length(state)-i+1]))
    end

    return FinEnv([state[i] for i in 1:length(state)],[state[i] for i in 1:length(state)],opp,leftenvs,reverse(rightenvs))
end

#automatically construct the correct leftstart/rightstart for a finitemps
function params(state::FiniteMps,ham::MpoHamiltonian)
    lll = l_LL(state);rrr = r_RR(state)
    rightstart = Array{eltype(state),1}();leftstart = Array{eltype(state),1}()

    for i in 1:ham.odim
        util_left = Tensor(ones,ham.domspaces[1][i]')
        util_right = Tensor(ones,ham.imspaces[length(state)][i]')

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
function params(state::MpsComoving,ham::MpoHamiltonian;lpars=params(state.left_gs,ham),rpars=params(state.right_gs,ham))
    params(state,ham,leftenv(lpars,1,state.left_gs),rightenv(rpars,length(state),state.right_gs))
end

function params(state::FiniteMpo,ham::ComAct)
    lll = l_LL(state);rrr = r_RR(state)

    @tensor sillyel[-1 -2;-3]:=lll[-1,-3]*Tensor(ones,ham.above.domspaces[1][1]')[-2]
    rightstart = Array{typeof(sillyel),1}(undef,ham.odim);
    leftstart = Array{typeof(sillyel),1}(undef,ham.odim);

    for i in 1:ham.odim
        util_left = Tensor(ones,ham.domspaces[1][i]')
        util_right = Tensor(ones,ham.imspaces[length(state)][i]')

        if isbelow(ham,i)
            @tensor ctl[-1 -2; -3]:= lll[-1,-3]*util_left[-2]
            @tensor ctr[-1 -2; -3]:= rrr[-1,-3]*util_right[-2]

            if i != 1
                ctl = zero(ctl)
            end

            if i != ham.below.odim
                ctr = zero(ctr)
            end

            leftstart[i] = ctl
            rightstart[i] = ctr
        else
            ci = i-ham.below.odim

            @tensor ctl[-1 -2; -3 ]:= lll[-1,-2]*util_left[-3]
            @tensor ctr[-1 -2; -3 ]:= rrr[-2,-3]*util_right[-1]

            if ci != 1
                ctl = zero(ctl)
            end

            if ci != ham.above.odim
                ctr = zero(ctr)
            end

            leftstart[i] = ctl
            rightstart[i] = ctr
        end
    end

    return params(state,ham,leftstart,rightstart)
end

#notify the cache that we updated in-place, so it should invalidate the dependencies
function poison!(ca::FinEnv,ind)
    ca.ldependencies[ind] = similar(ca.ldependencies[ind])
    ca.rdependencies[ind] = similar(ca.rdependencies[ind])
end

#rightenv[ind] will be contracteable with the tensor on site [ind]
function rightenv(ca::FinEnv,ind,state)
    a = findfirst(i -> !(state[i] === ca.rdependencies[i]), length(state):-1:1)
    a = a == nothing ? nothing : length(state)-a+1

    if a != nothing && a > ind
        #we need to recalculate
        for j = a:-1:ind+1
            ca.rightenvs[j] = transfer_right(ca.rightenvs[j+1],ca.opp,j,state[j])
            ca.rdependencies[j] = state[j]
        end
    end

    return ca.rightenvs[ind+1]
end

function leftenv(ca::FinEnv,ind,state)
    a = findfirst(i -> !(state[i] === ca.ldependencies[i]), 1:length(state))

    if a != nothing && a < ind
        #we need to recalculate
        for j = a:ind-1
            ca.leftenvs[j+1] = transfer_left(ca.leftenvs[j],ca.opp,j,state[j])
            ca.ldependencies[j] = state[j]
        end
    end

    return ca.leftenvs[ind]
end
