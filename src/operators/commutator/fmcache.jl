#the finitempo cache
mutable struct FiniteMpoCache{A <: MpoType,B <: ComAct,C <: MpsType} <: Cache
    ldependencies::FiniteMpo{A} #the data we used to calculate leftenvs/rightenvs
    rdependencies::FiniteMpo{A}

    opp::B #the hamiltonian used

    #todo : make this type stable
    leftenvs::Array{Array{C,1},1}
    rightenvs::Array{Array{C,1},1}
end

#constructor
function params(state::FiniteMpo,ham::ComAct)
    lll = l_LL(state);rrr = r_RR(state)

    @tensor sillyel[-1 -2;-3]:=lll[-1,-3]*Tensor(I,ham.above.domspaces[1][1]')[-2]
    rightstart = Array{typeof(sillyel),1}(undef,ham.odim);
    leftstart = Array{typeof(sillyel),1}(undef,ham.odim);

    for i in 1:ham.odim
        util_left = Tensor(I,ham.domspaces[1][i]')
        util_right = Tensor(I,ham.imspaces[length(state)][i]')

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


    leftenvs = [leftstart]
    rightenvs = [rightstart]


    for i in 1:length(state)
        push!(leftenvs,mps_apply_transfer_left(leftenvs[end],ham,i,state[i]))
        push!(rightenvs,mps_apply_transfer_right(rightenvs[end],ham,length(state)-i+1,state[length(state)-i+1]))
    end

    ldependencies = copy(state)
    rdependencies = copy(state)

    return FiniteMpoCache(ldependencies,rdependencies,ham,leftenvs,reverse(rightenvs))
end

function poison!(ca::FiniteMpoCache,ind)
    ca.rdependencies[ind]=similar(ca.rdependencies[ind])
    ca.ldependencies[ind]=similar(ca.ldependencies[ind])
end

#rightenv[ind] will be contracteable with the tensor on site [ind]
function rightenv(ca::FiniteMpoCache,ind,state)
    a = findfirst(i -> !(state[i] === ca.rdependencies[i]), length(state):-1:1)
    a = a == nothing ? nothing : length(state)-a+1

    if a != nothing && a > ind
        #we need to recalculate
        for j = a:-1:ind+1
            ca.rightenvs[j] = mps_apply_transfer_right(ca.rightenvs[j+1],ca.opp,j,state[j])
            ca.rdependencies[j]=state[j]
        end
    end

    return ca.rightenvs[ind+1]
end

function leftenv(ca::FiniteMpoCache,ind,state)
    a = findfirst(i -> !(state[i] === ca.ldependencies[i]), 1:length(state))

    if a != nothing && a < ind
        #we need to recalculate
        for j = a:ind-1
            ca.leftenvs[j+1] = mps_apply_transfer_left(ca.leftenvs[j],ca.opp,j,state[j])
            ca.ldependencies[j]=state[j]
        end
    end

    return ca.leftenvs[ind]
end
