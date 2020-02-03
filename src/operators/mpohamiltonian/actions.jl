# how does an mpohamiltonian act on a state ...

function ac_prime(x::MpsType,pos,mps,cache::Union{AutoCache,SimpleCache})
    ham=cache.ham

    toret=zero(x)
    for (i,j) in keys(ham,pos)
        @tensor toret[-1,-2,-3]+=leftenv(cache,pos,mps)[i][-1,5,4]*x[4,2,1]*ham[pos,i,j][5,-2,3,2]*rightenv(cache,pos,mps)[j][1,3,-3]
    end

    return toret
end

function ac2_prime(x::MpoType,pos,mps,cache::Union{AutoCache,SimpleCache})
    ham=cache.ham

    toret=zero(x)

    for (i,j) in keys(ham,pos)
        for k in 1:ham.odim
            if contains(ham,pos+1,j,k)
                @tensor toret[-1,-2,-3,-4]+=leftenv(cache,pos,mps)[i][-1,7,6]*x[6,5,3,1]*ham[pos,i,j][7,-2,4,5]*ham[pos+1,j,k][4,-3,2,3]*rightenv(cache,pos+1,mps)[k][1,2,-4]
            end
        end

    end

    return toret
end

#C to the right of pos
function c_prime(x::MpsVecType,pos,mps,cache::Union{AutoCache,SimpleCache})
    toret=zero(x)
    ham=cache.ham

    for i in 1:ham.odim
        @tensor toret[-1,-2]+=leftenv(cache,pos+1,mps)[i][-1,2,1]*x[1,3]*rightenv(cache,pos,mps)[i][3,2,-2]
    end

    return toret
end

#calculates the energy density
#mpscomoving returns both an energy density, and a fixed energy contribution
function expectation_value(state::MpsComoving,ham::MpoHamiltonian,pars=params(state,ham))
    vals = expectation_value_fimpl(state,ham,pars);

    #difference between total energy of the window and the onsite energy = the contribution due to inside <=> outside interactions
    tot = sum(vals);
    for i in 1:ham.odim
        tot-=@tensor leftenv(pars,length(state)+1,state)[i][1,2,3]*rightenv(pars,length(state),state)[i][3,2,1]
    end

    return vals,tot
end
expectation_value(state::FiniteMps,ham::MpoHamiltonian,pars=params(state,ham)) = expectation_value_fimpl(state,ham,pars);
function expectation_value_fimpl(state::Union{MpsComoving,FiniteMps},ham::MpoHamiltonian,pars)
    ens=zeros(eltype(state[1]),length(state))
    for i=1:length(state)
        for (j,k) in keys(ham,i)

            if !((j == 1 && k!= 1) || (k == ham.odim && j!=ham.odim))
                continue
            end

            cur = @tensor leftenv(pars,i,state)[j][1,2,3]*state[i][3,7,5]*rightenv(pars,i,state)[k][5,8,6]*conj(state[i][1,4,6])*ham[i,j,k][2,4,8,7]
            if !(j==1 && k == ham.odim)
                cur/=2
            end

            ens[i]+=cur
        end
    end

    return ens
end

function expectation_value(st::MpsCenterGauged,ham::MpoHamiltonian,prevca=params(st,ham))
    #calculate energy density
    len=length(st);
    ens=zeros(eltype(st.AR[1]),len)
    for i=1:len
        util=Tensor(I,space(prevca.lw[mod1(i+1,len),ham.odim],2))
        for j=ham.odim:-1:1
            apl =mps_apply_transfer_left(prevca.lw[i,j],ham[i,j,ham.odim],st.AL[i],st.AL[i]);
            ens[mod1(i+1,len)]+=@tensor apl[1,2,3]*r_LL(st,i)[3,1]*conj(util[2])
        end
    end
    return ens
end

#the mpo hamiltonian over n sites has energy f+n*edens, which is what we calculate here. f can then be found as this - n*edens
function expectation_value(st::MpsCenterGauged,ham::MpoHamiltonian,size::Int,prevca=params(st,ham))
    len=length(st)
    start=leftenv(prevca,1,st)
    start=[@tensor x[-1 -2;-3]:=y[1,-2,3]*st.CR[0][3,-3]*conj(st.CR[0][1,-1]) for y in start]

    for i in 1:size
        start=mps_apply_transfer_left(start,ham,i,st.AR[i],st.AR[i])
    end

    tot=0.0+0im
    for i=1:ham.odim
        tot+=@tensor start[i][1,2,3]*rightenv(prevca,size,st)[i][3,2,1]
    end

    return tot
end
