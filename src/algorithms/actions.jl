# Given a state and it's environments, we can act on it

"""
    One-site derivative
"""
function ac_prime(x::MpsType,pos::Int,mps::Union{FiniteMps,MpsCenterGauged,MpsComoving},cache)
    ham=cache.opp

    toret=zero(x)
    for (i,j) in keys(ham,pos)
        @tensor toret[-1,-2,-3]+=leftenv(cache,pos,mps)[i][-1,5,4]*x[4,2,1]*ham[pos,i,j][5,-2,3,2]*rightenv(cache,pos,mps)[j][1,3,-3]
    end

    return toret
end
function ac_prime(x::MpoType,pos::Int,mpo::FiniteMpo,cache)
    ham=cache.opp

    toret=zero(x)
    for (i,j) in keys(ham,pos)
        opp = ham[pos,i,j]

        if isbelow(ham,i)
            @tensor toret[-1,-2,-3,-4] +=   leftenv(cache,pos,mpo)[i][-1,8,7]*
                                            x[7,2,1,-4]*
                                            opp[8,-2,3,2]*
                                            rightenv(cache,pos,mpo)[j][1,3,-3]
        else
            @tensor toret[-1,-2,-3,-4] +=   leftenv(cache,pos,mpo)[i][-1,6,7]*
                                            x[7,-2,2,4]*
                                            opp[6,4,5,-4]*
                                            rightenv(cache,pos,mpo)[j][2,5,-3]
        end
    end

    return toret
end
function ac_prime(x::MpsType, row::Int,col::Int,mps::Union{MpsCenterGauged,MpsMultiline}, pars::PerMpoInfEnv)
    @tensor toret[-1 -2;-3]:=leftenv(pars,row,col,mps)[-1,2,1]*x[1,3,4]*(pars.opp[row,col])[2,-2,5,3]*rightenv(pars,row,col,mps)[4,5,-3]
end

"""
    Two-site derivative
"""
function ac2_prime(x::MpoType,pos::Int,mps::Union{FiniteMps,MpsCenterGauged,MpsComoving},cache)
    ham=cache.opp

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
function ac2_prime(x::TensorMap,pos::Int,mpo::FiniteMpo,cache)
    ham=cache.opp

    toret=zero(x)
    for (i,j) in keys(ham,pos)
        for (k,l) in keys(ham,pos+1)
            if j!=k
                continue
            end
            opp1 = ham[pos,i,j]
            opp2 = ham[pos+1,k,l]

            if isbelow(ham,i)
                @tensor toret[-1,-2,-3,-4,-5,-6] += leftenv(cache,pos,mpo)[i][-1,2,1]*
                                                    x[1,3,5,7,-5,-6]*
                                                    opp1[2,-2,4,3]*
                                                    opp2[4,-3,6,5]*
                                                    rightenv(cache,pos+1,mpo)[l][7,6,-4]
            else
                @tensor toret[-1,-2,-3,-4,-5,-6] += leftenv(cache,pos,mpo)[i][-1,2,1]*
                                                    x[1,-2,-3,7,5,3]*
                                                    opp1[2,3,4,-6]*
                                                    opp2[4,5,6,-5]*
                                                    rightenv(cache,pos+1,mpo)[l][7,6,-4]
            end
        end
    end

    return toret
end
function ac2_prime(x::MpoType, row::Int,col::Int,mps::Union{MpsCenterGauged,MpsMultiline}, pars::PerMpoInfEnv)
    @tensor toret[-1 -2;-3 -4]:=leftenv(pars,row,col,mps)[-1,2,1]*
                                x[1,3,4,5]*
                                pars.opp[row,col][2,-2,6,3]*
                                pars.opp[row,col+1][6,-3,7,4]*
                                rightenv(pars,row,col+1,mps)[5,7,-4]
end

"""
    Zero-site derivative (the C matrix to the right of pos)
"""
function c_prime(x::MpsVecType,pos::Int,mps::Union{FiniteMps,MpsCenterGauged,MpsComoving},cache)
    toret=zero(x)
    ham=cache.opp

    for i in 1:ham.odim
        @tensor toret[-1,-2]+=leftenv(cache,pos+1,mps)[i][-1,2,1]*x[1,3]*rightenv(cache,pos,mps)[i][3,2,-2]
    end

    return toret
end
function c_prime(x::MpsVecType,pos::Int,mpo::FiniteMpo,cache)
    toret=zero(x)
    ham=cache.opp

    for i in 1:ham.odim
        if isbelow(ham,i)
            @tensor toret[-1,-2]+=leftenv(cache,pos+1,mpo)[i][-1,2,1]*x[1,3]*rightenv(cache,pos,mpo)[i][3,2,-2]
        else
            @tensor toret[-1,-2]+=leftenv(cache,pos+1,mpo)[i][-1,2,1]*x[1,3]*rightenv(cache,pos,mpo)[i][3,2,-2]
        end
    end

    return toret
end
function c_prime(x::TensorMap, row::Int,col::Int, mps::Union{MpsCenterGauged,MpsMultiline}, pars::PerMpoInfEnv)
    @tensor toret[-1;-2] := leftenv(pars,row,col+1,mps)[-1,3,1]*x[1,2]*rightenv(pars,row,col,mps)[2,3,-2]
end


"""
    calculates the expectation value for the given operator/hamiltonian
"""
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
    len = length(st);
    ens = Periodic(zeros(eltype(st.AR[1]),len));
    for i=1:len
        util = Tensor(ones,space(prevca.lw[i+1,ham.odim],2))
        for j=ham.odim:-1:1
            apl = transfer_left(leftenv(prevca,i,st)[j],ham[i,j,ham.odim],st.AL[i],st.AL[i]);
            ens[i+1] += @tensor apl[1,2,3]*r_LL(st,i)[3,1]*conj(util[2])
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
        start=transfer_left(start,ham,i,st.AR[i],st.AR[i])
    end

    tot=0.0+0im
    for i=1:ham.odim
        tot+=@tensor start[i][1,2,3]*rightenv(prevca,size,st)[i][3,2,1]
    end

    return tot
end

expectation_value(st::MpsCenterGauged,opp::PeriodicMpo,ca=params(st,opp)) = expectation_value(convert(MpsMultiline,st),opp,ca);
function expectation_value(st::MpsMultiline,opp::PeriodicMpo,ca=params(st,opp))
    retval = Periodic{eltype(st.AC[1,1]),2}(size(st,1),size(st,2));
    for (i,j) in Iterators.product(1:size(st,1),1:size(st,2))
        retval[i,j] = @tensor   leftenv(ca,i,j,st)[1,2,3]*
                                opp[i,j][2,4,5,6]*
                                st.AC[i,j][3,6,7]*
                                rightenv(ca,i,j,st)[7,5,8]*
                                conj(st.AC[i,j][1,4,8])
    end
    return retval
end

function expectation_value(state::FiniteMpo,ham::ComAct,cache=params(state,ham))
    ens=zeros(eltype(state[1]),length(state))
    for i=1:length(state)
        for (j,k) in keys(ham,i)

            c_odim = isbelow(ham,j) ? ham.below.odim : ham.above.odim;
            cj = isbelow(ham,j) ? j : j-ham.below.odim;
            ck = isbelow(ham,k) ? k : k-ham.below.odim;

            if !((cj == 1 && ck!= 1) || (ck == c_odim && cj!=c_odim))
                continue
            end

            if isbelow(ham,j)
                cur = @tensor   leftenv(cache,i,state)[j][-1,8,7]*
                                state[i][7,2,1,-4]*
                                ham[i,j,k][8,-2,3,2]*
                                rightenv(cache,i,state)[k][1,3,-3]*
                                conj(state[i][-1,-2,-3,-4])
            else
                cur = @tensor   leftenv(cache,i,state)[j][-1,6,7]*
                                state[i][7,-2,2,4]*
                                ham[i,j,k][6,4,5,-4]*
                                rightenv(cache,i,state)[k][2,5,-3]*
                                conj(state[i][-1,-2,-3,-4])
            end

            if !(cj==1 && ck == c_odim)
                cur/=2
            end

            ens[i]+=cur
        end
    end

    return ens
end
