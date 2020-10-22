#works for general tensors
expectation_value(state::Union{InfiniteMPS,MPSComoving,FiniteMPS},opp::AbstractTensorMap) = expectation_value(state,fill(opp,length(state)))
function expectation_value(state::Union{InfiniteMPS,MPSComoving,FiniteMPS},opps::AbstractArray{<:AbstractTensorMap})
    #todo : gauge gets moved all over the place for finite and comoving states
    #this will invalidate possible caches
    #we should probably not be moving the gauge
    map(zip(state.AC,opps)) do (t,opp)
        tr(t'*permute(opp*permute(t,TensorKit.allind(t)[2:end-1],(1,TensorKit.numind(t))),(TensorKit.numind(t)-1,TensorKit.allind(t)[1:end-2]...),(TensorKit.numind(t),)))
    end
end

"""
calculates the expectation value of op, where op is a plain tensormap where the first index works on site at
"""
function expectation_value(state::Union{FiniteMPS{T},MPSComoving{T},InfiniteMPS{T}},op::AbstractTensorMap,at::Int) where T <: MPSTensor
    expectation_value(state,decompose_localmpo(add_util_leg(op)),at);
end

"""
calculates the expectation value of op = op1*op2*op3*... (ie an N site operator) starting at site at
"""
function expectation_value(state::Union{FiniteMPS{T},MPSComoving{T},InfiniteMPS{T}},op::AbstractArray{<:AbstractTensorMap}, at::Int) where T <: MPSTensor
    @tensor tmp[-1  -2; -3 -4] := state.AC[at+0][1,2,-4]*op[1][-1,3,-3,2]*conj(state.AC[at+0][1,3,-2])

    for index in 2:length(op)
        @tensor tmp[-1, -2, -3, -4] := tmp[-1,1,2,4]*state.AR[at+index-1][4,5,-4]*op[index][2,3,-3,5]*conj(state.AR[at+index-1][1,3,-2])
    end

    return( @tensor tmp[1,2,1,2] )
end



"""
    calculates the expectation value for the given operator/hamiltonian
"""
function expectation_value(state::MPSComoving,ham::MPOHamiltonian,pars=params(state,ham))
    vals = expectation_value_fimpl(state,ham,pars);

    tot = 0.0+0im;
    for i in 1:ham.odim
        for j in 1:ham.odim

            tot+= @tensor  leftenv(pars,length(state),state)[i][1,2,3]*
                            state.AC[end][3,4,5]*
                            rightenv(pars,length(state),state)[j][5,6,7]*
                            ham[length(state),i,j][2,8,6,4]*
                            conj(state.AC[end][1,8,7])

        end
    end

    return vals,tot/(norm(state.AC[end])^2);
end
expectation_value(state::FiniteMPS,ham::MPOHamiltonian,pars=params(state,ham)) = expectation_value_fimpl(state,ham,pars)
function expectation_value_fimpl(state::Union{MPSComoving,FiniteMPS},ham::MPOHamiltonian,pars)
    ens=zeros(eltype(eltype(state)),length(state))
    for i=1:length(state)
        for (j,k) in keys(ham,i)

            if !((j == 1 && k!= 1) || (k == ham.odim && j!=ham.odim))
                continue
            end

            cur = @tensor leftenv(pars,i,state)[j][1,2,3]*state.AC[i][3,7,5]*rightenv(pars,i,state)[k][5,8,6]*conj(state.AC[i][1,4,6])*ham[i,j,k][2,4,8,7]
            if !(j==1 && k == ham.odim)
                cur/=2
            end

            ens[i]+=cur
        end
    end

    n = norm(state.AC[end])^2
    return ens./n;
end

function expectation_value(st::InfiniteMPS,ham::MPOHamiltonian,prevca=params(st,ham))
    #calculate energy density
    len = length(st);
    ens = PeriodicArray(zeros(eltype(st.AR[1]),len));
    for i=1:len
        util = Tensor(ones,space(prevca.lw[i+1,ham.odim],2))
        for j=ham.odim:-1:1
            apl = transfer_left(leftenv(prevca,i,st)[j],ham[i,j,ham.odim],st.AL[i],st.AL[i]);
            ens[i] += @tensor apl[1,2,3]*r_LL(st,i)[3,1]*conj(util[2])
        end
    end
    return ens
end

#the mpo hamiltonian over n sites has energy f+n*edens, which is what we calculate here. f can then be found as this - n*edens
function expectation_value(st::InfiniteMPS,ham::MPOHamiltonian,size::Int,prevca=params(st,ham))
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

expectation_value(st::InfiniteMPS,opp::PeriodicMPO,ca=params(st,opp)) = expectation_value(convert(MPSMultiline,st),opp,ca);
function expectation_value(st::MPSMultiline,opp::PeriodicMPO,ca=params(st,opp))
    retval = PeriodicArray{eltype(st.AC[1,1]),2}(undef,size(st,1),size(st,2));
    for (i,j) in Iterators.product(1:size(st,1),1:size(st,2))
        retval[i,j] = @tensor   leftenv(ca,i,j,st)[1,2,3]*
                                opp[i,j][2,4,5,6]*
                                st.AC[i,j][3,6,7]*
                                rightenv(ca,i,j,st)[7,5,8]*
                                conj(st.AC[i,j][1,4,8])
    end
    return retval
end

function expectation_value(state::FiniteMPS,ham::ComAct,cache=params(state,ham))
    ens=zeros(eltype(eltype(state)),length(state))
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
                                state.AC[i][7,2,-3,1]*
                                ham[i,j,k][8,-2,3,2]*
                                rightenv(cache,i,state)[k][1,3,-4]*
                                conj(state.AC[i][-1,-2,-3,-4])
            else
                cur = @tensor   leftenv(cache,i,state)[j][-1,6,7]*
                                state.AC[i][7,-2,4,2]*
                                ham[i,j,k][6,4,5,-3]*
                                rightenv(cache,i,state)[k][2,5,-4]*
                                conj(state.AC[i][-1,-2,-3,-4])
            end

            if !(cj==1 && ck == c_odim)
                cur/=2
            end

            ens[i]+=cur
        end
    end
    n = norm(state.AC[end])^2
    return ens./n
end

expectation_value(state::FiniteQP,opp) = expectation_value(convert(FiniteMPS,state),opp)
