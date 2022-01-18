#works for general tensors
expectation_value(state::Union{InfiniteMPS,MPSComoving,FiniteMPS},opp::AbstractTensorMap) = expectation_value(state,fill(opp,length(state)))
function expectation_value(state::Union{InfiniteMPS,MPSComoving,FiniteMPS},opps::AbstractArray{<:AbstractTensorMap})
    map(zip(state.AC,opps)) do (ac,opp)
        tr(ac'*transpose(
            opp*transpose(ac,(TensorKit.allind(ac)[2:end-1]),(1,TensorKit.numind(ac))),
            (TensorKit.numind(ac)-1,TensorKit.allind(ac)[1:end-2]...),
            (TensorKit.numind(ac),)))
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
    firstspace = _firstspace(first(op));
    (firstspace == oneunit(firstspace) && _lastspace(last(op)) == firstspace') ||
        throw(ArgumentError("localmpo should start and end in a trivial leg, not with $(firstspace)"));

    ut = Tensor(ones,firstspace)
    @plansor v[-1 -2;-3] := isomorphism(left_virtualspace(state,at),left_virtualspace(state,at))[-1;-3]*conj(ut[-2])
    tmp = v*TransferMatrix(state.AL[at:at+length(op)-1],op,state.AL[at:at+length(op)-1])
    return @plansor tmp[1 2;3]*ut[2]*state.CR[at+length(op)-1][3;4]*conj(state.CR[at+length(op)-1][1;4]);
end



"""
    calculates the expectation value for the given operator/hamiltonian
"""
expectation_value(state,ham::MPOHamiltonian,envs=environments(state,ham)) = expectation_value(state,envs);

function expectation_value(state::MPSComoving,envs::FinEnv)
    ham = envs.opp;

    vals = expectation_value_fimpl(state,envs);

    tot = 0.0+0im;
    for i in 1:ham.odim,
        j in 1:ham.odim

        tot += @plansor  leftenv(envs,length(state),state)[i][1 2;3]*
                        state.AC[end][3 4;5]*
                        rightenv(envs,length(state),state)[j][5 6;7]*
                        ham[length(state)][i,j][2 8;4 6]*
                        conj(state.AC[end][1 8;7])

    end

    return vals,tot/(norm(state.AC[end])^2);
end

expectation_value(state::FiniteMPS,envs::FinEnv) = expectation_value_fimpl(state,envs)
function expectation_value_fimpl(state::Union{MPSComoving,FiniteMPS},envs::FinEnv)
    ham = envs.opp;

    ens=zeros(eltype(eltype(state)),length(state))
    for i in 1:length(state),
        (j,k) in keys(ham[i])

        !((j == 1 && k!= 1) || (k == ham.odim && j!=ham.odim)) && continue

        cur = @plansor leftenv(envs,i,state)[j][1 2;3]*state.AC[i][3 7;5]*rightenv(envs,i,state)[k][5 8;6]*conj(state.AC[i][1 4;6])*ham[i][j,k][2 4;7 8]
        if !(j==1 && k == ham.odim)
            cur/=2
        end

        ens[i]+=cur
    end

    n = norm(state.AC[end])^2
    return ens./n;
end

function expectation_value(st::InfiniteMPS,prevca::MPOHamInfEnv);
    ham = prevca.opp;
    #calculate energy density
    len = length(st);
    ens = PeriodicArray(zeros(eltype(st.AR[1]),len));
    for i=1:len
        util = Tensor(ones,space(prevca.lw[ham.odim,i+1],2))
        for j=ham.odim:-1:1
            apl = leftenv(prevca,i,st)[j]*TransferMatrix(st.AL[i],ham[i][j,ham.odim],st.AL[i]);
            ens[i] += @plansor apl[1 2;3]*r_LL(st,i)[3;1]*conj(util[2])
        end
    end
    return ens
end

#kept for backwards compatibility; the new way is to pass a unitrange
expectation_value(st::InfiniteMPS,ham::MPOHamiltonian,size::Int,prevca=environments(st,ham)) = expectation_value(st,prevca,1:size);
expectation_value(st::InfiniteMPS,prevca::MPOHamInfEnv,size::Int) = expectation_value(st,prevca,1:size);

#the mpo hamiltonian over n sites has energy f+n*edens, which is what we calculate here. f can then be found as this - n*edens
expectation_value(st::InfiniteMPS,ham::MPOHamiltonian,range::UnitRange{Int64},prevca = environments(st,ham)) = expectation_value(st,prevca,range)

function expectation_value(st::InfiniteMPS,prevca::MPOHamInfEnv,range::UnitRange{Int64})
    ham = prevca.opp;

    len = length(st)
    start = map(leftenv(prevca,range.start,st)) do y
        @plansor x[-1 -2;-3] := y[1 -2;3]*st.CR[range.start-1][3;-3]*conj(st.CR[range.start-1][1;-1])
    end

    for i in range
        start = start*TransferMatrix(st.AR[i],ham[i],st.AR[i])
    end

    tot = 0.0+0im
    for i=1:ham.odim
        tot += @plansor start[i][1 2;3]*rightenv(prevca,range.stop,st)[i][3 2;1]
    end

    return tot
end


expectation_value(st::InfiniteMPS,mpo::DenseMPO) = expectation_value(convert(MPSMultiline,st),convert(MPOMultiline,mpo));
expectation_value(st::MPSMultiline,mpo::MPOMultiline) = expectation_value(st,environments(st,mpo));
expectation_value(st::InfiniteMPS,ca::PerMPOInfEnv) = expectation_value(convert(MPSMultiline,st),ca);
function expectation_value(st::MPSMultiline,ca::PerMPOInfEnv)
    opp = ca.opp;
    retval = PeriodicArray{eltype(st.AC[1,1]),2}(undef,size(st,1),size(st,2));
    for (i,j) in product(1:size(st,1),1:size(st,2))
        retval[i,j] = @plansor   leftenv(ca,i,j,st)[1 2;3]*
                                opp[i,j][2 4;6 5]*
                                st.AC[i,j][3 6;7]*
                                rightenv(ca,i,j,st)[7 5;8]*
                                conj(st.AC[i+1,j][1 4;8])
    end
    return retval
end

expectation_value(state::FiniteQP,opp) = expectation_value(convert(FiniteMPS,state),opp)
