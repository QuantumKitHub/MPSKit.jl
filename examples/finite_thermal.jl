using MPSKit,MPSKitModels,TensorKit,Test


function bondmanage(state,ham,envs)
    final_bonddim = 12;

    upperbound = max_Ds(state);
    shouldincrease = reduce((a,i) -> a && dim(virtualspace(state,i))>=final_bonddim || upperbound[i+1] == dim(virtualspace(state,i)),1:length(state),init=true);

    if (shouldincrease)
        (state,envs) = changebonds(state, ham, OptimalExpand(),envs);
    end

    return (state,envs)
end


let
    #the operator used to evolve is the anticommutator
    th = nonsym_ising_ham()

    ham = anticommutator(th)

    ts = FiniteMPS(repeat(infinite_temperature(th),10))
    ca = environments(ts,ham);

    betastep=0.1;endbeta=2;betas=collect(0:betastep:endbeta);

    for beta in betas
        (ts,ca) = bondmanage(ts,ham,ca)
        (ts,ca) = timestep(ts,ham,-betastep*0.25im,Tdvp(),ca) # find exp(-beta*H)
    end

end
