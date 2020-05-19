using MPSKit,TensorKit,Test


function bondmanage(state,ham,pars)
    final_bonddim = 12;

    upperbound = max_Ds(state);
    shouldincrease = reduce((a,b) -> a && dim(virtualspace(state,i))>=final_bonddim || upperbound[i+1] == dim(virtualspace(state,i)),1:length(state),init=true);

    if (shouldincrease)
        (state,pars) = changebonds(state, ham, OptimalExpand(),pars);
    end

    return (state,pars)
end


function main()
    #the operator used to evolve is the anticommutator
    th = nonsym_ising_ham()

    ham = anticommutator(th)

    ts = FiniteMPS(repeat(infinite_temperature(th),10))
    ca = params(ts,ham);

    betastep=0.1;endbeta=2;betas=collect(0:betastep:endbeta);

    for beta in betas
        (ts,ca) = bondmanage(ts,ham,ca)
        (ts,ca) = timestep(ts,ham,-betastep*0.25im,Tdvp(),ca) # find exp(-beta*H)
    end

end

main();
