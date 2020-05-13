using MPSKit,TensorKit,Test

let
    #the operator used to evolve is the anticommutator
    th = nonsym_ising_ham()

    ham = anticommutator(th)

    ts = FiniteMPS(repeat(infinite_temperature(th),10))
    ca = params(ts,ham);

    betastep=0.1;endbeta=2;betas=collect(0:betastep:endbeta);

    for beta in betas
        (ts,ca) = managebonds(ts,ham,SimpleManager(10),ca);
        (ts,ca) = timestep(ts,ham,-betastep*0.25im,Tdvp(),ca) # find exp(-beta*H)
    end

end
