using MPSKit,TensorKit,Test,LinearAlgebra

let
    #the operator used to evolve is the anticommutator
    ham = anticommutator(nonsym_ising_ham())

    #the infinite temperature density matrix
    inftemp = infinite_temperature(ham)
    
    ts = FiniteMPO([copy(inftemp) for i in 1:10])
    ts = rightorth(ts)

    ca = params(ts,ham);

    sx = TensorMap([0 1;1 0],ℂ^2,ℂ^2);

    betastep=0.1;endbeta=2;betas=collect(0:betastep:endbeta);
    sxdat=Float64[];

    for beta in betas
        ts[1]/=norm(ts[1]) # by normalizing, we are fixing tr(exp(-beta*H)*exp(-beta*H))=1

        push!(sxdat,sum(real(expectation_value(ts,sx)))/length(ts)) # calculate the average magnetization at exp(-2*beta*H)

        (ts,ca)=timestep(ts,ham,-betastep*0.25im,Tdvp(),ca) # find exp(-beta*H)
    end

    @test sxdat[end]<sxdat[1]
end
