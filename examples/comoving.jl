using MPSKit,TensorKit,DelimitedFiles

let
    #defining the hamiltonian
    th = nonsym_ising_ham(lambda = 0.3)
    (sxt,syt,szt) = nonsym_spintensors(1//2);

    #Center gauging a random mps
    ts=InfiniteMPS([ℂ^2],[ℂ^12]);

    #Finding the groundstate
    (ts,pars,_)=find_groundstate(ts,th,Vumps(maxiter=400));

    len=20;deltat=0.05;totaltime=3.0

    #apply a single spinflip at the middle site
    mpco=MPSComoving(ts,[ts.AC[1];ts.AR[2:len]],ts)
    @tensor mpco.middle[Int(round(len/2))][-1 -2;-3]:=ts.AR[Int(round(len/2))][-1,1,-3]*sxt[-2,1]
    mpco = rightorth(mpco)

    pars=params(mpco,th)

    szdat=[expectation_value(mpco,szt)]

    for i in 1:(totaltime/deltat)
        mpco = changebonds(mpco,RandExpand(numvecs=1)&SvdCut(trschemes=[truncdim(20)])) # grow the bond dimension by 1, and truncate at bond dimension 20
        (mpco,pars) = timestep(mpco,th,deltat,Tdvp(),pars)
        push!(szdat,expectation_value(mpco,szt))
    end

    szdat=real.(reduce((a,b)->[a b],szdat))
    #writedlm("spinflip_evolution.csv",szdat)
end
