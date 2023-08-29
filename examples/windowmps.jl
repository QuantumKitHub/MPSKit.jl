using MPSKit,MPSKitModels,TensorKit,Plots

let
    #defining the hamiltonian
    th = nonsym_ising_ham(lambda = 0.3)
    (sx,sy,sz) = nonsym_spintensors(1//2);

    #initilizing a random mps
    ts = InfiniteMPS([ℂ^2],[ℂ^12]);

    #Finding the groundstate
    (ts,envs,_) = find_groundstate(ts,th,VUMPS(maxiter=400));

    len=20;deltat=0.05;totaltime=3.0
    middle = Int(round(len/2));

    #apply a single spinflip at the middle site
    mpco = WindowMPS(ts,len);
    @tensor mpco.AC[middle][-1 -2;-3] := mpco.AC[middle][-1,1,-3]*sx[-2,1]
    normalize!(mpco);

    envs = environments(mpco,th)

    szdat = [expectation_value(mpco,sz)]

    for i in 1:(totaltime/deltat)
        (mpco,envs) = timestep(mpco,th,deltat,TDVP2(trscheme = truncdim(20)),envs)
        push!(szdat,expectation_value(mpco,sz))
    end

    display(heatmap(real.(reduce((a,b)->[a b],szdat))))

    println("Enter to continue ...")
    readline();
end
