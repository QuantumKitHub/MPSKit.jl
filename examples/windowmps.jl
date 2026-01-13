using MPSKit, MPSKitModels, TensorKit, Plots

let
    #defining the Hamiltonian
    th = transverse_field_ising(; g = 0.3)
    sx, sy, sz = σˣ(ComplexF64), σʸ(ComplexF64), σᶻ(ComplexF64)

    #initilizing a random mps
    ts = InfiniteMPS([ℂ^2], [ℂ^12])

    #Finding the ground state
    ts, envs, _ = find_groundstate(ts, th, VUMPS(; maxiter = 400))

    len = 20
    deltat = 0.05
    totaltime = 3.0
    middle = Int(round(len / 2))

    #apply a single spinflip at the middle site
    mpco = WindowMPS(ts, len)
    @tensor mpco.AC[middle][-1 -2; -3] := mpco.AC[middle][-1, 1, -3] * sx[-2, 1]
    normalize!(mpco)

    envs = environments(mpco, th)

    szdat = [expectation_value(mpco, i => sz) for i in 1:length(mpco)]
    szdat = [szdat]

    for i in 1:(totaltime / deltat)
        mpco, envs = timestep(mpco, th, 0, deltat, TDVP2(; trscheme = truncbelow(10^(-8)) & truncdim(25)), envs)
        push!(szdat, [expectation_value(mpco, i => sz) for i in 1:length(mpco)])
    end

    display(heatmap(real.(reduce((a, b) -> [a b], szdat))))

    println("Enter to continue ...")
    readline()
end
