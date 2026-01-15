mutable struct FiniteCylinder
    L::Int
    N::Int
end
function linearize_index(lattice::FiniteCylinder, i, j)
    row = if isodd(j)
        i
    else
        lattice.L - i + 1
    end
    return mod1(row, lattice.L) + lattice.L * (j - 1)
end
function inverse_linearize_index(lattice::FiniteCylinder, idx::Int)
    L = lattice.L
    j = div(idx - 1, L) + 1
    row = mod1(idx, L)

    i = if isodd(j)
        row
    else
        L - row + 1
    end

    return i, j
end
function Shastry_Sutherland_neighbor(Lx, Ly)
    N = Lx * Ly
    mat = reshape(collect(1:N), Ly, Lx)


    for i in 2:2:Lx
        mat[:, i] .= reverse(mat[:, i])
    end

    horizontal_neighbors = [(mat[i, j], mat[i, j + 1]) for i in 1:Ly, j in 1:(Lx - 1)]
    vertical_neighbors = [(mat[i, j], mat[i < Ly ? i + 1 : 1, j]) for i in 1:Ly, j in 1:Lx]

    return vcat(vec(horizontal_neighbors), vec(vertical_neighbors))
    # return neighbours
end
function Shastry_Sutherland_next_neighbor(Lx, Ly)
    N = Lx * Ly
    mat = reshape(collect(1:N), Ly, Lx)


    for i in 2:2:Lx
        mat[:, i] .= reverse(mat[:, i])
    end

    neighbors = []
    for i in 1:2:(Lx - 1)
        for j in 1:2:Ly
            if j < Ly
                push!(neighbors, (mat[j, i], mat[j + 1, i + 1]))
            else
                push!(neighbors, (mat[j, i], mat[1, mod1(i + 1, Lx)]))
            end
        end
    end
    for i in 2:2:(Lx - 1)
        for j in 1:2:Ly
            push!(neighbors, (mat[j, i], mat[mod1(j - 1, Ly), mod1(i + 1, Lx)]))
        end
    end
    return neighbors
    # return neighbours
end
function third_neighbor(Lx, Ly)
    N = Lx * Ly
    mat = reshape(collect(1:N), Ly, Lx)


    for i in 2:2:Lx
        mat[:, i] .= reverse(mat[:, i])
    end

    horizontal_neighbors = [(mat[i, j], mat[i, j + 2]) for i in 1:Ly, j in 1:(Lx - 2)]
    vertical_neighbors = [(mat[i, j], mat[mod1(i + 2, Ly), j]) for i in 1:Ly, j in 1:Lx]
    return vcat(vec(horizontal_neighbors), vec(vertical_neighbors))

end
function hamiltonian(sites, Lx, Ly, Jxy, Jz, Jnx, Jnz, J3nx, J3nz, h)
    neighbors = Shastry_Sutherland_neighbor(Lx, Ly)
    next_neighbors = Shastry_Sutherland_next_neighbor(Lx, Ly)
    third_neighbors = third_neighbor(Lx, Ly)
    H = OpSum()
    for (i, j) in neighbors
        # H += Jxy, "Sx", i, "Sx", j
        # H += Jxy, "Sy", i, "Sy", j
        H += 0.5 * Jxy, "S+", i, "S-", j
        H += 0.5 * Jxy, "S-", i, "S+", j
        H += Jz, "Sz", i, "Sz", j
    end
    for (i, j) in next_neighbors
        H += 0.5 * Jnx, "S+", i, "S-", j
        H += 0.5 * Jnx, "S-", i, "S+", j
        # H += Jnx, "Sx", i, "Sx", j
        # H += Jnx, "Sy", i, "Sy", j
        H += Jnz, "Sz", i, "Sz", j
    end
    for (i, j) in third_neighbors
        H += 0.5 * J3nx, "S+", i, "S-", j
        H += 0.5 * J3nx, "S-", i, "S+", j
        # H += J3nx, "Sx", i, "Sx", j
        # H += J3nx, "Sy", i, "Sy", j
        H += J3nz, "Sz", i, "Sz", j
    end
    for i in 1:(Lx * Ly)
        H += h, "Sz", i
    end

    return ITensorMPS.MPO(H, sites)
end
function main(Lx, Ly, Δ, J1z, J2z, J3z, hz, Dcut; inistate = [])
    symmetry = true
    J1x = J1z
    J2x = J2z
    J3x = J3z
    Lx = Lx

    nsweeps = 10
    if isempty(inistate)
        if symmetry == true
            sites = siteinds("S=1/2", Lx * Ly, conserve_qns = true)
            initialstate = [isodd(n) ? "Up" : "Dn" for n in 1:(Lx * Ly)]
            psi0 = random_mps(sites, initialstate; linkdims = Dcut)
        else
            sites = siteinds("S=1/2", Lx * Ly, conserve_qns = false)
            psi0 = random_mps(sites, linkdims = Dcut)
        end
        maxdim = Int[Dcut + div(i, 2) * (Dcut // 1) for i in 2:(nsweeps + 1)]
    else
        psi0 = inistate

        sites = siteinds(psi)
        D = maximum(linkdims(psi0))
        maxdim = Int[D + div(i, 2) * (Dcut // 1) for i in 2:(nsweeps + 1)]
    end


    # maxdim = vcat(repeat([Dcut], 3), repeat([2*Dcut], 4), repeat([4 *Dcut], 5))

    cutoff = [1.0e-10]
    println("##################################################")
    println("Starting DMRG")
    println("J1 = $J1x, J2 = $J2x, J3 = $J3x, Lx = $Lx, Ly = $Ly, Dcut = $Dcut, hz = $hz")

    ham = hamiltonian(sites, Lx, Ly, J1x, J1z, J2x, J2z, J3x, J3z, hz)

    energy, psi = dmrg(ham, psi0; nsweeps, maxdim, cutoff, outputlevel = 1)


    filename = "./rslt/"
    magz = expect(psi, "Sz")

    WF_filename = filename * "WF_J1=$(J1z)_J3=$(J3z)_Lx=$(Lx)_Ly=$(Ly)_Dcut=$(Dcut)_hz=$(hz).jld2"
    save(WF_filename, "psi", psi, "bond_pl", bond_pl, "bond_dm", bond_dm, "magz", magz, "corr", corr)
    GC.gc()
    return
end
