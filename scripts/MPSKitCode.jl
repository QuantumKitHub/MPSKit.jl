using MPSKitModels

function Shastry_Sutherland_next_neighbor(lattice::InfiniteCylinder)
    rows = lattice.L
    cols = lattice.N ÷ lattice.L
    V = vertices(lattice)
    neighbours = Pair{eltype(V), eltype(V)}[]
    for i in 1:2:(rows - 1)
        for j in 1:2:cols
            if i < rows
                push!(neighbours, LatticePoint((i, j), lattice) => LatticePoint((i + 1, j + 1), lattice))
            else
                push!(neighbours, LatticePoint((i, j), lattice) => LatticePoint((1, mod1(j + 1, rows)), lattice))
            end
        end
    end
    for i in 2:2:cols
        for j in 1:2:rows
            # push!(neighbors, (mat[j, i], mat[mod1(j - 1, Ly), mod1(i + 1, Lx)]))
            push!(neighbours, LatticePoint((j, i), lattice) => LatticePoint((mod1(j - 1, rows), i + 1), lattice))
        end
    end
    return neighbours
    # return neighbours
end
function third_neighbours(lattice::InfiniteCylinder)
    V = vertices(lattice)
    neighbours = Pair{eltype(V), eltype(V)}[]
    for v in V
        push!(neighbours, v => v + (0, 2))
        if v.coordinates[1] < lattice.L ||
                lattice isa InfiniteCylinder ||
                lattice isa InfiniteHelix
            push!(neighbours, v => v + (2, 0))
        end
    end
    return neighbours
end

function Shastry_Sutherland_next_neighbor(lattice::FiniteCylinder)
    rows = lattice.L
    cols = lattice.N ÷ lattice.L
    V = vertices(lattice)
    neighbours = Pair{eltype(V), eltype(V)}[]
    for i in 1:2:(rows - 1)
        for j in 1:2:cols
            if i < rows
                push!(neighbours, LatticePoint((i, j), lattice) => LatticePoint((i + 1, j + 1), lattice))
            else
                push!(neighbours, LatticePoint((i, j), lattice) => LatticePoint((1, mod1(j + 1, rows)), lattice))
            end
        end
    end
    for i in 2:2:(cols - 1)
        for j in 1:2:rows
            # push!(neighbors, (mat[j, i], mat[mod1(j - 1, Ly), mod1(i + 1, Lx)]))
            push!(neighbours, LatticePoint((j, i), lattice) => LatticePoint((mod1(j - 1, rows), i + 1), lattice))
        end
    end
    return neighbours
    # return neighbours
end

function third_neighbours(lattice::FiniteCylinder)
    V = vertices(lattice)
    neighbours = Pair{eltype(V), eltype(V)}[]
    for v in V
        if v.coordinates[2] < lattice.N / lattice.L - 1
            push!(neighbours, v => v + (0, 2))
        end
        # if v.coordinates[1] < lattice.L
        push!(neighbours, v => v + (2, 0))
        # end
    end
    return neighbours
end

function inverse_linearize_index(lattice::Union{InfiniteCylinder, FiniteCylinder}, idx::Int)
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
function hamiltonian(J, J3, lattice; symmetry = [])
    next_neighbours = Shastry_Sutherland_next_neighbor(lattice)
    third = third_neighbours(lattice)
    if symmetry isa Type
        H = @mpoham sum(J * S_exchange(ComplexF64, symmetry){i, j} for (i, j) in nearest_neighbours(lattice)) +
            sum(S_exchange(ComplexF64, symmetry){i, j} for (i, j) in next_neighbours) +
            sum(J3 * S_exchange(ComplexF64, symmetry){i, j} for (i, j) in third)
    else
        H = @mpoham sum(J * S_exchange(ComplexF64){i, j} for (i, j) in nearest_neighbours(lattice)) +
            sum(S_exchange(ComplexF64){i, j} for (i, j) in next_neighbours) +
            sum(J3 * S_exchange(ComplexF64){i, j} for (i, j) in third)
    end

    return H
end
function linearize_index(lattice::Union{InfiniteCylinder, FiniteCylinder}, i::Int, j::Int)
    row = if isodd(j)
        i
    else
        lattice.L - i + 1
    end
    return mod1(row, lattice.L) + lattice.L * (j - 1)
end


initial_bond_dim = 10
symmetry = SU2Irrep
L = 8
W = 24

if symmetry == SU2Irrep
    physical_space = SU2Space(1 // 2 => 1)
    virtual_space = SU2Space(0 => initial_bond_dim, 1 // 2 => initial_bond_dim, 1 => initial_bond_dim)
    state = FiniteMPS(L * W, physical_space, virtual_space)
end;

J = J2 = J3 = 1.0
lattice = FiniteCylinder(L, L * W)
H = hamiltonian(J, J3, lattice; symmetry = symmetry)

Dcut = 400
groundstate, envs, delta = find_groundstate(
    state, H,
    DMRG2(; trscheme = truncdim(Dcut), maxiter = 5, tol = 1.0e-6, alg_eigsolve = (; krylovdim = 3, maxiter = 1))
)

save(filename, "groundstate", groundstate)

return groundstate, envs
