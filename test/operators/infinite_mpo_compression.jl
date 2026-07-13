using Test, Random
using TensorKit
using MPSKit
using MPSKitModels

function max_dim_ham(H)
    N = length(H)
    max_dim = 1
    for i in 1:N
        if max_dim < dims(H[i])[4]
            max_dim = dims(H[i])[4]
        end
    end
    return max_dim
end

# Utility functions to create Hamiltonians for testing

function create_long_range_ising_symmetries_infinite(k)
    infinite_chain = PeriodicVector([Z2Space(0=>1, 1=>1)])
    ZZ = S_zz(Z2Irrep)
    X = S_x(Z2Irrep)
    two_site_operators = [(1,j) => -ZZ for j in 1:k]
    H = InfiniteMPOHamiltonian(infinite_chain, two_site_operators...)
    return H
end

function create_long_range_ising_symmetries_infinite_msite_random(M, k, σⱼ, σₕ)
    V = Z2Space(0=>1, 1=>1)
    infinite_chain = PeriodicVector(fill(V, M))
    ZZ = S_zz(Z2Irrep)
    X = S_x(Z2Irrep)
    single_site_operators = [i => (σₕ * randn()) * X for i in 1:M]
    two_site_operators = [(i,i+j) => (1 + σⱼ * randn()) * ZZ for i in 1:M for j in 1:k]
    H = InfiniteMPOHamiltonian(infinite_chain, single_site_operators...,
                               two_site_operators...)
    return H
end

function create_long_range_ising_symmetries_infinite_twosite_diff_int_strength(k, J, h)
    infinite_chain = PeriodicVector([Z2Space(0=>1, 1=>1), Z2Space(0=>1, 1=>1)])
    ZZ = S_zz(Z2Irrep)
    X = S_x(Z2Irrep)
    two_site_operators_J12 = [(a,a+j) => J[1] * ZZ for a in [1 2] for j in 1:2:2k]
    two_site_operators_J11 = [(1,j) => J[2] * ZZ for j in 3:2:2k]
    two_site_operators_J22 = [(2,j) => J[3] * ZZ for j in 4:2:2k]
    single_site_operators = [1 => h[1] * X, 2 => h[2] * X]
    H = InfiniteMPOHamiltonian(infinite_chain, single_site_operators...,
                                two_site_operators_J11..., two_site_operators_J12...,
                                two_site_operators_J22...)
    return H
end

function transverse_field_ising_fermion_parity_infinite(M)
    chain = InfiniteChain(M)
    return transverse_field_ising(ComplexF64, FermionParity, chain)
end

function transverse_field_ising_z2_infinite(M)
    chain = InfiniteChain(M)
    return transverse_field_ising(ComplexF64, Z2Irrep, chain)
end

function transverse_field_ising_trivial_infinite(M)
    chain = InfiniteChain(M)
    return transverse_field_ising(ComplexF64, Trivial, chain)
end

function kitaev_model_infinite(M)
    chain = InfiniteChain(M)
    return kitaev_model(ComplexF64, chain)
end

function heisenberg_XXX_trivial_infinite(M)
    chain = InfiniteChain(M)
    return heisenberg_XXX(Trivial, chain)
end

function heisenberg_XXX_U1_infinite(M)
    chain = InfiniteChain(M)
    return heisenberg_XXX(U1Irrep, chain)
end

function heisenberg_XXX_SU2_infinite(M)
    chain = InfiniteChain(M)
    return heisenberg_XXX(SU2Irrep, chain)
end

function bilinear_biquadratic_model_trivial_infinite(M)
    chain = InfiniteChain(M)
    return bilinear_biquadratic_model(ComplexF64, Trivial, chain)
end

function quantum_potts_trivial_infinite(M)
    chain = InfiniteChain(M)
    return quantum_potts(ComplexF64, Trivial, chain)
end

function quantum_potts_Z3_infinite(M)
    chain = InfiniteChain(M)
    return quantum_potts(ComplexF64, Z3Irrep, chain)
end

@testset "infinite_mpo_compression_energy" begin
    # test the infinite compression for a long range Ising model
    # check if the energy after the compression is the same as before
    D = 4 # max bond dimensions
    range = 2 # cutoff for the maximum range between interactions
    trunc_st = notrunc()

    mps = InfiniteMPS(Z2Space(0=>1, 1=>1), Z2Space(0=>D, 1=>D))
    H = create_long_range_ising_symmetries_infinite(range)
    mps, = find_groundstate(mps, H, VUMPS(;maxiter=10))
    E0 = expectation_value(mps, H)
    #println("<mps|H|mps> = $real(E0)")

    # mps2 = InfiniteMPS(Z2Space(0=>1, 1=>1), Z2Space(0=>D, 1=>D))
    # H2, Rs = mpo_finite_compression(H)
    # find_groundstate!(mps2, H2, DMRG2(; maxiter=100, trscheme=trunc_st))
    # E1 = expectation_value(mps2, H2)
    mps2 = InfiniteMPS(Z2Space(0=>1, 1=>1), Z2Space(0=>D, 1=>D))
    Q, P = mpo_compression(H)
    H2 = Q
    mps2, = find_groundstate(mps2, H2, VUMPS(;maxiter=100))
    E0 = expectation_value(mps2, H2)
end

@testset "infinite_mpo_compression_energy_two_site" begin
    # test the infinite compression for a long range Ising model
    # check if the energy after the compression is the same as before
    D = 10 # max bond dimensions
    range = 4 # cutoff for the maximum range between interactions
    trunc_st = notrunc()
    J = [1, 1, 1]
    h = [0.001, 0.001]

    spacephys = fill(Z2Space(0=>1, 1=>1), 2)
    space_virt = fill(Z2Space(0=>D, 1=>D), 2)

    mps = InfiniteMPS(spacephys, space_virt)
    H = create_long_range_ising_symmetries_infinite_twosite_diff_int_strength(range, J, h)
    mps, = find_groundstate(mps, H, VUMPS(;maxiter=100))
    E0 = expectation_value(mps, H)
    #println("<mps|H|mps> = $real(E0)")

    # mps2 = InfiniteMPS(Z2Space(0=>1, 1=>1), Z2Space(0=>D, 1=>D))
    # H2, Rs = mpo_finite_compression(H)
    # find_groundstate!(mps2, H2, DMRG2(; maxiter=100, trscheme=trunc_st))
    # E1 = expectation_value(mps2, H2)
    mps2 = InfiniteMPS(spacephys, space_virt)
    Qs, Ps = mpo_compression(H, 0)
    H2 = Qs
    mps2, = find_groundstate(mps2, H2, VUMPS(;maxiter=100))
    E0 = expectation_value(mps2, H2)
end

@testset "infinite_mpo_compression_transverse_ising_model_energy" begin
    M = 4
    D = 10 # Max bond dimension

    mps1 = InfiniteMPS(fill(ℂ^2, M), fill(ℂ^D, M))
    mps2 = InfiniteMPS(fill(Z2Space(0=>1, 1=>1), M), fill(Z2Space(0=>D, 1=>D), M))
    mps3 = InfiniteMPS(fill(Vect[FermionParity](0=>1, 1=>1), M), fill(Vect[FermionParity](
        0=>D, 1=>D), M))

    H1 = transverse_field_ising_trivial_infinite(M)
    H2 = transverse_field_ising_z2_infinite(M)
    H3 = transverse_field_ising_fermion_parity_infinite(M)

    # Find groundstate of non-compressed Hamiltonian
    mps1, = find_groundstate(mps1, H1, VUMPS(; maxiter=100))
    E0_1 = expectation_value(mps1, H1)
    mps2, = find_groundstate(mps2, H2, VUMPS(; maxiter=100))
    E0_2 = expectation_value(mps2, H2)
    mps3, = find_groundstate(mps3, H3, VUMPS(; maxiter=100))
    E0_3 = expectation_value(mps3, H3)

    # Compress MPO
    mps1c = InfiniteMPS(fill(ℂ^2, M), fill(ℂ^D, M))
    mps2c = InfiniteMPS(fill(Z2Space(0=>1, 1=>1), M), fill(Z2Space(0=>D, 1=>D), M))
    mps3c = InfiniteMPS(fill(Vect[FermionParity](0=>1, 1=>1), M), fill(Vect[FermionParity](
        0=>D, 1=>D), M))

    H1c, _ = mpo_compression(H1, 10^-10)
    H2c, _ = mpo_compression(H2, 10^-10)
    H3c, _ = mpo_compression(H3, 10^-10)

    # Find groundstate of compressed Hamiltonian
    mps1c, = find_groundstate(mps1c, H1c, VUMPS(; maxiter=100))
    E0_1c = expectation_value(mps1c, H1c)
    mps2c, = find_groundstate(mps2c, H2c, VUMPS(; maxiter=100))
    E0_2c = expectation_value(mps2c, H2c)
    mps3c, = find_groundstate(mps3c, H3c, VUMPS(; maxiter=100))
    E0_3c = expectation_value(mps3c, H3c)

    # Assert that the ground state energies are equal up to a precision, display the 
    # compression
    @assert E0_1 ≈ E0_1c
    @assert E0_2 ≈ E0_2c
    @assert E0_3 ≈ E0_3c

    println("Max dim uncompressed H1: $(max_dim_ham(H1)) || Max dim compressed H1c: 
            $(max_dim_ham(H1c))" )
    println("Max dim uncompressed H2: $(max_dim_ham(H2)) || Max dim compressed H2c: 
            $(max_dim_ham(H2c))" )
    println("Max dim uncompressed H3: $(max_dim_ham(H3)) || Max dim compressed H3c: 
            $(max_dim_ham(H3c))" )

end

@testset "infinite_mpo_compression_kitaev_model_energy" begin
    M = 10
    D = 10 # Max bond dimension

    mps1 = InfiniteMPS(fill(Vect[FermionParity](0=>1, 1=>1), M), 
                       fill(Vect[FermionParity](0=>D, 1=>D), M))

    H1 = kitaev_model_infinite(M)

    # Find groundstate of non-compressed Hamiltonian
    mps1, = find_groundstate(mps1, H1, VUMPS(; maxiter=100))
    E0_1 = expectation_value(mps1, H1)

    # Compress MPO
    mps1c = InfiniteMPS(fill(Vect[FermionParity](0=>1, 1=>1), M), 
                       fill(Vect[FermionParity](0=>D, 1=>D), M))

    H1c, _ = mpo_compression(H1, 10^-10)

    # Find groundstate of compressed Hamiltonian
    mps1c, = find_groundstate(mps1c, H1c, VUMPS(; maxiter=100))
    E0_1c = expectation_value(mps1c, H1c)

    # Assert that the ground state energies are equal up to a precision, display the 
    # compression
    @assert E0_1 ≈ E0_1c

    println("Max dim uncompressed H1: $(max_dim_ham(H1)) || Max dim compressed H1c: 
            $(max_dim_ham(H1c))" )

end

@testset "infinite_mpo_compression_heisenberg_XXX_energy" begin
    M = 10
    D = 6 # Max bond dimension

    mps1 = InfiniteMPS(fill(ℂ^3, M),fill(ℂ^D, M))
    mps2 = InfiniteMPS(fill(U1Space(0=>1, 1=>1, -1=>1), M), fill(U1Space(0=>D, 1=>D, -1=>D), M))
    mps3 = InfiniteMPS(fill(SU2Space(1 => 1), M), fill(SU2Space(1=>D), M))

    H1 = heisenberg_XXX_trivial_infinite(M)
    H2 = heisenberg_XXX_U1_infinite(M)
    H3 = heisenberg_XXX_SU2_infinite(M)

    # Find groundstate of non-compressed Hamiltonian
    mps1, = find_groundstate(mps1, H1, VUMPS(; maxiter=100))
    E0_1 = expectation_value(mps1, H1)
    mps2, = find_groundstate(mps2, H2, VUMPS(; maxiter=100))
    E0_2 = expectation_value(mps2, H2)
    mps3, = find_groundstate(mps3, H3, VUMPS(; maxiter=100))
    E0_3 = expectation_value(mps3, H3)

    # Compress MPO
    mps1c = InfiniteMPS(fill(ℂ^3, M),fill(ℂ^D, M))
    mps2c = InfiniteMPS(fill(U1Space(0=>1, 1=>1, -1=>1), M), fill(U1Space(0=>D, 1=>D, -1=>D), M))
    mps3c = InfiniteMPS(fill(SU2Space(1 => 1), M), fill(SU2Space(1=>D), M))

    H1c, _ = mpo_compression(H1, 10^-10)
    H2c, _ = mpo_compression(H2, 10^-10)
    H3c, _ = mpo_compression(H3, 10^-10)

    # Find groundstate of compressed Hamiltonian
    mps1c, = find_groundstate(mps1c, H1c, VUMPS(; maxiter=100))
    E0_1c = expectation_value(mps1c, H1c)
    mps2c, = find_groundstate(mps2c, H2c, VUMPS(; maxiter=100))
    E0_2c = expectation_value(mps2c, H2c)
    mps3c, = find_groundstate(mps3c, H3c, VUMPS(; maxiter=100))
    E0_3c = expectation_value(mps3c, H3c)

    # Assert that the ground state energies are equal up to a precision, display the 
    # compression
    @assert E0_1 ≈ E0_1c
    @assert E0_2 ≈ E0_2c
    @assert E0_3 ≈ E0_3c

    println("Max dim uncompressed H1: $(max_dim_ham(H1)) || Max dim compressed H1c: 
            $(max_dim_ham(H1c))" )
    println("Max dim uncompressed H2: $(max_dim_ham(H2)) || Max dim compressed H2c: 
            $(max_dim_ham(H2c))" )
    println("Max dim uncompressed H3: $(max_dim_ham(H3)) || Max dim compressed H3c: 
            $(max_dim_ham(H3c))" )

end

@testset "infinite_mpo_compression_bilinear_biquadratic_model_energy" begin
    M = 10
    D = 6 # Max bond dimension

    mps1 = InfiniteMPS(fill(ℂ^3, M),fill(ℂ^D, M))


    H1 = bilinear_biquadratic_model_trivial_infinite(M)

    # Find groundstate of non-compressed Hamiltonian
    mps1, = find_groundstate(mps1, H1, VUMPS(; maxiter=100))
    E0_1 = expectation_value(mps1, H1)

    # Compress MPO
    mps1c = InfiniteMPS(fill(ℂ^3, M),fill(ℂ^D, M))

    H1c, _ = mpo_compression(H1, 10^-10)

    # Find groundstate of compressed Hamiltonian
    mps1c, = find_groundstate(mps1c, H1c, VUMPS(; maxiter=100))
    E0_1c = expectation_value(mps1c, H1c)

    # Assert that the ground state energies are equal up to a precision, display the 
    # compression
    @assert E0_1 ≈ E0_1c

    println("Max dim uncompressed H1: $(max_dim_ham(H1)) || Max dim compressed H1c: 
            $(max_dim_ham(H1c))" )

end

@testset "infinite_mpo_compression_quantum_potts_model_energy" begin
    M = 10
    D = 6 # Max bond dimension

    mps1 = InfiniteMPS(fill(ℂ^3, M), fill(ℂ^D, M))
    mps2 = InfiniteMPS(fill(Z3Space(0=>1, 1=>1, 2=>1), M), fill(Z3Space(0=>1, 1=>1, 2=>1), M))

    H1 = quantum_potts_trivial_infinite(M)
    H2 = quantum_potts_Z3_infinite(M)

    # Find groundstate of non-compressed Hamiltonian
    mps1, = find_groundstate(mps1, H1, VUMPS(; maxiter=100))
    E0_1 = expectation_value(mps1, H1)
    mps2, = find_groundstate(mps2, H2, VUMPS(; maxiter=100))
    E0_2 = expectation_value(mps2, H2)

    # Compress MPO
    mps1c = InfiniteMPS(fill(ℂ^3, M), fill(ℂ^D, M))
    mps2c = InfiniteMPS(fill(Z3Space(0=>1, 1=>1, 2=>1), M), fill(Z3Space(0=>1, 1=>1, 2=>1), M))

    H1c, _ = mpo_compression(H1, 10^-10)
    H2c, _ = mpo_compression(H2, 10^-10)

    # Find groundstate of compressed Hamiltonian
    mps1c, = find_groundstate(mps1c, H1c, VUMPS(; maxiter=100))
    E0_1c = expectation_value(mps1c, H1c)
    mps2c, = find_groundstate(mps2c, H2c, VUMPS(; maxiter=100))
    E0_2c = expectation_value(mps2c, H2c)

    # Assert that the ground state energies are equal up to a precision, display the 
    # compression
    @assert E0_1 ≈ E0_1c
    @assert E0_2 ≈ E0_2c

    println("Max dim uncompressed H1: $(max_dim_ham(H1)) || Max dim compressed H1c: 
            $(max_dim_ham(H1c))" )
    println("Max dim uncompressed H2: $(max_dim_ham(H2)) || Max dim compressed H2c: 
            $(max_dim_ham(H2c))" )

end

# Test random infinite Hamiltonian
@testset "infinite_random_long_range_ham" begin
    # test the finite compression for a long range random Ising model
    # check if the energy after the compression is the same as before
    Random.seed!()
    D = 20 # max bond dimensions
    L = 10 # number of sites
    range = 4 # cutoff for the maximum range between interactions

    mps = InfiniteMPS(fill(Z2Space(0=>1, 1=>1),L), fill(Z2Space(0=>D, 1=>D),L))
    H = create_long_range_ising_symmetries_infinite_msite_random(L, range, 0.1, 0.001)
    mps, = find_groundstate(mps, H, VUMPS(; maxiter=100))
    E0 = expectation_value(mps, H)
    #println("<mps|H|mps> = $real(E0)")

    mps2 = InfiniteMPS(fill(Z2Space(0=>1, 1=>1),L), fill(Z2Space(0=>D, 1=>D),L))
    H2, Rs = mpo_compression(H, 0)
    mps2, = find_groundstate(mps2, H2, VUMPS(; maxiter=100))
    E1 = expectation_value(mps2, H2)

    @assert E0 ≈ E1
end