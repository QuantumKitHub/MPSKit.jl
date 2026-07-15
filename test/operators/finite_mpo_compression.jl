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

function create_long_range_ising_symmetries_finite(M, k)
    chain = fill(Z2Space(0 => 1, 1 => 1), M)
    ZZ = S_zz(Z2Irrep)
    X = S_x(Z2Irrep)
    single_site_operators = [i => X for i in 1:M]
    two_site_operators = [(i, i + j) => ZZ for i in 1:M for j in 1:k if i + j ≤ M]
    H = FiniteMPOHamiltonian(
        chain, single_site_operators...,
        two_site_operators...
    )
    return H
end

function create_long_range_ising_symmetries_random_finite(M, k, σⱼ, σₕ)
    chain = fill(Z2Space(0 => 1, 1 => 1), M)
    ZZ = S_zz(Z2Irrep)
    X = S_x(Z2Irrep)
    single_site_operators = [i => (σₕ * randn()) * X for i in 1:M]
    two_site_operators = [(i, i + j) => (1 + σⱼ * randn()) * ZZ for i in 1:M for j in 1:k if i + j ≤ M]
    H = FiniteMPOHamiltonian(
        chain, single_site_operators...,
        two_site_operators...
    )
    return H
end

function transverse_field_ising_fermion_parity_finite(M)
    chain = FiniteChain(M)
    return transverse_field_ising(ComplexF64, FermionParity, chain)
end

function transverse_field_ising_z2_finite(M)
    chain = FiniteChain(M)
    return transverse_field_ising(ComplexF64, Z2Irrep, chain)
end

function transverse_field_ising_trivial_finite(M)
    chain = FiniteChain(M)
    return transverse_field_ising(ComplexF64, Trivial, chain)
end

function kitaev_model_finite(M)
    chain = FiniteChain(M)
    return kitaev_model(ComplexF64, chain)
end

function heisenberg_XXX_trivial_finite(M)
    chain = FiniteChain(M)
    return heisenberg_XXX(Trivial, chain)
end

function heisenberg_XXX_U1_finite(M)
    chain = FiniteChain(M)
    return heisenberg_XXX(U1Irrep, chain)
end

function heisenberg_XXX_SU2_finite(M)
    chain = FiniteChain(M)
    return heisenberg_XXX(SU2Irrep, chain)
end

function bilinear_biquadratic_model_trivial_finite(M)
    chain = FiniteChain(M)
    return bilinear_biquadratic_model(ComplexF64, Trivial, chain)
end

function bilinear_biquadratic_model_U1_finite(M)
    chain = FiniteChain(M)
    return bilinear_biquadratic_model(ComplexF64, U1Irrep, chain)
end

function bilinear_biquadratic_model_SU2_finite(M)
    chain = FiniteChain(M)
    return bilinear_biquadratic_model(ComplexF64, SU2Irrep, chain)
end

function quantum_potts_trivial_finite(M)
    chain = FiniteChain(M)
    return quantum_potts(ComplexF64, Trivial, chain)
end

function quantum_potts_Z3_finite(M)
    chain = FiniteChain(M)
    return quantum_potts(ComplexF64, Z3Irrep, chain)
end

# Tests

@testset "finite_mpo_compression_energy" begin
    # test the finite compression for a long range Ising model
    # check if the energy after the compression is the same as before
    D = 20 # max bond dimensions
    L = 10 # number of sites
    range = 4 # cutoff for the maximum range between interactions
    trunc_st = notrunc()

    mps = FiniteMPS(L, Z2Space(0 => 1, 1 => 1), Z2Space(0 => D, 1 => D))
    H = create_long_range_ising_symmetries_finite(L, range)
    find_groundstate!(mps, H, DMRG2(; maxiter = 100, trscheme = trunc_st))
    E0 = expectation_value(mps, H)
    #println("<mps|H|mps> = $real(E0)")

    mps2 = FiniteMPS(L, Z2Space(0 => 1, 1 => 1), Z2Space(0 => D, 1 => D))
    H2, Rs = mpo_compression(H)
    find_groundstate!(mps2, H2, DMRG2(; maxiter = 100, trscheme = trunc_st))
    E1 = expectation_value(mps2, H2)
end

@testset "finite_mpo_compression_transverse_ising_model_energy" begin
    M = 10
    D = 10 # Max bond dimension
    trunc_st = trunctol()

    mps1 = FiniteMPS(M, ℂ^2, ℂ^D)
    mps2 = FiniteMPS(M, Z2Space(0 => 1, 1 => 1), Z2Space(0 => D, 1 => D))
    mps3 = FiniteMPS(M, Vect[FermionParity](0 => 1, 1 => 1), Vect[FermionParity](0 => D, 1 => D))

    H1 = transverse_field_ising_trivial_finite(M)
    H2 = transverse_field_ising_z2_finite(M)
    H3 = transverse_field_ising_fermion_parity_finite(M)

    # Find groundstate of non-compressed Hamiltonian
    find_groundstate!(mps1, H1, DMRG2(; maxiter = 100, trscheme = trunc_st))
    E0_1 = expectation_value(mps1, H1)
    find_groundstate!(mps2, H2, DMRG2(; maxiter = 100, trscheme = trunc_st))
    E0_2 = expectation_value(mps2, H2)
    find_groundstate!(mps3, H3, DMRG2(; maxiter = 100, trscheme = trunc_st))
    E0_3 = expectation_value(mps3, H3)

    # Compress MPO
    mps1c = FiniteMPS(M, ℂ^2, ℂ^D)
    mps2c = FiniteMPS(M, Z2Space(0 => 1, 1 => 1), Z2Space(0 => D, 1 => D))
    mps3c = FiniteMPS(M, Vect[FermionParity](0 => 1, 1 => 1), Vect[FermionParity](0 => D, 1 => D))

    H1c, _ = mpo_compression(H1, 10^-10)
    H2c, _ = mpo_compression(H2, 10^-10)
    H3c, _ = mpo_compression(H3, 10^-10)

    # Find groundstate of compressed Hamiltonian
    find_groundstate!(mps1c, H1c, DMRG2(; maxiter = 100, trscheme = trunc_st))
    E0_1c = expectation_value(mps1c, H1c)
    find_groundstate!(mps2c, H2c, DMRG2(; maxiter = 100, trscheme = trunc_st))
    E0_2c = expectation_value(mps2c, H2c)
    find_groundstate!(mps3c, H3c, DMRG2(; maxiter = 100, trscheme = trunc_st))
    E0_3c = expectation_value(mps3c, H3c)

    # Assert that the ground state energies are equal up to a precision, display the
    # compression
    @assert E0_1 ≈ E0_1c
    @assert E0_2 ≈ E0_2c
    @assert E0_3 ≈ E0_3c

    println("Max dim uncompressed H1: $(max_dim_ham(H1)) || Max dim compressed H1c: 
            $(max_dim_ham(H1c))")
    println("Max dim uncompressed H2: $(max_dim_ham(H2)) || Max dim compressed H2c: 
            $(max_dim_ham(H2c))")
    println("Max dim uncompressed H3: $(max_dim_ham(H3)) || Max dim compressed H3c: 
            $(max_dim_ham(H3c))")

end

@testset "finite_mpo_compression_kitaev_model_energy" begin
    M = 10
    D = 10 # Max bond dimension
    trunc_st = trunctol()

    mps1 = FiniteMPS(M, Vect[FermionParity](0 => 1, 1 => 1), Vect[FermionParity](0 => D, 1 => D))

    H1 = kitaev_model_finite(M)

    # Find groundstate of non-compressed Hamiltonian
    find_groundstate!(mps1, H1, DMRG2(; maxiter = 100, trscheme = trunc_st))
    E0_1 = expectation_value(mps1, H1)

    # Compress MPO
    mps1c = FiniteMPS(M, Vect[FermionParity](0 => 1, 1 => 1), Vect[FermionParity](0 => D, 1 => D))

    H1c, _ = mpo_compression(H1, 10^-10)

    # Find groundstate of compressed Hamiltonian
    find_groundstate!(mps1c, H1c, DMRG2(; maxiter = 100, trscheme = trunc_st))
    E0_1c = expectation_value(mps1c, H1c)

    # Assert that the ground state energies are equal up to a precision, display the
    # compression
    @assert E0_1 ≈ E0_1c

    println("Max dim uncompressed H1: $(max_dim_ham(H1)) || Max dim compressed H1c: 
            $(max_dim_ham(H1c))")

end

@testset "finite_mpo_compression_heisenberg_XXX_energy" begin
    M = 10
    D = 6 # Max bond dimension
    trunc_st = trunctol()

    mps1 = FiniteMPS(M, ℂ^3, ℂ^D)
    mps2 = FiniteMPS(M, U1Space(0 => 1, 1 => 1, -1 => 1), U1Space(0 => D, 1 => D, -1 => D))
    mps3 = FiniteMPS(M, SU2Space(1 => 1), SU2Space(1 => D))

    H1 = heisenberg_XXX_trivial_finite(M)
    H2 = heisenberg_XXX_U1_finite(M)
    H3 = heisenberg_XXX_SU2_finite(M)

    # Find groundstate of non-compressed Hamiltonian
    find_groundstate!(mps1, H1, DMRG2(; maxiter = 100, trscheme = trunc_st))
    E0_1 = expectation_value(mps1, H1)
    find_groundstate!(mps2, H2, DMRG2(; maxiter = 100, trscheme = trunc_st))
    E0_2 = expectation_value(mps2, H2)
    find_groundstate!(mps3, H3, DMRG2(; maxiter = 100, trscheme = trunc_st))
    E0_3 = expectation_value(mps3, H3)

    # Compress MPO
    mps1c = FiniteMPS(M, ℂ^3, ℂ^D)
    mps2c = FiniteMPS(M, U1Space(0 => 1, 1 => 1, -1 => 1), U1Space(0 => D, 1 => D, -1 => D))
    mps3c = FiniteMPS(M, SU2Space(1 => 1), SU2Space(1 => D))

    H1c, _ = mpo_compression(H1, 10^-10)
    H2c, _ = mpo_compression(H2, 10^-10)
    H3c, _ = mpo_compression(H3, 10^-10)

    # Find groundstate of compressed Hamiltonian
    find_groundstate!(mps1c, H1c, DMRG2(; maxiter = 100, trscheme = trunc_st))
    E0_1c = expectation_value(mps1c, H1c)
    find_groundstate!(mps2c, H2c, DMRG2(; maxiter = 100, trscheme = trunc_st))
    E0_2c = expectation_value(mps2c, H2c)
    find_groundstate!(mps3c, H3c, DMRG2(; maxiter = 100, trscheme = trunc_st))
    E0_3c = expectation_value(mps3c, H3c)

    # Assert that the ground state energies are equal up to a precision, display the
    # compression
    @assert E0_1 ≈ E0_1c
    @assert E0_2 ≈ E0_2c
    @assert E0_3 ≈ E0_3c

    println("Max dim uncompressed H1: $(max_dim_ham(H1)) || Max dim compressed H1c: 
            $(max_dim_ham(H1c))")
    println("Max dim uncompressed H2: $(max_dim_ham(H2)) || Max dim compressed H2c: 
            $(max_dim_ham(H2c))")
    println("Max dim uncompressed H3: $(max_dim_ham(H3)) || Max dim compressed H3c: 
            $(max_dim_ham(H3c))")

end

@testset "finite_mpo_compression_bilinear_biquadratic_model_energy" begin
    M = 10
    D = 6 # Max bond dimension
    trunc_st = trunctol()

    mps1 = FiniteMPS(M, ℂ^3, ℂ^D)
    mps2 = FiniteMPS(M, U1Space(0 => 1, 1 => 1, -1 => 1), U1Space(0 => D, 1 => D, -1 => D))
    mps3 = FiniteMPS(M, SU2Space(1 => 1), SU2Space(1 => D))

    H1 = bilinear_biquadratic_model_trivial_finite(M)
    H2 = bilinear_biquadratic_model_U1_finite(M)
    H3 = bilinear_biquadratic_model_SU2_finite(M)

    # Find groundstate of non-compressed Hamiltonian
    find_groundstate!(mps1, H1, DMRG2(; maxiter = 100, trscheme = trunc_st))
    E0_1 = expectation_value(mps1, H1)
    find_groundstate!(mps2, H2, DMRG2(; maxiter = 100, trscheme = trunc_st))
    E0_2 = expectation_value(mps2, H2)
    find_groundstate!(mps3, H3, DMRG2(; maxiter = 100, trscheme = trunc_st))
    E0_3 = expectation_value(mps3, H3)

    # Compress MPO
    mps1c = FiniteMPS(M, ℂ^3, ℂ^D)
    mps2c = FiniteMPS(M, U1Space(0 => 1, 1 => 1, -1 => 1), U1Space(0 => D, 1 => D, -1 => D))
    mps3c = FiniteMPS(M, SU2Space(1 => 1), SU2Space(1 => D))

    H1c, _ = mpo_compression(H1, 10^-10)
    H2c, _ = mpo_compression(H2, 10^-10)
    H3c, _ = mpo_compression(H3, 10^-10)

    # Find groundstate of compressed Hamiltonian
    find_groundstate!(mps1c, H1c, DMRG2(; maxiter = 100, trscheme = trunc_st))
    E0_1c = expectation_value(mps1c, H1c)
    find_groundstate!(mps2c, H2c, DMRG2(; maxiter = 100, trscheme = trunc_st))
    E0_2c = expectation_value(mps2c, H2c)
    find_groundstate!(mps3c, H3c, DMRG2(; maxiter = 100, trscheme = trunc_st))
    E0_3c = expectation_value(mps3c, H3c)

    # Assert that the ground state energies are equal up to a precision, display the
    # compression
    @assert E0_1 ≈ E0_1c
    @assert E0_2 ≈ E0_2c
    @assert E0_3 ≈ E0_3c

    println("Max dim uncompressed H1: $(max_dim_ham(H1)) || Max dim compressed H1c: 
            $(max_dim_ham(H1c))")
    println("Max dim uncompressed H2: $(max_dim_ham(H2)) || Max dim compressed H2c: 
            $(max_dim_ham(H2c))")
    println("Max dim uncompressed H3: $(max_dim_ham(H3)) || Max dim compressed H3c: 
            $(max_dim_ham(H3c))")

end

@testset "finite_mpo_compression_quantum_potts_model_energy" begin
    M = 10
    D = 6 # Max bond dimension
    trunc_st = trunctol()

    mps1 = FiniteMPS(M, ℂ^3, ℂ^D)
    mps2 = FiniteMPS(M, Z3Space(0 => 1, 1 => 1, 2 => 1), Z3Space(0 => 1, 1 => 1, 2 => 1))

    H1 = quantum_potts_trivial_finite(M)
    H2 = quantum_potts_Z3_finite(M)

    # Find groundstate of non-compressed Hamiltonian
    find_groundstate!(mps1, H1, DMRG2(; maxiter = 100, trscheme = trunc_st))
    E0_1 = expectation_value(mps1, H1)
    find_groundstate!(mps2, H2, DMRG2(; maxiter = 100, trscheme = trunc_st))
    E0_2 = expectation_value(mps2, H2)

    # Compress MPO
    mps1c = FiniteMPS(M, ℂ^3, ℂ^D)
    mps2c = FiniteMPS(M, Z3Space(0 => 1, 1 => 1, 2 => 1), Z3Space(0 => 1, 1 => 1, 2 => 1))

    H1c, _ = mpo_compression(H1, 10^-10)
    H2c, _ = mpo_compression(H2, 10^-10)

    # Find groundstate of compressed Hamiltonian
    find_groundstate!(mps1c, H1c, DMRG2(; maxiter = 100, trscheme = trunc_st))
    E0_1c = expectation_value(mps1c, H1c)
    find_groundstate!(mps2c, H2c, DMRG2(; maxiter = 100, trscheme = trunc_st))
    E0_2c = expectation_value(mps2c, H2c)

    # Assert that the ground state energies are equal up to a precision, display the
    # compression
    @assert E0_1 ≈ E0_1c
    @assert E0_2 ≈ E0_2c

    println("Max dim uncompressed H1: $(max_dim_ham(H1)) || Max dim compressed H1c: 
            $(max_dim_ham(H1c))")
    println("Max dim uncompressed H2: $(max_dim_ham(H2)) || Max dim compressed H2c: 
            $(max_dim_ham(H2c))")

end

# Test random finite Hamiltonian
@testset "finite_random_long_range_ham" begin
    # test the finite compression for a long range random Ising model
    # check if the energy after the compression is the same as before
    Random.seed!()
    D = 20 # max bond dimensions
    L = 10 # number of sites
    range = 4 # cutoff for the maximum range between interactions
    trunc_st = notrunc()

    mps = FiniteMPS(L, Z2Space(0 => 1, 1 => 1), Z2Space(0 => D, 1 => D))
    H = create_long_range_ising_symmetries_random_finite(L, range, 0.3, 0.4)
    find_groundstate!(mps, H, DMRG2(; maxiter = 100, trscheme = trunc_st))
    E0 = expectation_value(mps, H)
    #println("<mps|H|mps> = $real(E0)")

    mps2 = FiniteMPS(L, Z2Space(0 => 1, 1 => 1), Z2Space(0 => D, 1 => D))
    H2, Rs = mpo_compression(H)
    find_groundstate!(mps2, H2, DMRG2(; maxiter = 100, trscheme = trunc_st))
    E1 = expectation_value(mps2, H2)
end
