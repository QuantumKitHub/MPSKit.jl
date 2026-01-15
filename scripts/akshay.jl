using MPSKit, TensorKit
using TensorKitTensors.BosonOperators
using Test

T = Float64
cutoff = 3
adag = b_plus(T, Trivial; cutoff)
a = b_min(T, Trivial; cutoff)


N = 6
U = rand(N, N, N, N);
aaaa = FiniteMPO(adag ⊗ adag ⊗ a ⊗ a);
lattice = fill(BosonOperators.boson_space(Trivial; cutoff), N)


four_site_terms = Pair{NTuple{4, Int64}, typeof(aaaa)}[]
for l in 1:N, m in 1:N, n in 1:N, o in 1:N
    # allunique((l, m, n, o)) || continue
    # ((l == m) || (m == n) || (n == o) || (o == l)) && continue
    push!(four_site_terms, (l, m, n, o) => U[l, m, n, o] * aaaa)
end

H = FiniteMPOHamiltonian(lattice, four_site_terms);
@testset "Finite MPOHamiltonian repeated indices" begin
    X = adag
    Y = adag'
    L = 4
    chain = fill(space(X, 1), 4)

    H1 = FiniteMPOHamiltonian(chain, (1,) => (X * X * Y * Y))
    H2 = FiniteMPOHamiltonian(chain, (1, 1, 1, 1) => (X ⊗ X ⊗ Y ⊗ Y))
    @test convert(TensorMap, H1) ≈ convert(TensorMap, H2)

    H1 = FiniteMPOHamiltonian(chain, (1, 2) => ((X * Y) ⊗ (X * Y)))
    H2 = FiniteMPOHamiltonian(chain, (1, 2, 1, 2) => (X ⊗ X ⊗ Y ⊗ Y))
    @test convert(TensorMap, H1) ≈ convert(TensorMap, H2)

    H1 = FiniteMPOHamiltonian(chain, (1, 2) => ((X * X * Y) ⊗ Y))
    H2 = FiniteMPOHamiltonian(chain, (1, 1, 1, 2) => (X ⊗ X ⊗ Y ⊗ Y))
    @test convert(TensorMap, H1) ≈ convert(TensorMap, H2)

    H1 = FiniteMPOHamiltonian(chain, (1, 2) => ((X * Y * Y) ⊗ X))
    H2 = FiniteMPOHamiltonian(chain, (1, 2, 1, 1) => (X ⊗ X ⊗ Y ⊗ Y))
    @test convert(TensorMap, H1) ≈ convert(TensorMap, H2)

    H1 = FiniteMPOHamiltonian(chain, (1, 2, 3) => FiniteMPO((X * X) ⊗ Y ⊗ Y))
    H2 = FiniteMPOHamiltonian(chain, (1, 1, 2, 3) => FiniteMPO(X ⊗ X ⊗ Y ⊗ Y))
    @test convert(TensorMap, H1) ≈ convert(TensorMap, H2)

    H1 = FiniteMPOHamiltonian(chain, (1, 2, 3) => FiniteMPO((Y * Y) ⊗ X ⊗ X))
    H2 = FiniteMPOHamiltonian(chain, (2, 3, 1, 1) => FiniteMPO(X ⊗ X ⊗ Y ⊗ Y))
    @test convert(TensorMap, H1) ≈ convert(TensorMap, H2)

    H1 = FiniteMPOHamiltonian(chain, (1, 2, 3) => FiniteMPO(X ⊗ (X * Y) ⊗ Y))
    H2 = FiniteMPOHamiltonian(chain, (1, 2, 2, 3) => FiniteMPO(X ⊗ X ⊗ Y ⊗ Y))
    @test convert(TensorMap, H1) ≈ convert(TensorMap, H2)
end

function hamiltonian_int(N, cutoff, U) # interaction
    adag = a_plus(ComplexF64, Trivial; cutoff = cutoff)
    a = a_min(ComplexF64, Trivial; cutoff = cutoff)

    ops = [
        FiniteMPO(adag * adag * a * a),
        FiniteMPO((adag * adag) ⊗ (a * a)),
        FiniteMPO((adag * a) ⊗ (adag * a)),
        FiniteMPO((adag * adag * a) ⊗ a),
        FiniteMPO((adag * a * a) ⊗ adag),
        FiniteMPO((adag * adag) ⊗ a ⊗ a), #
        FiniteMPO((a * a) ⊗ adag ⊗ adag), #
        FiniteMPO((adag * a) ⊗ adag ⊗ a),
        FiniteMPO(adag ⊗ adag ⊗ a ⊗ a),
    ]

    four_site_terms = Pair{NTuple{4, Int64}, FiniteMPO{TensorMap{ComplexF64, ComplexSpace, 2, 2, Vector{ComplexF64}}}}[]
    for l in 1:N, m in 1:N, n in 1:N, o in 1:N
        ((l == m) || (m == n) || (n == o) || (o == l)) && continue
        push!(four_site_terms, (l, m, n, o) => U[l, m, n, o] * ops[9])
    end

    three_site_terms = Pair{NTuple{3, Int64}, FiniteMPO{TensorMap{ComplexF64, ComplexSpace, 2, 2, Vector{ComplexF64}}}}[]
    for l in 1:N, m in 1:N, n in 1:N
        ((l == m) || (m == n) || (n == l)) && continue
        push!(
            three_site_terms,
            (l, m, n) => (
                (U[l, m, l, n] + U[l, m, n, l] + U[m, l, n, l] + U[m, l, l, n]) * ops[8] +
                    U[l, l, m, n] * ops[7] +
                    U[m, n, l, l] * ops[6]
            )
        )
    end

    two_site_terms = Pair{NTuple{2, Int64}, FiniteMPO{TensorMap{ComplexF64, ComplexSpace, 2, 2, Vector{ComplexF64}}}}[]
    for l in 1:N, m in 1:N
        (l == m) && continue
        push!(
            two_site_terms,
            (l, m) => (
                (U[l, m, l, l] + U[m, l, l, l]) * ops[5] +
                    (U[l, l, l, m] + U[l, l, m, l]) * ops[4] +
                    (U[l, m, l, m] + U[l, m, m, l]) * ops[3] +
                    U[l, l, m, m] * ops[2]
            )
        )
    end

    one_site_terms = Pair{NTuple{1, Int64}, FiniteMPO{TensorMap{ComplexF64, ComplexSpace, 2, 2, Vector{ComplexF64}}}}[]

    for l in 1:N
        push!(one_site_terms, (l,) => U[l, l, l, l] * ops[1])
    end

    return FiniteMPOHamiltonian(fill(ℂ^(cutoff + 1), N), one_site_terms..., two_site_terms..., three_site_terms..., four_site_terms...)
end


using TensorKit
using TensorKitTensors.SpinOperators

Sp = S_plus()
Sm = S_min()
Sx = S_x()
Sy = S_y()
Sz = S_z()


H1 = Sx ⊗ Sx - Sy ⊗ Sy + Sz ⊗ Sz
H2 = (Sp ⊗ Sm + Sm ⊗ Sp) / 2 + Sz ⊗ Sz

H1 ≈ H2
