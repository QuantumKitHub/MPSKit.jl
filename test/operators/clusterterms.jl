println("
--------------------
|   ClusterTerm tests  |
--------------------
")

using Test, TensorKit, MPSKit

"""
    embed_cluster(c::ClusterTerm, lattice)

Pad a `ClusterTerm`'s operator with identities on every site outside its
support, so it can be directly compared against/added to a dense `H`.
"""
function embed_cluster(c::ClusterTerm, lattice)
    op = c.op
    for j in reverse(1:(first(c.sites) - 1))
        op = id(lattice[j]) ⊗ op
    end
    for j in (last(c.sites) + 1):length(lattice)
        op = op ⊗ id(lattice[j])
    end
    return op
end

"""
    check_cluster_hamiltonians(H::FiniteMPOHamiltonian, lattice)

Reconstruct `H` from its extracted `ClusterTerm`s and compare against the
dense conversion of `H` itself. Returns `true` iff they match.
"""
function check_cluster_hamiltonians(H::FiniteMPOHamiltonian, lattice)
    clusters = cluster_hamiltonians(H)
    Hdense = convert(TensorMap, H)
    Hcheck = sum(embed_cluster(c, lattice) for c in clusters)
    return isapprox(Hdense, Hcheck)
end

@testset "cluster_hamiltonians" begin
    on_site = S_z()
    interaction_x = S_x_S_x()
    interaction_z = S_z_S_z()

    lattice = fill(ℂ^2, 5)

    @testset "on-site only" begin
        # exercises the cols == 1 boundary edge case at the last site directly
        H = FiniteMPOHamiltonian(lattice, (i,) => on_site for i in 1:length(lattice))
        @test check_cluster_hamiltonians(H, lattice)
    end

    @testset "nearest-neighbour only" begin
        H = FiniteMPOHamiltonian(lattice, (i, i + 1) => interaction_x for i in 1:(length(lattice) - 1))
        @test check_cluster_hamiltonians(H, lattice)
    end

    @testset "on-site + nearest-neighbour" begin
        H = FiniteMPOHamiltonian(lattice, (i,) => on_site for i in 1:length(lattice)) +
            FiniteMPOHamiltonian(lattice, (i, i + 1) => interaction_x for i in 1:(length(lattice) - 1))
        @test check_cluster_hamiltonians(H, lattice)
    end

    @testset "next-nearest-neighbour" begin
        # exercises multi-hop tracing through _trace_cluster!
        H = FiniteMPOHamiltonian(lattice, (i, i + 2) => interaction_x for i in 1:(length(lattice) - 2))
        @test check_cluster_hamiltonians(H, lattice)
    end

    @testset "mixed NN + NNN + on-site" begin
        # the real stress test: overlapping ranges, multiple simultaneous channels
        H = FiniteMPOHamiltonian(lattice, (i,) => on_site for i in 1:length(lattice)) +
            FiniteMPOHamiltonian(lattice, (i, i + 1) => interaction_x for i in 1:(length(lattice) - 1)) +
            FiniteMPOHamiltonian(lattice, (i, i + 2) => interaction_z for i in 1:(length(lattice) - 2))
        @test check_cluster_hamiltonians(H, lattice)
    end
end
