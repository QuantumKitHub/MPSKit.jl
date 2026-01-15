using MPSKit, TensorKit, MPSKitModels

function rydberg_model(
        elt::Type{<:Number} = Float64,
        lattice::AbstractLattice = InfiniteChain(1);
        Delta = 0.0, Omega = 0.0, V = 1.0, fluctuation = 0
    )
    sz = S_z(elt, Trivial)
    Id = id(elt, domain(sz))
    n = 0.5 * Id + sz
    #Id = 2 * (n - sz)
    sx = S_x(elt, Trivial)

    H = @mpoham begin
        sum(vertices(lattice)) do i
            return sum(V / j^6 * n{i} * n{i - j} for j in 1:10)
        end +
            sum(vertices(lattice)) do i
            return -Delta * (1 + fluctuation * rand()) * n{i} - Omega * (1 + fluctuation * rand()) * sx{i} - 1 * Id{i}
        end
    end
    return H
end

L = 20;
x = 4;
y = 4;
Omega = 1 / y^6
Delta = x * Omega
H0 = rydberg_model(Float64, InfiniteChain(2); Omega, Delta);
H = periodic_boundary_conditions(H0, L);

# Htrunc = changebonds(H, SvdCut(; trscheme = truncbelow(1.0e-14)))

mps = FiniteMPS([TensorMap(rand, ComplexF64, ComplexSpace(2) ⊗ ComplexSpace(2), ComplexSpace(2)) for j in 1:L])
# mps = normalize!(FiniteMPS(rand, ComplexF64, fill(ℂ^2, L), ℂ^2))

# alg = DMRG2(; trscheme = truncdim(10), maxiter = 5, verbosity = 3, alg_eigsolve = (; verbosity = 2, dynamic_tols = false, tol = 1.0e-6))
# gs, envs, delta = find_groundstate(mps, H, alg)


mps2 = approximate(mps, (H, mps), DMRG2(trscheme = truncbelow(1.0e-12)))
