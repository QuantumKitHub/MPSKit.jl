using LinearAlgebra: BLAS
using BenchmarkTools
using TensorKit, MPSKit, MPSKitModels
using CairoMakie

BLAS.set_num_threads(1)

# Parameters
L = 60
T = ComplexF64
sval = 1e-10
symmetry = SU2Irrep
bench_iter = 5

# derived
@assert iseven(L)
lattice = FiniteChain(L)
alg = DMRG2(; trscheme=truncdim(2048), tol=1e-8, verbosity=MPSKit.VERBOSE_ITER)

# Initialize MPS
H = heisenberg_XXX(T, symmetry, lattice; spin=1 // 2)

function init_mps(H)
    T = scalartype(H)
    P = physicalspace.(Ref(H), 1:length(H))
    V = repeat([P[1], fuse(P[1]^2)], (length(H) ÷ 2))
    return FiniteMPS(rand, T, P, V[1:(end - 1)])
end

psi = init_mps(H);
psi, = find_groundstate(psi, H, alg);

# Benchmark
site = L ÷ 2
Ds = reverse([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
times_sparsempo = let ψ = copy(psi), H = FiniteMPO(parent(H))
    times = map(Ds) do D
        ψ = changebonds(ψ, SvdCut(; trscheme=truncdim(D)))
        envs = environments(ψ, H)
        h_ac = MPSKit.∂∂AC(site, ψ, H, envs)
        ac = ψ.AC[site]
        if bench_iter <= 0
            t = @belapsed $h_ac($ac)
        else
            ts = map(1:bench_iter) do i
                return @elapsed h_ac(ac)
            end
            t = minimum(ts)
        end
        @info "Benchmark MPO:" D t
        return t
    end
end

times_jordanmpo = let ψ = copy(psi)
    times = map(Ds) do D
        ψ = changebonds(ψ, SvdCut(; trscheme=truncdim(D)))
        envs = environments(ψ, H)
        h_ac = MPSKit.∂∂AC(site, ψ, H, envs)
        ac = ψ.AC[site]
        if bench_iter <= 0
            t = @belapsed $h_ac($ac)
        else
            ts = map(1:bench_iter) do i
                return @elapsed h_ac(ac)
            end
            t = minimum(ts)
        end
        @info "Benchmark Jordan MPO:" D t
        return t
    end
end

# Plot results
# ------------
f = let f = Figure(; title="Absolute time of local application")
    ax = f[1, 1] = Axis(f; xlabel="D", ylabel="time (s)", xscale=log10, yscale=log10)
    positions = Ds
    lines!(positions, times_sparsempo; label="Sparse MPO")
    lines!(positions, times_jordanmpo; label="Jordan MPO")
    f[1, 2] = Legend(f, ax, "Operators"; framevisible=false)
    f
end
save("results_$symmetry.png", f)
