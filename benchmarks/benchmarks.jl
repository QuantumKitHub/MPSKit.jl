using MKL
using ThreadPinning
using LinearAlgebra: BLAS
using BenchmarkTools
using TensorKit, MPSKit, MPSKitModels
using CairoMakie

BLAS.set_num_threads(1)

ThreadPinning.threadinfo()
ThreadPinning.threadinfo(; blas=true)

bench_iter = 4 # set <= 0 to use benchmarktools
check_dense = true
check_nearestneighbour = true
check_jordanmpo = true

# settings
T = ComplexF64
N = 60
maxiter = 5 # number of sweeps
verbosity = MPSKit.VERBOSE_ITER
sval = 1e-6 # cut in spectrum

# Utility
# -------
function double_heisenberg_term(::Type{T}=ComplexF64,
                                symmetry=SU2Irrep ⊠ SU2Irrep) where {T<:Number}
    P = Vect[symmetry]((1 // 2, 1 // 2) => 1)
    V1 = Vect[symmetry]((1, 0) => 1)
    V2 = Vect[symmetry]((0, 1) => 1)

    HL1 = TensorMap(ones, T, P ← P ⊗ V1)
    HR1 = TensorMap(ones, T, V1 ⊗ P ← P)
    @tensor H1[-1 -2; -3 -4] := -HL1[-1; -3 3] * HR1[3 -2; -4]

    HL2 = TensorMap(ones, T, P ← P ⊗ V2)
    HR2 = TensorMap(ones, T, V2 ⊗ P ← P)
    @tensor H2[-1 -2; -3 -4] := -HL2[-1; -3 3] * HR2[3 -2; -4]

    return H1 + H2
end

function double_heisenberg_hamiltonian(::Type{T}, symmetry=SU2Irrep ⊠ SU2Irrep;
                                       N=60) where {T}
    lattice = FiniteChain(N)
    H = double_heisenberg_term(T, symmetry)
    return @mpoham sum(H{i,j} for (i, j) in nearest_neighbours(lattice))
end

alg = DMRG2(; eigalg=(; eager=true, verbosity=1, tol_min=1e-12), maxiter, verbosity,
            trscheme=TensorKit.truncbelow(sval), tol=1e-15);
sym = SU2Irrep ⊠ SU2Irrep

# Initialization
# --------------
H = double_heisenberg_hamiltonian(T; N)

V = Vect[sym]((0, 0) => 1, (0, 1 // 2) => 1, (1 // 2, 0) => 1)
V = fuse(V^4)
P = physicalspace(H, 1)
MPS_init = FiniteMPS(randn, T, N, P, V; left=Vect[sym]((0, 0) => 1),
                     right=Vect[sym]((0, 0) => 1))
temp, _ = find_groundstate(MPS_init, H, alg);

# check twosite actually is correct:
H2 = NN_Hamiltonian(double_heisenberg_term(T, sym))
temp2, _ = find_groundstate(MPS_init, H2, alg);
@info "overlap" dot(temp, temp2)

# Baseline
# --------
sites = 1:(N - 1)
times_sparsempo = let ψ = copy(temp)
    envs = environments(ψ, H)
    times = map(sites) do pos
        h_ac2 = MPSKit.∂∂AC2(pos, ψ, H, envs)
        @plansor ac2[-1 -2; -3 -4] := ψ.AC[pos][-1 -2; 1] * ψ.AR[pos + 1][1 -4; -3]
        if bench_iter <= 0
            t = @belapsed $h_ac2($ac2)
        else
            ts = map(1:bench_iter) do i
                return @elapsed h_ac2(ac2)
            end
            t = minimum(ts)
        end
        @info "position: $pos, timing: $t"
        return t
    end
end

if check_dense
    times_densempo = let ψ = copy(temp), H = MPSKit.DenseMPO(H)
        envs = environments(ψ, H)
        times = map(sites) do pos
            h_ac2 = MPSKit.∂∂AC2(pos, ψ, H, envs)
            @plansor ac2[-1 -2; -3 -4] := ψ.AC[pos][-1 -2; 1] * ψ.AR[pos + 1][1 -4; -3]
            if bench_iter <= 0
                t = @belapsed $h_ac2($ac2)
            else
                ts = map(1:bench_iter) do i
                    return @elapsed h_ac2(ac2)
                end
                t = minimum(ts)
            end
            @info "position: $pos, timing: $t"
            return t
        end
    end
end

# Attempt I
# ---------
if check_jordanmpo
    times_jordan = let ψ = copy(temp), H1 = H
        envs = environments(ψ, H1)
        Ws = MPSKit.JordanMPOTensor.(H1)
        H = FiniteMPO{eltype(Ws)}(Ws)

        times = map(sites) do pos
            h_ac2 = MPSKit.∂∂AC2(pos, ψ, H, envs)
            @plansor ac2[-1 -2; -3 -4] := ψ.AC[pos][-1 -2; 1] * ψ.AR[pos + 1][1 -4; -3]
            if bench_iter <= 0
                t = @belapsed $h_ac2($ac2)
            else
                ts = map(1:bench_iter) do i
                    return @elapsed h_ac2(ac2)
                end
                t = minimum(ts)
            end
            @info "position: $pos, timing: $t"
            return t
        end
    end
end

# Attempt II
# ----------
if check_nearestneighbour
    times_nn = let ψ = copy(temp)
        H = NN_Hamiltonian(double_heisenberg_term(T, sym))
        envs = environments(ψ, H)
        times = map(sites) do pos
            h_ac2 = MPSKit.∂∂AC2(pos, ψ, H, envs)
            @plansor ac2[-1 -2; -3 -4] := ψ.AC[pos][-1 -2; 1] * ψ.AR[pos + 1][1 -4; -3]
            if bench_iter <= 0
                t = @belapsed $h_ac2($ac2)
            else
                ts = map(1:bench_iter) do i
                    return @elapsed h_ac2(ac2)
                end
                t = minimum(ts)
            end
            @info "position: $pos, timing: $t"
            return t
        end
    end
end

# Plot results
# ------------
f = let f = Figure(; title="Absolute time of local application")
    ax = f[1, 1] = Axis(f; xlabel="site", ylabel="time (s)")
    positions = sites .+ (0.5)
    lines!(positions, times_sparsempo; label="Sparse MPO")
    check_dense &&
        lines!(positions, times_densempo; label="Dense MPO")
    check_jordanmpo &&
        lines!(positions, times_jordan; label="Jordan MPO")
    check_nearestneighbour &&
        lines!(positions, times_nn; label="Nearest Neighbours")

    f[1, 2] = Legend(f, ax, "Operators"; framevisible=false)
    f
end
save("results.png", f)

f2 = let f = Figure(; title="Relative time of local application")
    ax = f[1, 1] = Axis(f; xlabel="site", ylabel="time (%)")
    positions = sites .+ (0.5)
    check_dense &&
        lines!(positions, times_densempo ./ times_sparsempo; label="Dense MPO")
    check_jordanmpo &&
        lines!(positions, times_jordan ./ times_sparsempo; label="Jordan MPO")
    check_nearestneighbour &&
        lines!(positions, times_nn ./ times_sparsempo; label="Nearest Neighbours")

    f[1, 2] = Legend(f, ax, "Operators"; framevisible=false)
    f
end
save("results_relative.png", f2)
