using TensorKit, MPSKit, MPSKitModels
using MPSKit: JordanMPOTensor
using BlockTensorKit
using BenchmarkTools
using MPSKit: DefaultBackend, DefaultAllocator, GrowingBuffer

symmetry = SU2Irrep
T = Float64
D = 2048

H = heisenberg_XXX(T, symmetry, InfiniteChain(2); spin = 1 // 2);
pspaces = physicalspace(H)
A = rand(oneunit(pspaces[1]) ⊗ pspaces[1] ⊗ pspaces[2] ← oneunit(pspaces[1]))
As = MPSKit.decompose_localmps(A, trunctol(atol = 1.0e-12))
psi = InfiniteMPS(repeat(As, length(H) ÷ 2));
psi, envs = find_groundstate(psi, H, IDMRG2(; maxiter = 100, trscheme = truncrank(D)));

# regular application
function bench_single(psi, H, envs)
    Hac2 = MPSKit.AC2_hamiltonian(1, psi, H, psi, envs)
    ac2 = MPSKit.AC2(psi, 1; kind = :ACAR)
    return @benchmark $Hac2 * $ac2
end
function bench_prep(psi, H, envs; allocator = DefaultAllocator(), backend = DefaultBackend())
    Hac2 = MPSKit.AC2_hamiltonian(1, psi, H, psi, envs)
    return @benchmark MPSKit.prepare_operator!!($Hac2, $backend, $allocator)
end
function bench_prepped(psi, H, envs; allocator = DefaultAllocator(), backend = DefaultBackend())
    Hac2 = MPSKit.prepare_operator!!(MPSKit.AC2_hamiltonian(1, psi, H, psi, envs), backend, allocator)
    ac2 = MPSKit.AC2(psi, 1; kind = :ACAR)
    return @benchmark $Hac2 * $ac2
end
function densify(A::AbstractBlockTensorMap)
    B = TensorMap(A)
    return SparseBlockTensorMap(B, prod(SumSpace, codomain(B)) ← prod(SumSpace, domain(B)))
end
function densify(W::JordanMPOTensor)
    A = densify(W.A)
    B = densify(W.B)
    C = densify(W.C)
    D = densify(W.D)
    S = spacetype(W)
    Vl = oneunit(S) ⊞ left_virtualspace(A) ⊞ oneunit(S)
    Vr = oneunit(S) ⊞ right_virtualspace(A) ⊞ oneunit(S)

    return JordanMPOTensor(
        Vl ⊗ physicalspace(W) ← physicalspace(W) ⊗ Vr,
        A, B, C, D
    )
end

bench_single(psi, H, envs)
bench_prep(psi, H, envs)
bench_prepped(psi, H, envs)

SS = MPSKitModels.S_exchange(T, symmetry; spin = 1 // 2);
H_nnn = InfiniteMPOHamiltonian(physicalspace(H), (1, 2) => SS, (1, 3) => SS);
H_larger = InfiniteMPOHamiltonian(physicalspace(H), (1, i) => SS for i in 2:10);
H_largest = InfiniteMPOHamiltonian(map(densify, H_larger));
H_smallest = InfiniteMPOHamiltonian(map(densify, H));

envs_nnn = environments(psi, H_nnn);
envs_larger = environments(psi, H_larger);
envs_largest = environments(psi, H_largest);
envs_smallest = environments(psi, H_smallest);


b1 = bench_single(psi, H_nnn, envs_nnn)
b2 = bench_prep(psi, H_nnn, envs_nnn)
b3 = bench_prepped(psi, H_nnn, envs_nnn)

allocator = GrowingBuffer()
b4 = bench_single(psi, H_larger, envs_larger)
b5 = bench_prep(psi, H_larger, envs_larger)
b5 = bench_prep(psi, H_larger, envs_larger; allocator)
b6 = bench_prepped(psi, H_larger, envs_larger)
@profview b6 = bench_prepped(psi, H_larger, envs_larger; allocator)

b8 = bench_prep(psi, H_largest, envs_largest)

allocator = GrowingBuffer()
b7 = bench_single(psi, H_largest, envs_largest)
b9 = bench_prepped(psi, H_largest, envs_largest)
b9 = bench_prepped(psi, H_largest, envs_largest; allocator)

b10 = bench_single(psi, H_smallest, envs_smallest)
b11 = bench_prep(psi, H_smallest, envs_smallest)
b12 = bench_prepped(psi, H_smallest, envs_smallest)

println("next-nearest neighbour")
b1
b2
b3

println("10 sites (sparse)")
b4
b5
b6

println("10 sites (dense)")
b7
b8
b9

println("nearest-neighbour (dense)")
b10
b11
b12
