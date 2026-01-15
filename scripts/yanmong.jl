using TensorKit
using CategoryData: Object, RepA4
using MPSKit
using MPSKit.KrylovKit
using LinearAlgebra: eigvals

I = Object{RepA4}
dim.(values(I))  # => (1, 1, 1, 3)
V = Vect[I](4 => 1)

H_nn = randn(V ⊗ V ← V ⊗ V);
H_nn += H_nn';  # make it Hermitian

H = H_nn ⊗ id(V) + id(V) ⊗ H_nn;  # two-site Hamiltonian

E_symm = eigvals(H)
E_dense = eigvals(reshape(convert(Array, H), dim(V)^3, dim(V)^3))

H_mpo = InfiniteMPOHamiltonian([V], (1, 2) => H_nn);
H_mpo_finite = open_boundary_conditions(H_mpo, 3);

energies, states = exact_diagonalization(H_mpo_finite; num = 18);
x0 = rand(V^3);
vals, vecs, info = KrylovKit.eigsolve(x -> H * x, rand(V^3), 18, :SR);

@info "Eigenvalues for different methods:" sort(energies; by = real) sort(vals; by = real) sort(E_symm[one(I)]; by = real)

target_sector = I(4)
x0_charged = rand(V^3 ← Vect[I](target_sector => 1))

energies, states = exact_diagonalization(H_mpo_finite; num = 18, sector = target_sector);
vals, vecs, info = KrylovKit.eigsolve(x -> H * x, x0_charged, 18, :SR);

@info "Eigenvalues for different methods:" sort(energies; by = real) sort(vals; by = real) sort(E_symm[target_sector]; by = real)
