using TensorKit
using MPSKit
using MPSKitModels

rows = 4
cols = 4

J = -1
g = 1

lattice = FiniteStrip(rows, rows * cols)
H = transverse_field_ising(lattice; J, g);

vals, vecs, info = exact_diagonalization(H; num = 10);

vals
