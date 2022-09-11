using MPSKit, MPSKitModels, TensorKit

O = convert(DenseMPO, nonsym_xxz_ham())
length(O)
O = repeat(O, 5)
state = FiniteMPS(length(O), space(O, 1), ℂ^12)
alg = CBE_DMRG()
find_groundstate!(state, O, CBE_DMRG())