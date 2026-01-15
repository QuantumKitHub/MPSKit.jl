using TensorKit
using MPSKit
using MPSKitModels: FiniteChain, hubbard_model


H_u1_su2 = hubbard_model(ComplexF64, U1Irrep, SU2Irrep, FiniteChain(4); U = 8.0, mu = 4.0, t = 1.0);
charges = fill(FermionParity(1) ⊠ U1Irrep(1) ⊠ SU2Irrep(0), 4);
H = MPSKit.add_physical_charge(H_u1_su2, charges);

ρ₀ = MPSKit.infinite_temperature_density_matrix(H)
ρ_mps = convert(FiniteMPS, ρ₀)
βs = 0.0:0.2:8.0
for i in 2:length(βs)
    global ρ_mps
    @info "Computing β = $(βs[i])"
    ρ_mps, = timestep(
        ρ_mps, H, βs[i - 1] / 2, -im * (βs[i] - βs[i - 1]) / 2,
        TDVP2(; trscheme = truncdim(64))
    )
end
