
# Constructors
# ------------
function AC_hamiltonian(
        site::Int, below::_HAM_MPS_TYPES, operator::MPOHamiltonian{<:JordanMPOTensor},
        above::_HAM_MPS_TYPES, envs
    )
    GL = leftenv(envs, site, below)
    GR = rightenv(envs, site, below)
    W = operator[site]

    return AC_hamiltonian(TensorMap(GL), TensorMap(W), TensorMap(GR))
end

function AC2_hamiltonian(
        site::Int, below::_HAM_MPS_TYPES, operator::MPOHamiltonian{<:JordanMPOTensor},
        above::_HAM_MPS_TYPES, envs
    )
    GL = leftenv(envs, site, below)
    GR = rightenv(envs, site + 1, below)
    W1 = operator[site]
    W2 = operator[site + 1]
    return AC2_hamiltonian(TensorMap(GL), TensorMap(W1), TensorMap(W2), TensorMap(GR))
end
