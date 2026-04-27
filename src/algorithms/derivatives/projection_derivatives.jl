struct ProjectionDerivativeOperator{L, O <: Tuple, R} <: DerivativeOperator
    leftenv::L
    As::O
    rightenv::R
end

const Projection_AC_Hamiltonian{L, O, R} = ProjectionDerivativeOperator{L, Tuple{O}, R}
Projection_AC_Hamiltonian(GL, A, GR) = ProjectionDerivativeOperator(GL, (A,), GR)

const Projection_AC2_Hamiltonian{L, O₁, O₂, R} = ProjectionDerivativeOperator{L, Tuple{O₁, O₂}, R}
Projection_AC2_Hamiltonian(GL, A1, A2, GR) = ProjectionDerivativeOperator(GL, (A1, A2), GR)

# Constructors
# ------------
function AC_hamiltonian(
        site::Int, below, operator::ProjectionOperator, above, envs;
        prepare::Bool = true
    )
    GL = leftenv(envs, site, below)
    GR = rightenv(envs, site, below)
    H_AC = Projection_AC_Hamiltonian(GL, operator.ket.AC[site], GR)
    return prepare ? prepare_operator!!(H_AC) : H_AC
end
function AC2_hamiltonian(site::Int, below, operator::ProjectionOperator, above, envs; prepare::Bool = true)
    GL = leftenv(envs, site, below)
    GR = rightenv(envs, site + 1, below)
    H_AC2 = Projection_AC2_Hamiltonian(GL, operator.ket.AC[site], operator.ket.AR[site + 1], GR)
    return prepare ? prepare_operator!!(H_AC2) : H_AC2
end

# Actions
# -------
function (h::Projection_AC_Hamiltonian)(x::MPSTensor)
    @plansor v[-1; -2 -3 -4] := h.leftenv[4; -1 -2 5] * h.As[1][5 2; 1] *
        h.rightenv[1; -3 -4 3] * conj(x[4 2; 3])
    @plansor y[-1 -2; -3] := conj(v[1; 2 5 6]) * h.leftenv[-1; 1 2 4] * h.As[1][4 -2; 3] *
        h.rightenv[3; 5 6 -3]
    return y
end
function (h::Projection_AC2_Hamiltonian)(x::MPOTensor)
    @plansor v[-1; -2 -3 -4] := h.leftenv[6; -1 -2 7] * h.As[1][7 4; 5] * h.As[2][5 2; 1] *
        h.rightenv[1; -3 -4 3] * conj(x[6 4; 3 2])
    @plansor y[-1 -2; -3 -4] := conj(v[2; 3 5 6]) * h.leftenv[-1; 2 3 4] *
        h.As[1][4 -2; 7] * h.As[2][7 -4; 1] * h.rightenv[1; 5 6 -3]
    return y
end
