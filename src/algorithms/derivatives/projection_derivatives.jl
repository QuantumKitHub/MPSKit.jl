"""
    struct ProjectionDerivativeOperator{L, O <: Tuple, R, B, A}

Effective local operator obtained from taking the partial derivative of the projector `|ψ⟩⟨ψ|`
onto an MPS.

The `backend` and `allocator` fields are the ones used by the application of the operator. They
default to `DefaultBackend()` and `DefaultAllocator()`, i.e. this operator does not hold on to
any scratch space of its own unless it is explicitly given some.
"""
struct ProjectionDerivativeOperator{L, O <: Tuple, R, B <: AbstractBackend, A} <: DerivativeOperator
    leftenv::L
    As::O
    rightenv::R
    backend::B
    allocator::A
end

const Projection_AC_Hamiltonian{L, O, R, B, A} = ProjectionDerivativeOperator{L, Tuple{O}, R, B, A}
Projection_AC_Hamiltonian(
    GL, A, GR, backend::AbstractBackend = DefaultBackend(), allocator = DefaultAllocator()
) = ProjectionDerivativeOperator(GL, (A,), GR, backend, allocator)

const Projection_AC2_Hamiltonian{L, O₁, O₂, R, B, A} =
    ProjectionDerivativeOperator{L, Tuple{O₁, O₂}, R, B, A}
Projection_AC2_Hamiltonian(
    GL, A1, A2, GR, backend::AbstractBackend = DefaultBackend(), allocator = DefaultAllocator()
) = ProjectionDerivativeOperator(GL, (A1, A2), GR, backend, allocator)

# Constructors
# ------------
function AC_hamiltonian(
        site::Int, below, operator::ProjectionOperator, above, envs;
        prepare::Bool = true,
        backend::AbstractBackend = DefaultBackend(), allocator = DefaultAllocator()
    )
    GL = leftenv(envs, site, below)
    GR = rightenv(envs, site, below)
    H_AC = Projection_AC_Hamiltonian(GL, operator.ket.AC[site], GR, backend, allocator)
    return prepare ? prepare_operator!!(H_AC, backend, allocator) : H_AC
end
function AC2_hamiltonian(
        site::Int, below, operator::ProjectionOperator, above, envs;
        prepare::Bool = true,
        backend::AbstractBackend = DefaultBackend(), allocator = DefaultAllocator()
    )
    GL = leftenv(envs, site, below)
    GR = rightenv(envs, site + 1, below)
    H_AC2 = Projection_AC2_Hamiltonian(
        GL, operator.ket.AC[site], operator.ket.AR[site + 1], GR, backend, allocator
    )
    return prepare ? prepare_operator!!(H_AC2, backend, allocator) : H_AC2
end

# Actions
# -------
function (h::Projection_AC_Hamiltonian)(x::MPSTensor)
    backend, allocator = h.backend, h.allocator
    @plansor backend = backend allocator = allocator begin
        v[-1; -2 -3 -4] := h.leftenv[4; -1 -2 5] * h.As[1][5 2; 1] *
            h.rightenv[1; -3 -4 3] * conj(x[4 2; 3])
    end
    @plansor backend = backend allocator = allocator begin
        y[-1 -2; -3] := conj(v[1; 2 5 6]) * h.leftenv[-1; 1 2 4] * h.As[1][4 -2; 3] *
            h.rightenv[3; 5 6 -3]
    end
    return y
end
function (h::Projection_AC2_Hamiltonian)(x::MPOTensor)
    backend, allocator = h.backend, h.allocator
    @plansor backend = backend allocator = allocator begin
        v[-1; -2 -3 -4] := h.leftenv[6; -1 -2 7] * h.As[1][7 4; 5] * h.As[2][5 2; 1] *
            h.rightenv[1; -3 -4 3] * conj(x[6 4; 3 2])
    end
    @plansor backend = backend allocator = allocator begin
        y[-1 -2; -3 -4] := conj(v[2; 3 5 6]) * h.leftenv[-1; 2 3 4] *
            h.As[1][4 -2; 7] * h.As[2][7 -4; 1] * h.rightenv[1; 5 6 -3]
    end
    return y
end
