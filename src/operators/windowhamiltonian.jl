# todo - what is the required interface for abstractmpo?
# support densempo windows?
"""
$(TYPEDEF)

The Hamiltonian counterpart of a [`WindowMPS`](@ref): a finite region embedded between an
infinite environment to the left and to the right.
It consists of an infinite Hamiltonian to the left, a finite Hamiltonian in the middle, and
an infinite Hamiltonian to the right.

Acts similar to just a finite Hamiltonian, but we "remember" the boundary Hamiltonians.

## Fields

$(TYPEDFIELDS)

## Constructors

    WindowMPOHamiltonian(ham::InfiniteMPOHamiltonian, interval::UnitRange)

Construct a `WindowMPOHamiltonian` by carving a finite `interval` out of an infinite
Hamiltonian `ham`.
The finite window consists of the sites in `interval`, while the left and right environments
are copies of `ham` whose unit cells are circshifted so that they line up with the window
boundaries.
"""
struct WindowMPOHamiltonian{O} <: AbstractMPO{O}
    "Hamiltonian acting on the infinite environment to the left of the window"
    left_ham::InfiniteMPOHamiltonian{O}
    "Hamiltonian acting on the finite window"
    finite_ham::FiniteMPOHamiltonian{O}
    "Hamiltonian acting on the infinite environment to the right of the window"
    right_ham::InfiniteMPOHamiltonian{O}
end

function WindowMPOHamiltonian(ham::InfiniteMPOHamiltonian, interval::UnitRange)

    # to make sure the interval corresponds with finite_ham, it is important that the unitcell of the left/right hamiltonians is circshifted correctly
    left_edge = (interval.start - 1) % length(ham)
    left_ham = circshift(ham, -left_edge)
    right_edge = (interval.stop + 1) % length(ham)
    right_ham = circshift(ham, -right_edge + 1)

    finite_ham = FiniteMPOHamiltonian([ham[i] for i in interval])
    return WindowMPOHamiltonian(left_ham, finite_ham, right_ham)
end

Base.parent(h::WindowMPOHamiltonian) = h.finite_ham
Base.copy(h::WindowMPOHamiltonian) = WindowMPOHamiltonian(copy(h.left_ham), copy(h.finite_ham), copy(h.right_ham))

# some basic linalg
# NOTE: `+` cannot be delegated to the regular `FiniteMPOHamiltonian` addition: the finite
# window carries the full (non-trivial) Jordan virtual spaces at its boundaries, so the two
# summands have to be block-diagonalized at every site -- including the edges -- exactly like
# for an `InfiniteMPOHamiltonian`. `-` reuses `+` through the `AbstractMPO` fallback
# (`a - b == a + (-b)`), and scaling is space-preserving so it works out of the box.
function Base.:+(a::WindowMPOHamiltonian, b::WindowMPOHamiltonian)
    return WindowMPOHamiltonian(
        a.left_ham + b.left_ham,
        _add_finite_window(a.finite_ham, b.finite_ham),
        a.right_ham + b.right_ham
    )
end
function Base.:*(a::WindowMPOHamiltonian, b::WindowMPOHamiltonian)
    return WindowMPOHamiltonian(
        a.left_ham * b.left_ham, a.finite_ham * b.finite_ham, a.right_ham * b.right_ham
    )
end

# Scaling a Jordan Hamiltonian scales every path exactly once; since each path starts in
# exactly one of the three parts, scaling each part by `α` scales the whole operator by `α`.
# This also powers unary `-`, `*` and `/` through the `AbstractMPO` fallbacks.
function VectorInterface.scale(H::WindowMPOHamiltonian, α::Number)
    return WindowMPOHamiltonian(
        scale(H.left_ham, α), scale(H.finite_ham, α), scale(H.right_ham, α)
    )
end

# Block-diagonal addition of two finite Jordan Hamiltonians that have non-trivial virtual
# spaces on their boundaries (as produced by slicing an infinite Hamiltonian into a window).
# Contrary to `Base.:+(::FiniteMPOHamiltonian, ::FiniteMPOHamiltonian)` the boundary spaces
# are grown as well, mirroring the `InfiniteMPOHamiltonian` implementation.
function _add_finite_window(
        H₁::FiniteMPOHamiltonian{O}, H₂::FiniteMPOHamiltonian{O}
    ) where {O <: JordanMPOTensor}
    N = check_length(H₁, H₂)
    H = similar(parent(H₁))
    Vtriv = leftunitspace(first(physicalspace(H₁)))
    for i in 1:N
        A = cat(H₁[i].A, H₂[i].A; dims = (1, 4))
        B = cat(H₁[i].B, H₂[i].B; dims = 1)
        C = cat(H₁[i].C, H₂[i].C; dims = 3)
        D = H₁[i].D + H₂[i].D

        Vleft = ⊞(Vtriv, left_virtualspace(A), Vtriv)
        Vright = ⊞(Vtriv, right_virtualspace(A), Vtriv)
        V = Vleft ⊗ physicalspace(A) ← physicalspace(A) ⊗ Vright

        H[i] = JordanMPOTensor(V, A, B, C, D)
    end
    return FiniteMPOHamiltonian(H)
end

TensorKit.dot(
    bra::WindowMPS, H::WindowMPOHamiltonian, ket::WindowMPS = bra,
    envs = environments(bra, H, ket)
) = dot(bra.window, H.finite_ham, ket.window, envs)
