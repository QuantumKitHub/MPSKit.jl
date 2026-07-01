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
    left_edge = (interval.start - 1) % length(ham)
    left_ham = circshift(ham, -left_edge)
    right_edge = (interval.stop + 1) % length(ham)
    right_ham = circshift(ham, -right_edge + 1)
    finite_ham = FiniteMPOHamiltonian([ham[i] for i in interval])
    return WindowMPOHamiltonian(left_ham, finite_ham, right_ham)
end

Base.parent(h::WindowMPOHamiltonian) = h.finite_ham
Base.copy(h::WindowMPOHamiltonian) = WindowMPOHamiltonian(copy(h.left_ham), copy(h.finite_ham), copy(h.right_ham))

function Base.:+(a::WindowMPOHamiltonian, b::WindowMPOHamiltonian)
    # the finite window carries full Jordan virtual spaces at its boundaries, so it has to be
    # block-diagonalized at every site (like an InfiniteMPOHamiltonian) rather than through
    # the regular FiniteMPOHamiltonian addition
    Ha, Hb = a.finite_ham, b.finite_ham
    N = check_length(Ha, Hb)
    finite_ham = similar(parent(Ha))
    Vtriv = leftunitspace(first(physicalspace(Ha)))
    for i in 1:N
        A = cat(Ha[i].A, Hb[i].A; dims = (1, 4))
        B = cat(Ha[i].B, Hb[i].B; dims = 1)
        C = cat(Ha[i].C, Hb[i].C; dims = 3)
        D = Ha[i].D + Hb[i].D

        Vleft = ⊞(Vtriv, left_virtualspace(A), Vtriv)
        Vright = ⊞(Vtriv, right_virtualspace(A), Vtriv)
        V = Vleft ⊗ physicalspace(A) ← physicalspace(A) ⊗ Vright

        finite_ham[i] = JordanMPOTensor(V, A, B, C, D)
    end
    return WindowMPOHamiltonian(
        a.left_ham + b.left_ham, FiniteMPOHamiltonian(finite_ham), a.right_ham + b.right_ham
    )
end
function Base.:*(a::WindowMPOHamiltonian, b::WindowMPOHamiltonian)
    return WindowMPOHamiltonian(
        a.left_ham * b.left_ham, a.finite_ham * b.finite_ham, a.right_ham * b.right_ham
    )
end
function VectorInterface.scale(H::WindowMPOHamiltonian, α::Number)
    return WindowMPOHamiltonian(
        scale(H.left_ham, α), scale(H.finite_ham, α), scale(H.right_ham, α)
    )
end

TensorKit.dot(
    bra::WindowMPS, H::WindowMPOHamiltonian, ket::WindowMPS = bra,
    envs = environments(bra, H, ket)
) = dot(bra.window, H.finite_ham, ket.window, envs)
