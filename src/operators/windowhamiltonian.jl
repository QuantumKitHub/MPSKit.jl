"""
A WindowMPS is a finite MPS embedded between an infinite mps to the left, and an inifite mps to the right. 
The associated hamiltonian has also an infinite part to the left, a finite hamiltonian in the middle, and an infinite part to the right.

Acts simalar as just a finite hamiltonian, but we 'remember' the boundary hamiltonians.
"""

# todo - what is the required interface for abstractmpo?
# support densempo windows?
struct WindowMPOHamiltonian{O} <: AbstractMPO{O}
    left_ham :: InfiniteMPOHamiltonian{O}
    finite_ham :: FiniteMPOHamiltonian{O}
    right_ham :: InfiniteMPOHamiltonian{O}
end

#utility constructor
function WindowMPOHamiltonian(ham::InfiniteMPOHamiltonian, interval::UnitRange)
    
    # to make sure the interval corresponds with finite_ham, it is important that the unitcell of the left/right hamiltonians is circshifted correctly
    left_edge = (interval.start-1) % length(ham)
    left_ham = InfiniteMPOHamiltonian([ham[i] for i in (left_edge-length(ham)+1):left_edge])
    right_edge = (interval.stop+1)%length(ham)
    right_ham = InfiniteMPOHamiltonian([ham[i] for i in right_edge:(right_edge+length(ham)-1)])

    finite_ham = FiniteMPOHamiltonian([ham[i] for i in  interval])
    WindowMPOHamiltonian(left_ham, finite_ham, right_ham)
end


Base.copy(h::WindowMPOHamiltonian) = WindowMPOHamiltonian(copy(h.left_ham), copy(h.finite_ham), copy(h.right_ham))

# some basic linalg
for fun in (:(Base.:+), :(Base.:-), :(Base.:*))
    @eval $fun(a::WindowMPOHamiltonian,b::WindowMPOHamiltonian) = WindowMPOHamiltonian($fun(a.left_ham,b.left_ham),$fun(a.finite_ham,b.finite_ham),$fun(a.right_ham,b.right_ham))
end

TensorKit.dot(
        bra::WindowMPS, H::WindowMPOHamiltonian, ket::WindowMPS = bra,
        envs = environments(bra, H, ket)
        ) = dot(bra.window, H.finite_ham, ket.window,envs)
