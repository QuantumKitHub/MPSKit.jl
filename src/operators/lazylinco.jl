#=
the sum of different operators can be represented using this operator type
this will allow us to write for example the dmrg excitation code in a rather elegant way

The sum is taking in a lazy way. We calculate the effective excitation hamiltonian, and then sum those
=#

struct LazyLinco{A<:Tuple,B<:Tuple}
    opps::A
    coeffs::B
end
