# ------------------------------------------
# | transfers for (vector, tensor, tensor) |
# ------------------------------------------

# transfer of density matrix (with possible utility legs in its domain) by generic mps tensors

"""
    transfer_left(v, A, Ā)

apply a transfer matrix to the left.

```
 ┌─A─
-v │
 └─Ā─
```
"""
@generated function transfer_left(v::AbstractTensorMap{<:Any,S,1,N₁},
                                  A::GenericMPSTensor{S,N₂},
                                  Abar::GenericMPSTensor{S,N₂}) where {S,N₁,N₂}
    t_out = tensorexpr(:v, -1, -(2:(N₁ + 1)))
    t_top = tensorexpr(:A, 2:(N₂ + 1), -(N₁ + 1))
    t_bot = tensorexpr(:Abar, (1, (3:(N₂ + 1))...), -1)
    t_in = tensorexpr(:v, 1, (-(2:N₁)..., 2))
    return :(return @plansor $t_out := $t_in * $t_top * conj($t_bot))
end

"""
    transfer_right(v, A, Ā)
    
apply a transfer matrix to the right.

```
─A─┐
 │ v-
─Ā─┘
```
"""
@generated function transfer_right(v::AbstractTensorMap{<:Any,S,1,N₁},
                                   A::GenericMPSTensor{S,N₂},
                                   Abar::GenericMPSTensor{S,N₂}) where {S,N₁,N₂}
    t_out = tensorexpr(:v, -1, -(2:(N₁ + 1)))
    t_top = tensorexpr(:A, (-1, reverse(3:(N₂ + 1))...), 1)
    t_bot = tensorexpr(:Abar, (-(N₁ + 1), reverse(3:(N₂ + 1))...), 2)
    t_in = tensorexpr(:v, 1, (-(2:N₁)..., 2))
    return :(return @plansor $t_out := $t_top * conj($t_bot) * $t_in)
end

# transfer, but the upper A is an excited tensor
function transfer_left(v::MPSBondTensor, A::MPOTensor, Ab::MPSTensor)
    @plansor t[-1; -2 -3] := v[1; 2] * A[2 3; -2 -3] * conj(Ab[1 3; -1])
end
function transfer_right(v::MPSBondTensor, A::MPOTensor, Ab::MPSTensor)
    @plansor t[-1; -2 -3] := A[-1 3; -2 1] * v[1; 2] * conj(Ab[-3 3; 2])
end

# transfer, but the upper A is an excited tensor and there is an mpo leg being passed through
function transfer_left(v::MPSTensor, A::MPOTensor, Ab::MPSTensor)
    @plansor t[-1 -2; -3 -4] := v[1 3; 4] * A[4 5; -3 -4] * τ[3 2; 5 -2] * conj(Ab[1 2; -1])
end

function transfer_right(v::MPSTensor, A::MPOTensor, Ab::MPSTensor)
    @plansor t[-1 -2; -3 -4] := A[-1 4; -3 5] * τ[-2 3; 4 2] * conj(Ab[-4 3; 1]) * v[5 2; 1]
end

# the transfer operation of a density matrix with a utility leg in its codomain is ill defined - how should one braid the utility leg?
# hence the checks - to make sure that this operation is uniquely defined
function transfer_left(v::MPSTensor{S}, A::MPSTensor{S}, Ab::MPSTensor{S}) where {S}
    _can_unambiguously_braid(space(v, 2)) ||
        throw(ArgumentError("transfer is not uniquely defined with utility space $(space(v,2))"))
    @plansor v[-1 -2; -3] := v[1 2; 4] * A[4 5; -3] * τ[2 3; 5 -2] * conj(Ab[1 3; -1])
end
function transfer_right(v::MPSTensor{S}, A::MPSTensor{S}, Ab::MPSTensor{S}) where {S}
    _can_unambiguously_braid(space(v, 2)) ||
        throw(ArgumentError("transfer is not uniquely defined with utility space $(space(v,2))"))
    @plansor v[-1 -2; -3] := A[-1 2; 1] * τ[-2 4; 2 3] * conj(Ab[-3 4; 5]) * v[1 3; 5]
end

# the transfer operation with a utility leg in both the domain and codomain is also ill defined - only due to the codomain utility space
function transfer_left(v::MPOTensor{S}, A::MPSTensor{S}, Ab::MPSTensor{S}) where {S}
    _can_unambiguously_braid(space(v, 2)) ||
        throw(ArgumentError("transfer is not uniquely defined with utility space $(space(v,2))"))
    @plansor t[-1 -2; -3 -4] := v[1 2; -3 4] * A[4 5; -4] * τ[2 3; 5 -2] * conj(Ab[1 3; -1])
end
function transfer_right(v::MPOTensor{S}, A::MPSTensor{S}, Ab::MPSTensor{S}) where {S}
    _can_unambiguously_braid(space(v, 2)) ||
        throw(ArgumentError("transfer is not uniquely defined with utility space $(space(v,2))"))
    @plansor t[-1 -2; -3 -4] := A[-1 2; 1] * τ[-2 4; 2 3] * conj(Ab[-4 4; 5]) * v[1 3; -3 5]
end

#transfer for 2 mpo tensors
function transfer_left(v::MPSBondTensor, A::MPOTensor, B::MPOTensor)
    @plansor t[-1; -2] := v[1; 2] * A[2 3; 4 -2] * conj(B[1 3; 4 -1])
end
function transfer_right(v::MPSBondTensor, A::MPOTensor, B::MPOTensor)
    @plansor t[-1; -2] := A[-1 3; 4 1] * conj(B[-2 3; 4 2]) * v[1; 2]
end

# ----------------------------------------------------
# | transfers for (vector, operator, tensor, tensor) |
# ----------------------------------------------------

transfer_left(v, ::Nothing, A, B) = transfer_left(v, A, B);
transfer_right(v, ::Nothing, A, B) = transfer_right(v, A, B);
#mpo transfer
function transfer_left(x::MPSTensor, O::MPOTensor, A::MPSTensor, Ab::MPSTensor)
    @plansor y[-1 -2; -3] := x[1 2; 4] * A[4 5; -3] * O[2 3; 5 -2] * conj(Ab[1 3; -1])
end
function transfer_right(v::MPSTensor, O::MPOTensor, A::MPSTensor, Ab::MPSTensor)
    @plansor v[-1 -2; -3] := A[-1 2; 1] * O[-2 4; 2 3] * conj(Ab[-3 4; 5]) * v[1 3; 5]
end

#mpo transfer, but with A an excitation-tensor
function transfer_left(v::MPSTensor, O::MPOTensor, A::MPOTensor, Ab::MPSTensor)
    @plansor t[-1 -2; -3 -4] := v[4 2; 1] * A[1 3; -3 -4] * O[2 5; 3 -2] * conj(Ab[4 5; -1])
end
function transfer_right(v::MPSTensor, O::MPOTensor, A::MPOTensor, Ab::MPSTensor)
    @plansor t[-1 -2; -3 -4] := A[-1 4; -3 5] * O[-2 2; 4 3] * conj(Ab[-4 2; 1]) * v[5 3; 1]
end

#mpo transfer, with an excitation leg
function transfer_left(v::MPOTensor, O::MPOTensor, A::MPSTensor, Ab::MPSTensor)
    @plansor v[-1 -2; -3 -4] := v[4 2; -3 1] * A[1 3; -4] * O[2 5; 3 -2] * conj(Ab[4 5; -1])
end
function transfer_right(v::MPOTensor, O::MPOTensor, A::MPSTensor, Ab::MPSTensor)
    @plansor v[-1 -2; -3 -4] := A[-1 4; 5] * O[-2 2; 4 3] *
                                conj(Ab[-4 2; 1]) * v[5 3; -3 1]
end