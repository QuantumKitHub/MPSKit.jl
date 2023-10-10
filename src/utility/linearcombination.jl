struct LinearCombination{O<:Tuple,C<:Tuple}
    opps::O
    coeffs::C
end

Base.:*(h::LinearCombination, v) = sum((c * (o * v) for (o, c) in zip(h.opps, h.coeffs)))
(h::LinearCombination)(v) = sum((c * o(v) for (o, c) in zip(h.opps, h.coeffs)))
