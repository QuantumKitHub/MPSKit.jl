function braille(io::IO, H::Union{SparseMPO,MPOHamiltonian})
    isfinite = (H isa FiniteMPO) || (H isa FiniteMPOHamiltonian)
    dash = "🭻"
    stride = 2 #amount of dashes between braille
    L = length(H)

    brailles = Vector{Vector{String}}(undef, L)
    buffer = IOBuffer()
    for (i, W) in enumerate(H)
        BlockTensorKit.show_braille(buffer, W)
        brailles[i] = split(String(take!(buffer)))
    end

    maxheight = maximum(length.(brailles))

    for i in 1:maxheight
        line = ""
        line *= ((i == 1 && !isfinite) ? ("... " * dash) : " ")
        line *= (i > 1 && !isfinite) ? "    " : ""
        for (j, braille) in enumerate(brailles)
            line *= (checkbounds(Bool, braille, i) ? braille[i] :
                     repeat(" ", length(braille[1])))
            if j < L
                line *= repeat(((i == 1) ? dash : " "), stride)
            end
        end
        line *= ((i == 1 && !isfinite) ? (dash * " ...") : " ")
        println(io, line)
    end
    return nothing
end

braille(H::Union{SparseMPO,MPOHamiltonian}) = braille(stdout, H)
