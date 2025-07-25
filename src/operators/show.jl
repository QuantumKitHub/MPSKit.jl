# AbstractMPO
# -----------
function Base.show(io::IO, ::MIME"text/plain", W::AbstractMPO)
    L = length(W)
    println(io, L == 1 ? "single site " : "$L-site ", typeof(W), ":")
    context = IOContext(io, :typeinfo => eltype(W), :compact => true)
    return show(context, W)
end

Base.show(io::IO, mpo::AbstractMPO) = show(convert(IOContext, io), mpo)
function Base.show(io::IOContext, mpo::AbstractMPO)
    charset = (; top = "┬", bot = "┴", mid = "┼", ver = "│", dash = "──")
    limit = get(io, :limit, false)::Bool
    half_screen_rows = limit ? div(displaysize(io)[1] - 8, 2) : typemax(Int)
    L = length(mpo)

    # used to align all mposite infos regardless of the length of the mpo (100 takes up more space than 5)
    npad = floor(Int, log10(L))
    mpoletter = mpo isa MPOHamiltonian ? "W" : "O"
    isfinite = (mpo isa FiniteMPO) || (mpo isa FiniteMPOHamiltonian)

    !isfinite && println(io, "╷  ⋮")
    for site in reverse(1:L)
        if site < half_screen_rows || site > L - half_screen_rows
            if site == L && isfinite
                println(
                    io, charset.top, " $mpoletter[$site]: ",
                    repeat(" ", npad - floor(Int, log10(site))), mpo[site]
                )
            elseif (site == 1) && isfinite
                println(
                    io, charset.bot, " $mpoletter[$site]: ",
                    repeat(" ", npad - floor(Int, log10(site))), mpo[site]
                )
            else
                println(
                    io, charset.mid, " $mpoletter[$site]: ",
                    repeat(" ", npad - floor(Int, log10(site))), mpo[site]
                )
            end
        elseif site == half_screen_rows
            println(io, "   ", "⋮")
        end
    end
    !isfinite && println(io, "╵  ⋮")
    return nothing
end

# braille
# -------
"""
    braille(io::IO, H::Union{SparseMPO, MPOHamiltonian})
    braille(H::Union{SparseMPO, MPOHamiltonian})

Prints a compact, human-readable "braille" visualization of a sparseMPO or MPOHamiltonian.
Each site of the MPO is represented as a block of Unicode braille characters, with sites separated by dashes.
This visualization is useful for quickly inspecting the structure and sparsity pattern of MPOs.

# Arguments
- `io::IO`: The output stream to print to (e.g., `stdout`).
- `H::Union{SparseMPO, MPOHamiltonian}`: The `SparseMPO` or `MPOHamiltonian` to visualize.

If called without an `io` argument, output is printed to `stdout`.
"""
function braille(io::IO, H::Union{SparseMPO, MPOHamiltonian})
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
        line *= ((i == 1 && !isfinite(H)) ? ("... " * dash) : " ")
        line *= (i > 1 && !isfinite(H)) ? "    " : ""
        for (j, braille) in enumerate(brailles)
            line *= (
                checkbounds(Bool, braille, i) ? braille[i] : repeat(" ", length(braille[1]))
            )
            if j < L
                line *= repeat(((i == 1) ? dash : " "), stride)
            end
        end
        line *= ((i == 1 && !isfinite(H)) ? (dash * " ...") : " ")
        println(io, line)
    end
    return nothing
end

braille(H::Union{SparseMPO, MPOHamiltonian}) = braille(stdout, H)
