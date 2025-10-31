# AbstractMPO
# -----------

function Base.summary(io::IO, mpo::AbstractMPO)
    L = length(mpo)
    D = maximum(dim, left_virtualspace(mpo))
    print(io, "$L-site $(typeof(mpo)) with maximal dimension $D")
    return nothing
end

function Base.show(io::IO, mime::MIME"text/plain", mpo::AbstractMPO)
    summary(io, mpo)

    get(io, :compact, false)::Bool && return nothing
    println(io, ":")
    io = IOContext(io, :typeinfo => eltype(mpo))
    context = :compact => (get(io, :compact, false)::Bool)

    charset = (; top = "┬", bot = "┴", mid = "┼", ver = "│", dash = "──")
    limit = get(io, :limit, false)::Bool
    half_screen_rows = limit ? div(displaysize(io)[1] - 8, 2 * 8) : typemax(Int)
    L = length(mpo)

    # used to align all mposite infos regardless of the length of the mpo (100 takes up more space than 5)
    npad = floor(Int, log10(L))
    mpoletter = mpo isa MPOHamiltonian ? "W" : "O"
    isfinite = (mpo isa FiniteMPO) || (mpo isa FiniteMPOHamiltonian)

    !isfinite && println(io, "╷  ⋮")
    for site in reverse(1:L)
        if site < half_screen_rows || site > L - half_screen_rows
            if site == L && isfinite
                print(
                    io, charset.top, " $mpoletter[$site]: ",
                    repeat(" ", npad - floor(Int, log10(site)))
                )
                replace(io, sprint((x, y) -> show(x, mime, y), mpo[site]; context), "\n" => "\n" * charset.ver)
                println(io)
            elseif (site == 1) && isfinite
                print(
                    io, charset.bot, " $mpoletter[$site]: ",
                    repeat(" ", npad - floor(Int, log10(site)))
                )
                replace(io, sprint((x, y) -> show(x, mime, y), mpo[site]; context), "\n" => "\n" * charset.ver)
                println(io)
            else
                print(
                    io, charset.mid, " $mpoletter[$site]: ",
                    repeat(" ", npad - floor(Int, log10(site)))
                )
                if site == 1
                    show(io, mime, mpo[site])
                else
                    replace(io, sprint((x, y) -> show(x, mime, y), mpo[site]; context), "\n" => "\n" * charset.ver)
                end
                println(io)
            end
        elseif site == half_screen_rows
            println(io, charset.ver)
            println(io, charset.ver, "    ⋮")
            println(io, charset.ver)
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
