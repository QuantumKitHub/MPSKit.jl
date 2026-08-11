for T in (:FiniteMPS, :InfiniteMPS, :FiniteMPO, :InfiniteMPO, :FiniteMPOHamiltonian, :InfiniteMPOHamiltonian)
    @eval function Base.summary(io::IO, x::$T)
        L = length(x)
        D = maximum(dim, left_virtualspace(x))
        E = scalartype(x)
        S = TensorKit.type_repr(spacetype(x))
        print(io, L, "-site ", $(string(T)), "(", E, ", ", S, ") with maximal dimension ", D)
        $(T === :FiniteMPS) && print(io, " and center ", x.center)
        return nothing
    end
end

function Base.summary(io::IO, m::MultilineMPS)
    R, C = size(m)
    D = maximum(dim, left_virtualspace(m))
    E = scalartype(m)
    S = TensorKit.type_repr(spacetype(m))
    print(io, R, "×", C, " MultilineMPS(", E, ", ", S, ") with maximal dimension ", D)
    return nothing
end
function Base.summary(io::IO, m::MultilineMPO)
    R, C = size(m)
    D = maximum(dim, left_virtualspace(m))
    E = scalartype(m)
    S = TensorKit.type_repr(spacetype(m))
    print(io, R, "×", C, " MultilineMPO(", E, ", ", S, ") with maximal dimension ", D)
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", ψ::FiniteMPS)
    summary(io, ψ)
    get(io, :compact, false)::Bool && return nothing

    println(io, ":")
    io = IOContext(io, :typeinfo => spacetype(ψ))

    limit = get(io, :limit, true)::Bool
    half_screen_rows = limit ? div(displaysize(io)[1] - 2, 4) : typemax(Int)
    L = length(ψ)
    if L <= 2 * half_screen_rows # everything fits!
        half_screen_rows = typemax(Int)
    end

    # special handling of edge spaces => don't print if trivial
    Vright = right_virtualspace(ψ, L)
    right_trivial = Vright == oneunit(Vright)
    Vleft = left_virtualspace(ψ, 1)
    left_trivial = Vleft == oneunit(Vleft)


    right_trivial || println(io, "│ ", Vright)
    for i in reverse(1:L)
        if i > L - half_screen_rows
            if i == L
                connector = right_trivial ? "┌" : "├"
                println(io, connector, "─[$i]─ ", physicalspace(ψ, i))
            elseif i == 1
                connector = left_trivial ? "└" : "├"
                println(io, connector, "─[$i]─ ", physicalspace(ψ, i))
            else
                println(io, "├─[$i]─ ", physicalspace(ψ, i))
            end

            i != 1 && println(io, "│ ", left_virtualspace(ψ, i))
        elseif i == half_screen_rows
            println(io, "│ ⋮")
        elseif i < half_screen_rows
            i != L && println(io, "│ ", right_virtualspace(ψ, i))
            if i == L
                connector = right_trivial ? "┌" : "├"
                println(io, connector, "─[$i]─ ", physicalspace(ψ, i))
            elseif i == 1
                connector = left_trivial ? "└" : "├"
                println(io, connector, "─[$i]─ ", physicalspace(ψ, i))
            else
                println(io, "├─[$i]─ ", physicalspace(ψ, i))
            end
        end
    end
    left_trivial || println(io, "│ ", Vleft)

    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", ψ::InfiniteMPS)
    summary(io, ψ)
    get(io, :compact, false)::Bool && return nothing

    println(io, ":")
    io = IOContext(io, :typeinfo => spacetype(ψ))

    limit = get(io, :limit, true)::Bool
    half_screen_rows = limit ? div(displaysize(io)[1] - 6, 4) : typemax(Int)
    L = length(ψ)
    if L <= 2 * half_screen_rows # everything fits!
        half_screen_rows = typemax(Int)
    end

    println(io, "| ⋮")
    println(io, "| ", right_virtualspace(ψ, L))
    for i in reverse(1:L)
        if i > L - half_screen_rows || i < half_screen_rows
            println(io, "├─[$i]─ ", physicalspace(ψ, i))
            println(io, "│ ", left_virtualspace(ψ, i))
        elseif i == half_screen_rows
            println(io, "│ ⋮")
            println(io, "│ ", left_virtualspace(ψ, i))
        end
    end
    println(io, "| ⋮")

    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", mpo::AbstractMPO)
    summary(io, mpo)
    get(io, :compact, false)::Bool && return nothing

    println(io, ":")
    io = IOContext(io, :typeinfo => spacetype(mpo))

    limit = get(io, :limit, true)::Bool
    half_screen_rows = limit ? div(displaysize(io)[1] - 6, 4) : typemax(Int)
    L = length(mpo)
    if L <= 2 * half_screen_rows # everything fits!
        half_screen_rows = typemax(Int)
    end

    if isfinite(mpo)
        # special handling of edge spaces => don't print if trivial
        Vright = right_virtualspace(mpo, L)
        right_trivial = Vright == oneunit(Vright)
        Vleft = left_virtualspace(mpo, 1)
        left_trivial = Vleft == oneunit(Vleft)

        right_trivial || println(io, "│ ", Vright)
        for i in reverse(1:L)
            if i > L - half_screen_rows
                if i == L
                    connector = right_trivial ? "┬" : "┼"
                    println(io, connector, "─[$i]─ ", physicalspace(mpo, i))
                elseif i == 1
                    connector = left_trivial ? "┴" : "┼"
                    println(io, connector, "─[$i]─ ", physicalspace(mpo, i))
                else
                    println(io, "┼─[$i]─ ", physicalspace(mpo, i))
                end

                i != 1 && println(io, "│ ", left_virtualspace(mpo, i))
            elseif i == half_screen_rows
                println(io, "│ ⋮")
            elseif i < half_screen_rows
                i != L && println(io, "│ ", right_virtualspace(mpo, i))
                if i == L
                    connector = right_trivial ? "┬" : "┼"
                    println(io, connector, "─[$i]─ ", physicalspace(mpo, i))
                elseif i == 1
                    connector = left_trivial ? "┴" : "┼"
                    println(io, connector, "─[$i]─ ", physicalspace(mpo, i))
                else
                    println(io, "┼─[$i]─ ", physicalspace(mpo, i))
                end
            end
        end
        left_trivial || println(io, "│ ", Vleft)
    else
        println(io, "| ⋮")
        println(io, "| ", right_virtualspace(mpo, L))
        for i in reverse(1:L)
            if i > L - half_screen_rows || i < half_screen_rows
                println(io, "┼─[$i]─ ", physicalspace(mpo, i))
                println(io, "│ ", left_virtualspace(mpo, i))
            elseif i == half_screen_rows
                println(io, "│ ⋮")
                println(io, "│ ", left_virtualspace(mpo, i))
            end
        end
        println(io, "| ⋮")
    end

    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", m::Union{MultilineMPS, MultilineMPO})
    summary(io, m)
    get(io, :compact, false)::Bool && return nothing
    println(io, ":")

    R = size(m, 1)
    limit = get(io, :limit, true)::Bool
    # rows are typically few so the row budget itself is small and fixed
    # rather than computed from `displaysize`
    half_rows = (limit && R > 4) ? 2 : R
    shown = R <= 2 * half_rows ? (1:R) : [1:half_rows; (R - half_rows + 1):R]

    for (k, i) in enumerate(shown)
        println(io, "row ", i, ":")
        show(io, MIME"text/plain"(), m[i]) # dispatches to the line's own show for correct connectors
        println(io)
        if k < length(shown)
            next = shown[k + 1]
            if next == i + 1
                if m isa MultilineMPO
                    println(io, "  ↓  (row ", i, " maps onto row ", next, ")")
                else
                    println(io, "  ⋮")
                end
            else
                println(io, "  ⋮\n  ⋮")
            end
        end
    end
    m isa MultilineMPO && println(io, "  ↓  (row ", R, " maps onto row 1)")

    return nothing
end

function Base.summary(io::IO, envs::Union{FiniteEnvironments, InfiniteEnvironments})
    print(io, length(envs.GLs), "-site ")
    Base.showarg(io, envs, true)
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", envs::Union{InfiniteEnvironments, FiniteEnvironments})
    Base.summary(io, envs)
    get(io, :compact, false)::Bool && return nothing
    println(io, ":")

    for i in reverse(eachindex(envs.GLs))
        println(io, "GL[$i]: ", space(envs.GLs[i]))
    end
    for i in reverse(eachindex(envs.GRs))
        println(io, "GR[$i]: ", space(envs.GRs[i]))
    end

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
