"""
    mpo_compression(H::FiniteMPOHamiltonian; η::Number=10^-8) -> Hᵪ, Rs
    mpo_compression(H::InfiniteMPOHamiltonian; η::Number=10^-8) -> Qs, Ps

Returns the compressed version of the FiniteMPOHamiltonian or InfiniteMPOHamiltonian,
as described in: https://arxiv.org/pdf/1909.06341.

### Arguments
- `H`: Hamiltonian to be compressed. Either FiniteMPOHamiltonian or InfiniteMPOHamiltonian.

### Keyword arguments
- `η::Number=10^-8`: precision of the compression. The values smaller than η in the SVD 
  decomposition during compression will be disregarded. 

### Returns
- mpo_compression(H::FiniteMPOHamiltonian; η::Number=10^-8)
    `Hᵪ::FiniteMPOHamiltonian`: compressed Hamiltonian in the left canonical form.
    `Rs::Vector{BlockTensorMap}`: matrices that can be applied to change between left and 
      right canonical forms of Hᵪ.
- mpo_compression(H::InfiniteMPOHamiltonian; η::Number=10^-8)
    `Qs::InfiniteMPOHamiltonian`: compressed Hamiltonian in the left canonical form. 
    `Ps::InfiniteMPOHamiltonian`: compressed Hamiltonian in the right canonical form.
"""

function mpo_compression end

function mpo_compression(H::FiniteMPOHamiltonian, η::Number=10^-8)
    t = @elapsed begin
        # make sure that the MPO have at least three sites
        N = length(H)
        @assert N ≥ 3
        # change MPO from N x 1 x 1 x M blocks to 3 x 1 x 1 x 3 block structure
        TT = TensorMap{scalartype(H), spacetype(H[1]), 2, 2, Vector{scalartype(H)}}
        Hnew = Vector{SparseBlockTensorMap{TT, scalartype(H), spacetype(H[1]), 2, 2, 4}}(
                    undef, N)
        for n in 1:N
            Hnew[n] = SparseBlockTensorMap(reduce_blocks_mpo(H[n]))
        end

        Qs, Ls = right_canonical_mpo_finite(Hnew)
        TT = AbstractTensorMap{scalartype(H), spacetype(H), 2, 2}
        Cs = Vector{SparseBlockTensorMap{TT, scalartype(H), spacetype(H), 2, 2, 4}}(undef, N)
        Cs[1] = Qs[1]
        Rs = Vector{AbstractBlockTensorMap}(undef, N-2)
        R = id(codomain(Qs[2])[1])
        for n in 2:N-1
            @tensor Qnew[a,i;j,b] := R[a,c] * Qs[n][c,i;j,b]
            Q, R = qr_block_respecting(Qnew)
            # For now it is safe to assume that R will be 3x3 blocked BlockTensorMap,
            # since it went through right_canonical_mpo_finite first
            M, R′ = decompose_R(R)
            U, S, V′ = block_respecting_svd(M, η)
            @tensor C[a,i;j,b] := Q[a,i;j,c] * U[c,b]
            @tensor R[a;b] := S[a,c] * V′[c,d] * R′[d,b]
            Cs[n] = C
            Rs[n-1] = R
        end
        @tensor Cnew[a,i;j,b] := R[a,c] * Qs[N][c,i;j,b]
        Cs[N] = Cnew
        # Last step: Make new Cs into FiniteMPOHamiltonian
        # TA = AbstractTensorMap{scalartype(H), spacetype(H), 2, 2}
        # TB = AbstractTensorMap{scalartype(H), spacetype(H), 2, 1}
        # TC = AbstractTensorMap{scalartype(H), spacetype(H), 1, 2}
        # TD = AbstractTensorMap{scalartype(H), spacetype(H), 1, 1}
        Hᵪtype = JordanMPOTensor{scalartype(H), spacetype(H), Vector{scalartype(H)}}
        Hᵪ = Vector{Hᵪtype}(undef, N)
        for n in 1:N
            W = SparseBlockTensorMap(Cs[n])
            A = W[2:(end-1), 1, 1, 2:(end-1)]
            B = removeunit(W[2:(end-1), 1, 1, end], 4)
            C = removeunit(W[1,1,1,2:(end-1)], 1)
            D = removeunit(removeunit(W[1,1,1,end:end], 4), 1)
            W = JordanMPOTensor(space(W), A, B, C, D)
            Hᵪ[n] = W
        end
        Hᵪ = FiniteMPOHamiltonian(Hᵪ)
    end
    # Print reduction data
    tot_dim_original = total_virt_dimension(H)
    tot_dim_compressed = total_virt_dimension(Hᵪ)
    printstyled("┌───────────────────────────────────────\n", color=:cyan)
    printstyled("| MPO Compression performed succesfully\n", color=:cyan)
    printstyled("│ ⏱  Time:   ", color=:cyan)
    printstyled("$(round(t, digits=1)) s\n", color=:yellow)
    printstyled("| Initial total virtual dimensions:    ", color=:cyan)
    printstyled("$(tot_dim_original)\n", color=:yellow)
    printstyled("| Final total virtual dimensions:    ", color=:cyan)
    printstyled("$(tot_dim_compressed)\n", color=:yellow)
    printstyled("└───────────────────────────────────────\n")

    return Hᵪ, Rs
end

function mpo_compression(H::InfiniteMPOHamiltonian, η::Number=10^-8)
    tot_dim_original = total_virt_dimension(H)
    t = @elapsed begin
        N = length(H)   # Number of sites
        # change MPO from N x 1 x 1 x M blocks to 3 x 1 x 1 x 3 block structure
        TT = TensorMap{scalartype(H), spacetype(H[1]), 2, 2, Vector{scalartype(H)}}
        Hnew = Vector{SparseBlockTensorMap{TT, scalartype(H), spacetype(H[1]), 2, 2, 4}}(
                    undef, N)
        for n in 1:N
            Hnew[n] = SparseBlockTensorMap(reduce_blocks_mpo(H[n]))
        end

        HL, _ = left_canonical_mpo_infinite_iter_msites(Hnew)
        _, HR = right_canonical_mpo_infinite_iter_msites(HL)
        HL, Cs = left_canonical_mpo_infinite_iter_msites(HR)
        Us = Vector{BlockTensorMap}(undef, N)
        Ss = Vector{BlockTensorMap}(undef, N)
        Vs = Vector{BlockTensorMap}(undef, N)
        for n = 1:N
            Us[n], Ss[n], Vs[n] = block_respecting_svd(Cs[n], η)
        end
        Qs = Vector{BlockTensorMap}(undef, N)
        Ps = Vector{BlockTensorMap}(undef, N)
        for n = 1:N
            # Left canonical version of compressed iMPO
            @tensor Q[a,i;j,b] := Us[mod1(n-1,N)]'[a,c] * HL[n][c,i;j,d] * Us[n][d,b]
            # Right canonical version of compressed iMPO
            @tensor P[a,i;j,b] := Vs[mod1(n-1,N)][a,c] * HR[n][c,i;j,d] * Vs[n]'[d,b]
            Qs[n] = Q
            Ps[n] = P
        end

        Q̂s = create_impoham_from_mpos_msites(Qs)
        P̂s = create_impoham_from_mpos_msites(Ps)
    end

    # Print reduction data
    tot_dim_compressed = total_virt_dimension(Q̂s)
    printstyled("┌───────────────────────────────────────\n", color=:cyan)
    printstyled("| MPO Compression performed succesfully\n", color=:cyan)
    printstyled("│ ⏱  Time:   ", color=:cyan)
    printstyled("$(round(t, digits=1)) s\n", color=:yellow)
    printstyled("| Initial total virtual dimensions:    ", color=:cyan)
    printstyled("$(tot_dim_original)\n", color=:yellow)
    printstyled("| Final total virtual dimensions:    ", color=:cyan)
    printstyled("$(tot_dim_compressed)\n", color=:yellow)
    printstyled("└───────────────────────────────────────\n")

    return Q̂s, P̂s
end

# utility
# -------
function trace_single_block(d::AbstractTensorMap)
    return @tensor trd[a;b] := d[a,i;i,b]
end

function change_value_at_fusiontree!(T::AbstractTensorMap, i::Integer, v::Number)
    # change value of the fusiontree at index i of a TensorMap T with new value v
    f = fusiontrees(T)
    if f isa Vector
        T[f[i]...] .= v
    elseif f isa Indices
        T[f.values[i]...] .= v
    else
        throw(ArgumentError("fusiontrees of T is neither a Vector or a Indices. It is
                            a $(typeof(f))"))
    end
end

function total_virt_dimension(H::FiniteMPOHamiltonian)
    # calculate the sum of bond dimensions for entire H MPO
    N = length(H)
    tvd = sum([dims(H[i])[4] for i in 1:N-1])
    return tvd
end

function total_virt_dimension(H::InfiniteMPOHamiltonian)
    # calculate the sum of bond dimensions for entire H MPO
    N = length(H)
    tvd = sum([dims(H[i])[4] for i in 1:N])
    return tvd
end

# functions
# ---------
function qr_block_respecting(W::AbstractBlockTensorMap)
    # Block respecting qr (qr applied only to the upper-left part of the matrix)
    bl_st = size(W)
    dim_st = dims(W)

    # sanity checks
    @assert length(bl_st) == length(dim_st) == 4
    blN = bl_st[1]
    blM = bl_st[4]

    # First, for each nonzero fusiontree from blocks in the first row, make them
    # traceless
    id1 = id(codomain(W)[2])

    t = zeros(ComplexF64, domain(W)[2][1] ← domain(W)[2][2:end-1])
    for block in 2:blM-1
        d = W[1,1,1,block]
        trd = trace_single_block(d)
        @tensor W[1,1,1,block][a,i;j,b] = d[a,i;j,b] - trd[a;b] * id1[i;j]
        # Here goes the creation of matrix R' with a t' tensor
        t[block-1] = trd / dim_st[2]
    end
    # Perform QR of the 2:blM-1 block
    V = W[1:blN-1, :, :, 2:blM-1]
    V = permute(V, ((1,2,3), (4,)))
    
    V = convert(SparseBlockTensorMap{TensorMap{eltype(V).parameters...}, 
                eltype(storagetype(V)), spacetype(V), 3, 1, 4,}, V)
    Q, R = qr_compact(V)

    Q = permute(Q, ((1,2),(3,4)))

    # Put Q into larger Q̂
    Q̂ = zeros(scalartype(W), codomain(W) ← domain(W)[1] ⊗ (domain(W)[2][1] ⊞ domain(Q)[2] ⊞ 
                domain(W)[2][end]))
    Q̂[1,1,1,1] = Q̂[end,1,1,end] = W[1,1,1,1]
    Q̂[1:end-1,1,1,2:end-1] = Q
    Q̂[1:end-1,1,1,end] = W[1:end-1,1,1,end]

    # Put R and t into larger R̂
    R̂ = zeros(scalartype(W), domain(W)[2][1] ⊞ codomain(R)[1] ⊞ domain(W)[2][end] ←
              domain(W)[2][1] ⊞ domain(R)[1] ⊞ domain(W)[2][end])
    R̂[2:end-1,2:end-1] = R
    R̂[1,2:end-1] = t
    change_value_at_fusiontree!(R̂[1,1],1,1)
    change_value_at_fusiontree!(R̂[end,end],1,1)
    
    return Q̂, R̂
end

function lq_block_respecting(W::AbstractBlockTensorMap)
    # Block respecting lq (lq applied only to the bottom-right part of the matrix)
    bl_st = size(W)
    dim_st = dims(W)

    # sanity checks
    @assert length(bl_st) == length(dim_st) == 4
    blN = bl_st[1]
    blM = bl_st[4]
    # First, for each nonzero fusiontree from blocks in the last column, make them
    # traceless
    id1 = id(codomain(W)[2])

    t = zeros(ComplexF64, codomain(W)[1][2:end-1] ← codomain(W)[1][end])
    for block in 2:blN-1
        d = W[block,1,1,end]
        trd = trace_single_block(d)
        @tensor W[block,1,1,end][a,i;j,b] = d[a,i;j,b] - trd[a;b] * id1[i;j]
        # Here goes the creation of matrix R' with a t' tensor
        t[block-1] = trd / dim_st[2]
    end
    # Perform QR of the 2:blM-1 block (with virtual indices transposed)
    V = W[2:blN-1, :, :, 2:blM]
    V = permute(V, ((1,), (2,3,4)))

    V = convert(SparseBlockTensorMap{TensorMap{eltype(V).parameters...}, 
                eltype(storagetype(V)), spacetype(V), 1, 3, 4,}, V)
    # to perform QR
    L, Q = lq_compact(V)
    # transpose vitrual indices back (now to reconstruct initial matrix
    # one has to apply R on the left)
    Q = permute(Q, ((1,2),(3,4)))

    # Put Q into larger Q̂
    Q̂ = zeros(scalartype(W), (codomain(W)[1][1] ⊞ codomain(Q)[1] ⊞ codomain(W)[1][end]) ⊗ 
                codomain(W)[2] ← domain(W))
    Q̂[1,1,1,1] = Q̂[end,1,1,end] = W[1,1,1,1]
    Q̂[2:end-1,1,1,2:end] = Q
    Q̂[1,1,1,2:end] = W[1,1,1,2:end]

    # Put L and t into larger L̂
    L̂ = zeros(scalartype(W), domain(W)[2][1] ⊞ codomain(L)[1] ⊞ domain(W)[2][end] ←
              domain(W)[2][1] ⊞ domain(L)[1] ⊞ domain(W)[2][end])
    L̂[2:end-1,2:end-1] = L
    L̂[2:end-1,end] = t
    change_value_at_fusiontree!(L̂[1,1],1,1)
    change_value_at_fusiontree!(L̂[end,end],1,1)

    return L̂, Q̂
end

function left_canonical_mpo_finite(
        H::Union{FiniteMPOHamiltonian, V}
    ) where {T<:SparseBlockTensorMap{<:Any, <:Any, <:Any, 2, 2, 4}, V<:Vector{T}}
    # change finite Hamiltonian MPO into its Left canonical form. The 'H' is a
    # FiniteMPOHamiltonian from MPSKit (l and r boundary conditions are already
    # part of the left-most and right-most MPOs, they can not be compressed, and
    # they can be excluded from left canonicalisation algorithm).

    # make sure that the MPO have at least three sites
    N = length(H)
    @assert N ≥ 3

    Q, R = qr_block_respecting(H[2])
    TT = AbstractTensorMap{scalartype(H), spacetype(H[1]), 2, 2}
    Qs = Vector{SparseBlockTensorMap{TT, scalartype(H), spacetype(H[1]), 2, 2, 4}}(undef, N)
    Qs[1] = H[1]
    Qs[2] = Q
    Rs = Vector{AbstractBlockTensorMap}(undef, N-2)
    Rs[1] = R
    for n in 3:N-1
        @tensor Wnew[a,i;j,b] := R[a,c] * H[n][c,i;j,b]
        Q, R = qr_block_respecting(Wnew)
        Qs[n] = Q
        Rs[n-1] = R
    end
    @tensor Wlast[a,i;j,b] := R[a,c] * H[N][c,i;j,b]
    Qs[N] = Wlast
    return Qs, Rs
end

function right_canonical_mpo_finite(
        H::Union{FiniteMPOHamiltonian, V}
    ) where {T<:SparseBlockTensorMap{<:Any, <:Any, <:Any, 2, 2, 4}, V<:Vector{T}}
    # change finite Hamiltonian MPO into its Right canonical form. The 'H' is a
    # FiniteMPOHamiltonian from MPSKit (l and r boundary conditions are already
    # part of the left-most and right-most MPOs, they can not be compressed, and
    # they can be excluded from right canonicalisation algorithm).

    # make sure that the MPO have at least three sites
    N = length(H)
    @assert N ≥ 3

    L, Q = lq_block_respecting(H[end-1])
    TT = AbstractTensorMap{scalartype(H), spacetype(H[1]), 2, 2}
    Qs = Vector{SparseBlockTensorMap{TT, scalartype(H), spacetype(H[1]), 2, 2, 4}}(undef, N)
    Qs[end] = H[end]
    Qs[end-1] = Q
    Ls = Vector{AbstractBlockTensorMap}(undef, N-2)
    Ls[end] = L
    for n in N-2:-1:2
        @tensor Wnew[a,i;j,b] := H[n][a,i;j,c] * L[c,b]
        L, Q = lq_block_respecting(Wnew)
        Qs[n] = Q
        Ls[n-1] = L
    end
    @tensor Wlast[a,i;j,b] := H[1][a,i;j,c] * L[c,b]
    Qs[1] = Wlast
    return Qs, Ls
end

function decompose_R(R::AbstractBlockTensorMap)
    M = zeros(ComplexF64, codomain(R) ← domain(R))
    R′ = zeros(ComplexF64, domain(R) ← domain(R))
    change_value_at_fusiontree!(M[1,1], 1, 1)
    change_value_at_fusiontree!(M[end,end], 1, 1)
    change_value_at_fusiontree!(R′[1,1], 1, 1)
    change_value_at_fusiontree!(R′[end,end], 1, 1)
    M[2:end-1,2:end-1] = R[2:end-1,2:end-1]
    R′[1,2:end-1] = R[1,2:end-1]
    # R′[2:end-1,2:end-1] = id(codomain(R[2,2]))
    # Make diagonal matrix even with codomain and domain being different spaces
    # Right now I'm assuming that each of the fusiontree is a square block
    for (s, f) in fusiontrees(R′[2:end-1,2:end-1])
        if s.uncoupled[1] == f.uncoupled[1]
            dim = size(R′[2:end-1,2:end-1][s,f])[1]
            R′[2:end-1,2:end-1][s,f] .= diagm(ones(dim))
        end 
    end
    return M, R′
end

function block_respecting_svd(M::AbstractBlockTensorMap, η::Number; perform_truncation=true)
    # Perform block-respecting svd on matrix M, with precision η
    # TODO Change it so that it works when sectors are numbered by indices rather than 
    #      enumerated
    M2 = M[2:end-1,2:end-1]
    U, S, V′ = svd_compact(M2)

    num_sectors = length(blocksectors(S))
    new_dims = Vector{Int32}(undef, num_sectors)

    # Old code - not working if blocksectors(S) returned Indices rather than Vector
    # for n in 1:num_sectors
    #     csec = blocksectors(S)[n]
    #     new_dims[n] = blockdim(S.domain, csec)
    #     if perform_truncation
    #         for diag_el in block(S, csec).diag
    #             if diag_el ≤ η
    #                 new_dims[n] -= 1
    #             end
    #         end
    #     end
    # end
    
    # New code - works for both cases
    for (n, csec) in enumerate(blocksectors(S))
        new_dims[n] = blockdim(S.domain, csec)
        if perform_truncation
            for diag_el in block(S, csec).diag
                if diag_el ≤ η
                    new_dims[n] -= 1
                end
            end
        end
    end

    fdims = Vector{Pair{sectortype(S), Int32}}()
    # Old code 
    # for n in 1:num_sectors
    #     if new_dims[n] > 0
    #         push!(fdims, blocksectors(S)[n] => new_dims[n])
    #     end
    # end

    # New code - works for both cases
    for (n, csec) in enumerate(blocksectors(S))
        if new_dims[n] > 0
            push!(fdims, csec => new_dims[n])
        end
    end
    red_space = spacetype(S)(fdims...)
    S_new = DiagonalTensorMap(undef, red_space)
    U_new = zeros(ComplexF64, codomain(U) ← red_space)
    V′_new = zeros(ComplexF64, red_space ← domain(V′))

    #num_sectors2 = length(blocksectors(S_new))

    # Old code
    # for n in 1:num_sectors2
    #     csec = blocksectors(S_new)[n]
    #     cl = length(block(S_new, csec).diag)
    #     block(S_new, csec).diag[1:end] = block(S, csec).diag[1:cl]
    #     block(U_new, csec)[:,:] = block(U, csec)[:,1:cl]
    #     block(V′_new, csec)[:,:] = block(V′, csec)[1:cl,:]
    # end

    # New code - works for both cases
    for csec in blocksectors(S_new)
        cl = length(block(S_new, csec).diag)
        block(S_new, csec).diag[1:end] = block(S, csec).diag[1:cl]
        block(U_new, csec)[:,:] = block(U, csec)[:,1:cl]
        block(V′_new, csec)[:,:] = block(V′, csec)[1:cl,:]
    end

    Û = zeros(ComplexF64, codomain(M) ← domain(M)[1][1] ⊞ domain(U_new)[1] ⊞ domain(M)[1][end])
    Ŝ = zeros(ComplexF64, codomain(M)[1][1] ⊞ codomain(S_new)[1] ⊞ codomain(M)[1][end] ← 
              domain(M)[1][1] ⊞ domain(S_new)[1] ⊞ domain(M)[1][end])
    V̂′ = zeros(ComplexF64, codomain(M)[1][1] ⊞ codomain(V′_new)[1] ⊞ codomain(M)[1][end] ← domain(M))

    change_value_at_fusiontree!(Û[1,1], 1, 1)
    change_value_at_fusiontree!(Û[end,end], 1, 1)
    change_value_at_fusiontree!(Ŝ[1,1], 1, 1)
    change_value_at_fusiontree!(Ŝ[end,end], 1, 1)
    change_value_at_fusiontree!(V̂′[1,1], 1, 1)
    change_value_at_fusiontree!(V̂′[end,end], 1, 1)

    Û[2:end-1,2:end-1] = U_new
    Ŝ[2,2] = S_new
    V̂′[2:end-1,2:end-1] = V′_new

    return Û, Ŝ, V̂′
end

function reduce_blocks_mpo(W::AbstractBlockTensorMap)
    # Function that copies BlockTensorMap W from N x 1 x 1 x M blocked matrix to a 
    # 3 x 1 x 1 x 3 blocked matrix, with the middle block being joined dimensions of N-2, 
    # M-2 blocks from the original matrix
    # TODO Make exception if either N or M is smaller than 3

    # If the matrix is already blocked properly, return it
    if size(W)[1] ≤ 3 && size(W)[4] ≤ 3
        return W
    end
    
    old_cod = codomain(W)[1][2:end-1]
    old_dom = domain(W)[2][2:end-1]

    new_cod = spacetype(W)([sec => blockdim(old_cod, sec) for sec in sectors(old_cod)]...)
    new_dom = spacetype(W)([sec => blockdim(old_dom, sec) for sec in sectors(old_dom)]...)

    # Create new matrix with zeros
    if size(W)[1] ≤ 3
        Vcod = codomain(W)
    else
        Vcod = (codomain(W)[1][1] ⊞ new_cod ⊞ codomain(W)[1][end]) ⊗ codomain(W)[2]
    end
    if size(W)[4] ≤ 3
        Vdom = domain(W)
    else
        Vdom = domain(W)[1] ⊗ (domain(W)[2][1] ⊞ new_dom ⊞ domain(W)[2][end])
    end
    W2 = zeros(scalartype(W), Vcod ← Vdom)

    # Corners can be copied trivially
    W2[1,1,1,1] = W[1,1,1,1]
    W2[end,1,1,end] = W[end,1,1,end]
    W2[1,1,1,end] = W[1,1,1,end]
    W2[end,1,1,1] = W[end,1,1,1]
    # Treat the cases with smaller codomain or domain separately
    if size(W)[1] ≤ 2
        # Middle upperblock
        for (s,f) in fusiontrees(W[1,1,1,2:end-1])
            W2[1,1,1,2][s,f] = W[1,1,1,2:end-1][s,f]
        end
        # Middle bottomblock (if size(W)[1] is 1, this is the same block as in 
        # previous loop)
        for (s,f) in fusiontrees(W[end,1,1,2:end-1])
            W2[end,1,1,2][s,f] = W[end,1,1,2:end-1][s,f]
        end
    elseif size(W)[4] ≤ 2
        # Middle leftblock
        for (s,f) in fusiontrees(W[2:end-1,1,1,1])
            W2[2,1,1,1][s,f] = W[2:end-1,1,1,1][s,f]
        end
        # Middle rightblock (if size(W)[4] is 1, this is the same block as in previous loop)
        for (s,f) in fusiontrees(W[2:end-1,1,1,end])
            W2[2,1,1,end][s,f] = W[2:end-1,1,1,end][s,f]
        end
    else
        # All other blocks can be dealt with fusiontrees
        # Middle block (most important)
        for (s,f) in fusiontrees(W[2:end-1,1,1,2:end-1])
            W2[2,1,1,2][s,f] = W[2:end-1,1,1,2:end-1][s,f]
        end
        # Upper block
        for (s,f) in fusiontrees(W[1,1,1,2:end-1])
            W2[1,1,1,2][s,f] = W[1,1,1,2:end-1][s,f]
        end
        # Bottom block
        for (s,f) in fusiontrees(W[end,1,1,2:end-1])
            W2[3,1,1,2][s,f] = W[end,1,1,2:end-1][s,f]
        end
        # Left block
        for (s,f) in fusiontrees(W[2:end-1,1,1,1])
            W2[2,1,1,1][s,f] = W[2:end-1,1,1,1][s,f]
        end
        # Right block
        for (s,f) in fusiontrees(W[2:end-1,1,1,end])
            W2[2,1,1,3][s,f] = W[2:end-1,1,1,end][s,f]
        end
    end
    return W2
end

function check_how_far_from_id(R::AbstractBlockTensorMap)
    # Function checking how far the diagonal elements of R are from either
    # +1 or -1. Assume R is 3x3 and [1,1] and [3,3] blocks are always id
    tdev = 0
    for (s,f) in fusiontrees(R[2,2])
        if s.uncoupled[1] == f.uncoupled[1]
            for d in size(R[2,2][s,f])[1]
                tdev += min(abs(R[2,2][s,f][d,d] - 1), abs(R[2,2][s,f][d,d] + 1))
            end
        end
    end
    return tdev
end

function create_impoham_from_mpos_msites(Qs::Vector{BlockTensorMap})
    # Given the Vector{BlockTensorMap} Qs, return the multisite InfiniteMPOHamiltonian 
    N = length(Qs)
    sctype = scalartype(Qs[1])
    sptype = spacetype(Qs[1])
    # TA = AbstractTensorMap{sctype, sptype, 2, 2}
    # TB = AbstractTensorMap{sctype, sptype, 2, 1}
    # TC = AbstractTensorMap{sctype, sptype, 1, 2}
    # TD = AbstractTensorMap{sctype, sptype, 1, 1}
    CType = JordanMPOTensor{sctype, sptype, Vector{sctype}}
    Cs = Vector{CType}(undef, N)
    for n in 1:N
        W = SparseBlockTensorMap(Qs[n])
        A = W[2:(end-1), 1, 1, 2:(end-1)]
        B = removeunit(W[2:(end-1), 1, 1, end], 4)
        C = removeunit(W[1,1,1,2:(end-1)], 1)
        D = removeunit(removeunit(W[1,1,1,end:end], 4), 1)
        W = JordanMPOTensor(space(W), A, B, C, D)
        Cs[n] = W
    end
    Hnew = InfiniteMPOHamiltonian(Cs)
    return Hnew
end

function left_canonical_mpo_infinite_iter_msites(
        H::Union{InfiniteMPOHamiltonian, V},
        η=10^-10
    ) where {T<:SparseBlockTensorMap{<:Any, <:Any, <:Any, 2, 2, 4}, V<:Vector{T}}
    # Return the left_canonical form of the iMPO, H can have multiple sites in the unit cell
    N = length(H)       # Number of sites

    εs = fill(Inf, N)
    Ls = fill(id(ComplexF64, codomain(H[1])[1]), N)
    Rs = fill(id(ComplexF64, codomain(H[1])[1]), N)
    Q = similar(H[1])
    printstyled("┌─── Started iterative left orthogonalization ────\n", color=:cyan)
    while sum(εs)/N > η
        # First store the R matrices and change them H into Qs
        for n in 1:N
            Q, R = qr_block_respecting(H[n])
            H[n] = Q
            Rs[n] = R
            Ls[n] = R * Ls[n]
            εs[n] = check_how_far_from_id(R)
        end
        # Then assign R[(n+1)//N] * Q[n] to H[n]
        for n in 1:N
            @tensor H[n][a,i;j,b] = Rs[mod1(n-1,N)][a;c] * H[n][c,i;j,b]
        end
        printstyled("| total err = $(sum(εs)/N)\n", color=:white)
    end
    printstyled("└─── Left orthogonalization finished correctly ──\n", color=:cyan)
    return H, Ls
end

function right_canonical_mpo_infinite_iter_msites(
        H::Union{InfiniteMPOHamiltonian, V},
        η=10^-10
    ) where {T<:SparseBlockTensorMap{<:Any, <:Any, <:Any, 2, 2, 4}, V<:Vector{T}}
    # Return the right_canonical form of the iMPO, H can have multiple sites in the unit 
    # cell
    N = length(H)       # Number of sites
    
    εs = fill(Inf, N)
    Rs = fill(id(ComplexF64, codomain(H[1])[1]), N)
    Ls = fill(id(ComplexF64, codomain(H[1])[1]), N)
    Q = similar(H[N])
    printstyled("┌─── Started iterative right orthogonalization ────\n", color=:cyan)
    while sum(εs)/N > η
        # First store the R matrices and change them H into Qs
        for n in N:-1:1
            L, Q = lq_block_respecting(H[n])
            H[n] = Q
            Ls[n] = L
            Rs[n] = Rs[n] * L
            εs[n] = check_how_far_from_id(L)
        end
        # Then assign R[(n+1)//N] * Q[n] to H[n]
        for n in N:-1:1
            @tensor H[n][a,i;j,b] = H[n][a,i;j,c] * Ls[mod1(n+1, N)][c,b]
        end
        printstyled("| total err = $(sum(εs)/N)\n", color=:white)
    end
    printstyled("└─── Right orthogonalization finished correctly ──\n", color=:cyan)
    return Rs, H
end