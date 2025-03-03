"""
    exact_diagonalization(H::FiniteMPOHamiltonian;
                          sector=first(sectors(oneunit(physicalspace(H, 1)))),
                          len::Int=length(H), num::Int=1, which::Symbol=:SR,
                          alg=Defaults.alg_eigsolve(; dynamic_tols=false))
                            -> vals, state_vecs, convhist

Use [`KrylovKit.eigsolve`](@extref) to perform exact diagonalization on a
`FiniteMPOHamiltonian` to find its eigenvectors as `FiniteMPS` of maximal rank, essentially
equivalent to dense eigenvectors.

### Arguments
- `H::FiniteMPOHamiltonian`: the Hamiltonian to diagonalize.

### Keyword arguments
- `sector=first(sectors(oneunit(physicalspace(H, 1))))`: the total charge of the
  eigenvectors, which is chosen trivial by default.
- `len::Int=length(H)`: the length of the system.
- `num::Int=1`: the number of eigenvectors to find.
- `which::Symbol=:SR`: the kind eigenvalues to find, see [`KrylovKit.eigsolve`](@extref). 
- `alg=Defaults.alg_eigsolve(; dynamic_tols=false)`: the diagonalization algorithm to use,
  see [`KrylovKit.eigsolve`](@extref).

!!! note "Valid `sector` values"
    The total charge of the eigenvectors is imposed by adding a charged auxiliary space as
    the leftmost virtualspace of each eigenvector. Specifically, this is achieved by passing
    `left=Vect[typeof(sector)](sector => 1)` to the [`FiniteMPS`](@ref) constructor. As
    such, the only valid `sector` values (i.e. `sector` values for which the corresponding
    eigenstates have valid fusion channels) are those that occur in the dual of the fusion
    of all the physical spaces in the system.

"""
function exact_diagonalization(H::FiniteMPOHamiltonian;
                               sector=first(sectors(oneunit(physicalspace(H, 1)))),
                               len::Int=length(H), num::Int=1, which::Symbol=:SR,
                               alg=Defaults.alg_eigsolve(; dynamic_tols=false))
    left = ℂ[typeof(sector)](sector => 1)
    right = oneunit(left)

    middle_site = Int(round(len / 2))

    Ot = eltype(H[1])

    mpst_type = tensormaptype(spacetype(Ot), 2, 1, storagetype(Ot))
    mpsb_type = tensormaptype(spacetype(Ot), 1, 1, storagetype(Ot))
    Cs = Vector{Union{Missing,mpsb_type}}(missing, len + 1)
    ALs = Vector{Union{Missing,mpst_type}}(missing, len)
    ARs = Vector{Union{Missing,mpst_type}}(missing, len)
    ACs = Vector{Union{Missing,mpst_type}}(missing, len)

    for i in 1:(middle_site - 1)
        ALs[i] = isomorphism(storagetype(Ot), left * physicalspace(H, i),
                             fuse(left * physicalspace(H, i)))
        left = _lastspace(ALs[i])'
    end
    for i in len:-1:(middle_site + 1)
        ARs[i] = _transpose_front(isomorphism(storagetype(Ot),
                                              fuse(right * physicalspace(H, i)'),
                                              right * physicalspace(H, i)'))
        right = _firstspace(ARs[i])
    end
    ACs[middle_site] = randomize!(similar(H[1][1, 1, 1, 1],
                                          left * physicalspace(H, middle_site) ← right))
    norm(ACs[middle_site]) == 0 && throw(ArgumentError("invalid sector"))
    normalize!(ACs[middle_site])

    #construct the largest possible finite mps of that length
    state = FiniteMPS(ALs, ARs, ACs, Cs)
    envs = environments(state, H)

    #optimize the middle site. Because there is no truncation, this single site captures the entire possible hilbert space
    H_ac = ∂∂AC(middle_site, state, H, envs) # this linear operator is now the actual full hamiltonian!
    (vals, vecs, convhist) = eigsolve(H_ac, state.AC[middle_site], num, which, alg)

    state_vecs = map(vecs) do v
        cs = copy(state)
        cs.AC[middle_site] = v
        return cs
    end

    return vals, state_vecs, convhist
end
