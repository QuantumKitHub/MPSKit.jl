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
function exact_diagonalization(
        H::FiniteMPOHamiltonian;
        sector = one(sectortype(H)), num::Int = 1, which::Symbol = :SR,
        alg = Defaults.alg_eigsolve(; dynamic_tols = false)
    )
    L = length(H)
    @assert L > 1 "FiniteMPOHamiltonian must have length > 1"
    middle_site = (L >> 1) + 1

    T = storagetype(eltype(H))
    TA = tensormaptype(spacetype(H), 2, 1, T)

    # fuse from left to right
    ALs = Vector{Union{Missing, TA}}(missing, L)
    left = spacetype(H)(one(sector) => 1) # might need to be rightone, leave this for now
    for i in 1:(middle_site - 1)
        P = physicalspace(H, i)
        ALs[i] = isomorphism(T, left ⊗ P ← fuse(left ⊗ P))
        left = right_virtualspace(ALs[i])
    end

    # fuse from right to left
    ARs = Vector{Union{Missing, TA}}(missing, L)
    right = spacetype(H)(sector => 1)
    for i in reverse((middle_site + 1):L)
        P = physicalspace(H, i)
        ARs[i] = _transpose_front(isomorphism(T, fuse(right ⊗ P') ← right ⊗ P'))
        right = left_virtualspace(ARs[i])
    end

    # center
    ACs = Vector{Union{Missing, TA}}(missing, L)
    P = physicalspace(H, middle_site)
    ACs[middle_site] = rand!(similar(ALs[1], left ⊗ P ← right))

    TB = tensormaptype(spacetype(H), 1, 1, T)
    Cs = Vector{Union{Missing, TB}}(missing, L + 1)
    state = FiniteMPS(ALs, ARs, ACs, Cs)
    envs = environments(state, H)

    # optimize the middle site
    # Because the MPS is full rank - this is equivalent to the full Hamiltonian
    AC₀ = state.AC[middle_site]
    H_ac = AC_hamiltonian(middle_site, state, H, state, envs)
    vals, vecs, convhist = eigsolve(H_ac, AC₀, num, which, alg)

    # repack eigenstates
    state_vecs = map(vecs) do v
        cs = copy(state)
        cs.AC[middle_site] = v
        return cs
    end

    return vals, state_vecs, convhist
end
