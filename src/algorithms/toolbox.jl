"calculates the entropy of a given state"
entropy(state::InfiniteMPS) = map(c->-tr(safe_xlogx(c*c')),state.CR);
entropy(state::Union{FiniteMPS,WindowMPS,InfiniteMPS},loc::Int) = -tr(safe_xlogx(state.CR[loc]*state.CR[loc]'));

infinite_temperature(ham::MPOHamiltonian) = [permute(isomorphism(storagetype(ham[1,1,1]),oneunit(sp)*sp,oneunit(sp)*sp),(1,2,4),(3,)) for sp in ham.pspaces]

"calculates the galerkin error"
function calc_galerkin(state::Union{InfiniteMPS,FiniteMPS,WindowMPS},loc,envs)::Float64
    out = ∂∂AC(loc,state,envs.opp,envs)*state.AC[loc];
    out -= state.AL[loc]*state.AL[loc]'*out
    norm(out)
end
calc_galerkin(state::Union{InfiniteMPS,FiniteMPS,WindowMPS}, envs)::Float64 =
    maximum([calc_galerkin(state,loc,envs) for loc in 1:length(state)])
function calc_galerkin(state::MPSMultiline, envs::PerMPOInfEnv)::Float64
    above = isnothing(envs.above) ? state : envs.above;

    maximum([norm(leftnull(state.AC[row+1,col])'*
        (∂∂AC(row,col,state,envs.opp,envs)*above.AC[row,col]))
            for (row,col) in product(1:size(state,1),1:size(state,2))][:])
end

"
Calculates the (partial) transfer spectrum
"
function transfer_spectrum(above::InfiniteMPS;below=above,tol=Defaults.tol,num_vals = 20,sector=first(sectors(oneunit(left_virtualspace(above,1)))))
    init = randomize!(similar(above.AL[1], left_virtualspace(below,0),ℂ[typeof(sector)](sector => 1)'*left_virtualspace(above,0)))

    transferspace = fuse(left_virtualspace(above, 0) * left_virtualspace(below, 0)')
    num_vals = min(dim(transferspace, sector), num_vals); # we can ask at most this many values
    eigenvals, eigenvecs,convhist = eigsolve(flip(TransferMatrix(above.AL,below.AL)), init, num_vals, :LM, tol=tol)
    convhist.converged < num_vals && @warn "correlation length failed to converge $(convhist.normres)"

    return eigenvals
end

"
Returns the (full) entanglement spectrum at site I
"
function entanglement_spectrum(st::Union{InfiniteMPS,FiniteMPS,WindowMPS},site::Int=0)
    @assert site<=length(st)

    (_,S,_) = tsvd(st.CR[site]);
    diag(convert(Array,S))
end

"""
Find the closest fractions of π, differing at most ```tol_angle```
"""
function approx_angles(spectrum; tol_angle=0.1)
    angles = angle.(spectrum) ./ π                          # ∈ ]-1, 1]
    angles_approx = rationalize.(angles, tol=tol_angle)     # ∈ [-1, 1]

    # Remove the effects of the branchcut.
    angles_approx[findall(angles_approx .== -1)] .= 1       # ∈ ]-1, 1]

    return angles_approx .* π                               # ∈ ]-π, π]
end


"""
Given an InfiniteMPS, compute the gap ```ϵ``` for the asymptotics of the transfer matrix, as well as the Marek gap ```δ``` as a scaling measure of the bond dimension.
"""
function marek_gap(above::InfiniteMPS; tol_angle=0.1, kwargs...)
    spectrum = transfer_spectrum(above; kwargs...)
    return marek_gap(spectrum; tol_angle)
end

function marek_gap(spectrum; tol_angle=0.1)
    # Remove 1s from the spectrum
    inds = findall(abs.(spectrum) .< 1 - 1e-12)
    length(spectrum) - length(inds) < 2 || @warn "Non-injective mps?"

    spectrum = spectrum[inds]

    angles = approx_angles(spectrum; tol_angle=tol_angle)
    θ = first(angles)

    spectrum_at_angle = spectrum[findall(angles .== θ)]


    lambdas = -log.(abs.(spectrum_at_angle));

    ϵ = first(lambdas);

    δ = Inf;
    if length(lambdas) > 2
        δ = lambdas[2]-lambdas[1]
    end

    return ϵ, δ, θ

end

"""
Compute the correlation length of a given InfiniteMPS.
"""
function correlation_length(above::InfiniteMPS; kwargs...)
    ϵ, = marek_gap(above; kwargs...)
    return 1/ϵ
end

function correlation_length(spectrum; kwargs...)
    ϵ, = marek_gap(spectrum; kwargs...)
    return 1/ϵ
end


"""
    variance(state, hamiltonian, [envs=environments(state, hamiltonian)])

Compute the variance of the energy of the state with respect to the hamiltonian.
"""
function variance end

function variance(state::InfiniteMPS,ham::MPOHamiltonian,envs = environments(state,ham))
    rescaled_ham = ham-expectation_value(state,ham,envs);
    real(sum(expectation_value(state,rescaled_ham*rescaled_ham)))
end

function variance(state::FiniteMPS,ham::MPOHamiltonian,envs = environments(state,ham))
    ham2 = ham*ham;
    real(sum(expectation_value(state,ham2)) - sum(expectation_value(state,ham,envs))^2)
end

function variance(state::WindowMPS,ham::MPOHamiltonian,envs = environments(state,ham))
    #tricky to define
    (ham2,nenvs) = squaredenvs(state,ham,envs);
    real(expectation_value(state,ham2,nenvs)[2] - expectation_value(state,ham,envs)[2]^2)
end

variance(state::FiniteQP,ham::MPOHamiltonian,args...) = variance(convert(FiniteMPS,state),ham);

function variance(state::InfiniteQP,ham::MPOHamiltonian,envs=environments(state,ham))
    # I remember there being an issue here @gertian?
    state.trivial || throw(ArgumentError("variance of domain wall excitations is not implemented"));

    rescaled_ham = ham - expectation_value(state.left_gs,ham);

    #I don't remember where the formula came from
    E_ex = dot(state,effective_excitation_hamiltonian(ham,state,envs));
    E_f = expectation_value(state.left_gs,rescaled_ham,1:0);

    ham2 = rescaled_ham*rescaled_ham

    real(dot(state,effective_excitation_hamiltonian(ham2,state))-2*(E_f+E_ex)*E_ex+E_ex^2)
end

"""
You can impose periodic boundary conditions on an mpo-hamiltonian (for a given size)
That creates a new mpo-hamiltonian with larger bond dimension
The interaction never wraps around multiple times
"""
function periodic_boundary_conditions(ham::MPOHamiltonian{S,T,E},len = ham.period) where {S,T,E}
    sanitycheck(ham) || throw(ArgumentError("invalid ham"))
    mod(len,ham.period) == 0 || throw(ArgumentError("$(len) is not a multiple of unitcell"))

    fusers = PeriodicArray(map(1:len) do loc
        map(Iterators.product(ham.domspaces[loc,:],ham.domspaces[len+1,:],ham.domspaces[loc,:])) do (v1,v2,v3)
            isomorphism(storagetype(T),fuse(v1*v2'*v3),v1*v2'*v3)
        end
    end)

    #a -> what progress have I made in the upper layer?
    #b -> what virtual space did I "lend" in the beginning?
    #c -> what progress have I made in the lower layer?
    χ = ham.odim;

    indmap = zeros(Int,χ,χ,χ);
    χ´ = 0;
    for b in χ:-1:2,c in b:χ
        χ´+=1;
        indmap[1,b,c] = χ´;
    end

    for a in 2:χ,b in χ:-1:a
        χ´+=1;
        indmap[a,b,χ] = χ´;
    end

    #do the bulk
    bulk = PeriodicArray(convert(Array{Union{T,E},3},fill(zero(E),ham.period,χ´,χ´)));

    for loc in 1:ham.period,
        (j,k) in keys(ham[loc])

        #apply (j,k) above
        l = ham.odim
        for i in 2:ham.odim

            k <= i && i<=l || continue

            f1 = fusers[loc][j,i,l]
            f2 = fusers[loc+1][k,i,l]

            @plansor bulk[loc,indmap[j,i,l],indmap[k,i,l]][-1 -2;-3 -4]:=
                ham[loc][j,k][1 2;-3 6]*f1[-1;1 3 5]*conj(f2[-4;6 7 8])*τ[2 3;7 4]*τ[4 5;8 -2]

        end


        #apply (j,k) below
        i = 1
        for l in 2:(ham.odim-1)

            l > 1 && l >= i && l<=j || continue

            f1 = fusers[loc][i,l,j];
            f2 = fusers[loc+1][i,l,k];

            @plansor bulk[loc,indmap[i,l,j],indmap[i,l,k]][-1 -2;-3 -4] :=
                ham[loc][j,k][1 -2;3 6]*f1[-1;4 2 1]*conj(f2[-4;8 7 6])*τ[5 2;7 3]*τ[-3 4;8 5]
        end
    end


    # make the starter
    starter = convert(Array{Union{T,E},2},fill(zero(E),χ´,χ´));
    for (j,k) in keys(ham[1])

        #apply (j,k) above
        if j == 1
            f1 = fusers[1][1,end,end];
            f2 = fusers[2][k,end,end];

            @plansor starter[1,indmap[k,ham.odim,ham.odim]][-1 -2;-3 -4]:=
                ham[1][j,k][-1 -2;-3 2]*conj(f2[-4;2 3 3])
        end

        #apply (j,k) below
        if j > 1 && j < ham.odim
            f1 = fusers[1][1,j,j];
            f2 = fusers[2][1,j,k];

            @plansor starter[1,indmap[1,j,k]][-1 -2;-3 -4]:=
                ham[1][j,k][4 -2;3 1]*conj(f2[-4;6 2 1])*τ[5 4;2 3]*τ[-3 -1;6 5]

        end


    end
    starter[1,1] = one(E);
    starter[end,end] = one(E);

    # make the ender
    ender = convert(Array{Union{T,E},2},fill(zero(E),χ´,χ´));
    for (j,k) in keys(ham[ham.period])

        if k > 1
            f1 = fusers[end][j,k,ham.odim]
            @plansor ender[indmap[j,k,ham.odim],end][-1 -2;-3 -4]:=
                f1[-1;1 2 6]*ham[ham.period][j,k][1 3;-3 4]*τ[3 2;4 5]*τ[5 6;-4 -2]
        end
    end
    ender[1,1] = one(E);
    ender[end,end] = one(E);

    # fill in the entire ham
    nos = convert(Array{Union{T,E},3},fill(zero(E),len,χ´,χ´))
    nos[1,:,:] = starter[:,:];
    nos[end,:,:] = ender[:,:];

    for i in 2:len-1
        nos[i,:,:] = bulk[i,:,:];
    end

    return MPOHamiltonian(nos)
end

#impose periodic boundary conditions on a normal mpo
function periodic_boundary_conditions(mpo::DenseMPO{O},len = length(mpo)) where O
    mod(len,length(mpo)) == 0 || throw(ArgumentError("len not a multiple of unitcell"))

    output = PeriodicArray{O,1}(undef,len);


    sp = _firstspace(mpo[1])';
    utleg = fill_data!(similar(mpo[1],oneunit(sp)),one)

    #do the bulk
    for j in 2:len-1
        f1 = isomorphism(storagetype(O),fuse(sp*_firstspace(mpo[j])),sp*_firstspace(mpo[j]))
        f2 = isomorphism(storagetype(O),fuse(sp*_lastspace(mpo[j])'),sp*_lastspace(mpo[j])')

        @plansor output[j][-1 -2;-3 -4] := mpo[j][2 -2;3 5]*f1[-1;1 2]*conj(f2[-4;4 5])*τ[-3 1;4 3]
    end

    #do the left
    f2 = isomorphism(storagetype(O),fuse(sp*_lastspace(mpo[1])'),sp*_lastspace(mpo[1])')
    @plansor output[1][-1 -2;-3 -4] := mpo[1][1 -2;3 5]*conj(f2[-4;4 5])*τ[-3 1;4 3]*utleg[-1]

    #do the right
    f2 = isomorphism(storagetype(O),fuse(sp*_firstspace(mpo[len])),sp*_firstspace(mpo[len]))
    @plansor output[end][-1 -2;-3 -4] := mpo[len][2 -2;3 4]*f2[-1;1 2]*τ[-3 1;4 3]*conj(utleg[-4])

    DenseMPO(output)
end
