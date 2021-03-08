#=
I don't know if I should rescale by system size / unit cell
=#
function fidelity_susceptibility(state::Union{FiniteMPS,InfiniteMPS},H₀::T,Vs::AbstractVector{T},henvs = environments(state,H₀);maxiter=Defaults.maxiter,tol=Defaults.tol) where T<:MPOHamiltonian
    init_v = LeftGaugedQP(rand,state);

    tangent_vecs = map(Vs) do V
        venvs = environments(state,V)

        Tos = similar(init_v)
        for (i,ac) in enumerate(state.AC)
            temp = ac_prime(ac,i,state,venvs);
            help = Tensor(ones,oneunit(space(state.AC[1],1)))
            @tensor tor[-1 -2;-3 -4]:= temp[-1,-2,-4]*help[-3]
            Tos[i] = tor
        end

        (vec,convhist) = linsolve(Tos,Tos,GMRES(maxiter=maxiter,tol=tol)) do x
            effective_excitation_hamiltonian(H₀, x, environments(x,H₀,henvs))
        end
        convhist.converged == 0 && @info "failed to converge $(convhist.normres)"

        vec
    end

    map(Iterators.product(tangent_vecs,tangent_vecs)) do (v,w)
        dot(v,w)
    end
end
