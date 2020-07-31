#=
I don't know if I should rescale by system size / unit cell
=#
function fidelity_susceptibility(state::Union{FiniteMPS,InfiniteMPS},H₀::T,Vs::AbstractVector{T},hpars = params(state,H₀);maxiter=Defaults.maxiter,tol=Defaults.tol) where T<:MPOHamiltonian
    VL = [adjoint(rightnull(adjoint(v))) for v in state.AL]

    tangent_vecs = map(Vs) do V
        vpars = params(state,V)

        Tos = map(enumerate(zip(VL,state.AC))) do (i,(vl,ac))
            temp = adjoint(vl)*ac_prime(ac,i,state,vpars);
            help = complex(Tensor(ones,oneunit(space(temp,1))))
            @tensor tor[-1;-2 -3]:= temp[-1,-3]*help[-2]
        end

        (vec,convhist) = linsolve(RecursiveVec(Tos),RecursiveVec(Tos),GMRES(maxiter=maxiter,tol=tol)) do x
            B = [ln*cx for (ln,cx) in zip(VL,x.vecs)]
            Bseff = effective_excitation_hamiltonian(H₀, B, state, hpars)
            out = [adjoint(ln)*cB for (ln,cB) in zip(VL,Bseff)]
            RecursiveVec(out)
        end
        convhist.converged == 0 && @info "failed to converge $(convhist.normres)"

        vec
    end

    map(Iterators.product(tangent_vecs,tangent_vecs)) do (v,w)
        dot(v,w)
    end
end
