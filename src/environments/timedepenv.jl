"""
    Structure representing a time dependent hamiltonian. Consists of
        - A tuple of the sub-hamiltonians Hᵢ representing each time dependent part
        - A tuple of functions fᵢ giving the time dependence of each sub-hamiltonian such that H(t) = ∑ᵢ fᵢ(t)*Hᵢ
"""
#
struct TimeDepProblem{O,A}
    hamiltonians::NTuple{A,O}
    
    funs::NTuple{A,Function}
end

#constructors for TimeDepProblem
TimeDepProblem(H0) = TimeDepProblem((H0,),(x->1,));

#%%
"""
    Bundle the sub-environments (=env of each time-dependent term in H) into a TimeDepProblemEnvs struct
    which contains the TimeDepProblem in the opp field and the sub-envs in the envs field.
"""
#considering moving this to a seperate Env file
struct TimeDepProblemEnvs{O,E,A} <: Cache
    opp :: TimeDepProblem{O,A}
    envs :: NTuple{A,E}
end


#let's use aliases to reduce clutter
Bundled{T,B} = NamedTuple{(:left,:window,:right),Tuple{B,T,B}} where {T,B}
BundledHams  = Bundled{TimeDepProblem,Union{TimeDepProblem,Nothing}}
BundledEnvs  = Bundled{TimeDepProblemEnvs,TimeDepProblemEnvs}

"""
    constructor for a TimeDepProblemEnvs from a Finite/Infinite MPS and a TimeDepProblem
"""
function environments(st::Union{FiniteMPS,InfiniteMPS},H::TimeDepProblem)
    #for each time-dependent term in H, calcaulate its environment and put it
    #in a TimeDepProblemEnvs container
    TimeDepProblemEnvs(H,map(h->environments(st,h),H.hamiltonians))
end

"""
    constructor for a TimeDepProblemEnvs from a Comoving MPS and a TimeDepProblem
"""
# *force* the definition of the left/right environments, as the default is probably wrong
function environments(st::MPSComoving,H::TimeDepProblem,lenvs::TimeDepProblemEnvs,renvs::TimeDepProblemEnvs)
    TimeDepProblemEnvs(H, map((le,h,re)->environments(st,h,lenvs=le,renvs=re),lenvs.envs,H.hamiltonians,renvs.envs))
end


"""
    constructor for a TimeDepProblemEnvs from a Comoving MPS and a (named) tuple of RampingProblems for (left,window,right)
"""
function environments(st::MPSComoving,Hs::BundledHams)
    env_left  = environments(st.left_gs,Hs.left)
    env_right = environments(st.right_gs,Hs.right)
    BundledEnvs( (env_left,environments(st,Hs.window,env_left,env_right),env_right) )
end
"""
    Recalculate in-place each sub-env in TimeDepProblemEnvs
"""
function recalculate!(env::TimeDepProblemEnvs,args...)
    for en in env.envs
        recalculate!(en,args...)
    end
    env
end
