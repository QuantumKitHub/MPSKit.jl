"""
    WindowEnv keeps track of the environments for WindowMPS
    Changes in the infinite environments are checked and taken into account whenever the environment of the finite part is queried.

"""
struct WindowEnv{A,B,C} <: Cache
    left_env::A
    fin_env::B
    right_env::C
end

#automatically construct the correct leftstart/rightstart for a WindowMPS
function WindowEnv(state::WindowMPS, O::Union{SparseMPO,MPOHamiltonian,DenseMPO}, above=nothing; lenvs=environments(state.left_gs, O),renvs=environments(state.right_gs, O))
    fin_env = environments(state, O, above, copy(leftenv(lenvs, 1, state.left_gs)),
    copy(rightenv(renvs, length(state), state.right_gs)))
    return WindowEnv(lenvs,fin_env,renvs)
end

function environments(below::WindowMPS, above::WindowMPS)
    above.left_gs == below.left_gs || throw(ArgumentError("left gs differs"))
    above.right_gs == below.right_gs || throw(ArgumentError("right gs differs"))

    opp = fill(nothing, length(below))
    return environments(below, opp, above, l_LL(above), r_RR(above))
end

#notify the cache that we updated in-place, so it should invalidate the dependencies
invalidate!(ca::WindowEnv, ind) = invalidate!(ca.fin_env,ind)

# check and recalculate the left and right start
function check_rightinfenv!(ca::WindowEnv)
    if ca.right_env.rw[:,0] != ca.fin_env.rightstart[end]# !== doesn't work as intended
        
        invalidate!(ca, lastindex(ca.fin_env.rightstart)) #forces transfers to be recalculated lazily 

        ca.fin_env.rightstart = ca.right_env.rw[:,0] #do some other checks and recalcs for bonddimensions?
    
    end
end

function check_leftinfenv!(ca::WindowEnv)
    if ca.left_env.lw[:,end+1] != ca.fin_env.leftstart[1]# !== doesn't work as intended
        
        invalidate!(ca, firstindex(ca.fin_env.leftstart)) #forces transfers to be recalculated lazily 

        ca.fin_env.leftstart = ca.left_env.rw[:,end+1] #do some other checks and recalcs for bonddimensions?
    
    end
end

#rightenv[ind] will be contracteable with the tensor on site [ind]
function rightenv(ca::WindowEnv, ind, state)
    check_rightinfenv!(ca)
    return rightenv(ca.fin_env, ind, state)
end

function leftenv(ca::FinEnv, ind, state)
    check_leftinfenv!(ca)
    return leftenv(ca.fin_env, ind, state)
end

function fix_infinite(ψ::WindowMPS,env::WindowEnv)
    newenv = FinEnv(env.above,env.opp,env.ldependencies,env.rdependencies,copy(env.leftenvs),copy(env.rightenvs))
    return fix_infinite(ψ),newenv
end
