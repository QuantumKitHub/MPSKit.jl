"
    FinEnv keeps track of the environments for FiniteMPS / WindowMPS
    It automatically checks if the queried environment is still correctly cached and if not - recalculates

    if above is set to nothing, above === below.

    opp can be a vector of nothing, in which case it'll just be the overlap
"
struct FinEnv{A,B,C,D} <: Cache
    above::A

    opp::B #the operator

    ldependencies::Vector{C} #the data we used to calculate leftenvs/rightenvs
    rdependencies::Vector{C}

    leftenvs::Vector{D}
    rightenvs::Vector{D}
end

environments(below,t::Tuple,args...;kwargs...) = environments(below,t[1],t[2],args...;kwargs...);
environments(below,opp,leftstart,rightstart) = environments(below,opp,nothing,leftstart,rightstart);
function environments(below,opp,above,leftstart,rightstart)
    leftenvs = [leftstart]
    rightenvs = [rightstart]

    for i in 1:length(below)
        push!(leftenvs,similar(leftstart))
        push!(rightenvs,similar(rightstart))
    end
    t = similar(below.AL[1]);
    return FinEnv(above,opp,fill(t,length(below)),fill(t,length(below)),leftenvs,reverse(rightenvs))
end

#automatically construct the correct leftstart/rightstart for a finitemps
function environments(below::FiniteMPS{S},ham::Union{SparseMPO,MPOHamiltonian},above=nothing) where S
    lll = l_LL(below);rrr = r_RR(below)
    rightstart = Vector{S}();leftstart = Vector{S}()

    for i in 1:ham.odim
        util_left = Tensor(x->storagetype(S)(undef,x),ham.domspaces[1,i]'); fill_data!(util_left,one);
        util_right = Tensor(x->storagetype(S)(undef,x),ham.imspaces[length(below),i]'); fill_data!(util_right,one);

        @plansor ctl[-1 -2; -3]:= lll[-1;-3]*util_left[-2]
        @plansor ctr[-1 -2; -3]:= rrr[-1;-3]*util_right[-2]

        if i != 1
            ctl = zero(ctl)
        end

        if (i != ham.odim && ham isa MPOHamiltonian) || (i != 1 && ham isa SparseMPO)
            ctr = zero(ctr)
        end

        push!(leftstart,ctl)
        push!(rightstart,ctr)
    end

    return environments(below,ham,above,leftstart,rightstart)
end

#extract the correct leftstart/rightstart for WindowMPS
function environments(state::WindowMPS,ham::Union{SparseMPO,MPOHamiltonian,DenseMPO,TimedOperator},above=nothing;lenvs=environments(state.left_gs,ham),renvs=environments(state.right_gs,ham))
    environments(state,ham,above,copy(leftenv(lenvs,1,state.left_gs)),copy(rightenv(renvs,length(state),state.right_gs)))
end

environments(below,opp::TimedOperator,above,leftstart,rightstart) = environments(below,opp.op,above,leftstart,rightstart)

# unnecesarry I think
#function environments(state::WindowMPS,ham::TimedOperator,above=nothing;lenvs=environments(state.left_gs,ham.op),renvs=environments(state.right_gs,ham.op))
#    environments(state,ham.op,above,copy(leftenv(lenvs,1,state.left_gs)),copy(rightenv(renvs,length(state),state.right_gs)))
#end

function environments(Ψ::WindowMPS,windowH::Window)
    lenvs = environments(Ψ.left_gs,windowH.left)
    renvs = environments(Ψ.right_gs,windowH.right)
    Window(lenvs, environments(Ψ,windowH.middle;lenvs=lenvs,renvs=renvs), renvs)
end

function environments(below::S,above::S) where S <: Union{FiniteMPS,WindowMPS}
    S isa WindowMPS && (above.left_gs == below.left_gs || throw(ArgumentError("left gs differs")))
    S isa WindowMPS && (above.right_gs == below.right_gs || throw(ArgumentError("right gs differs")))

    opp = fill(nothing,length(below));
    environments(below,opp,above,l_LL(above),r_RR(above))
end

function environments(state::Union{FiniteMPS,WindowMPS},opp::ProjectionOperator)
    @plansor leftstart[-1;-2 -3 -4] := l_LL(opp.ket)[-3;-4]*l_LL(opp.ket)[-1;-2]
    @plansor rightstart[-1;-2 -3 -4] := r_RR(opp.ket)[-1;-2]*r_RR(opp.ket)[-3;-4]
    environments(state,fill(nothing,length(state)),state,leftstart,rightstart)
end

#notify the cache that we updated in-place, so it should invalidate the dependencies
function poison!(ca::FinEnv,ind)
    ca.ldependencies[ind] = similar(ca.ldependencies[ind])
    ca.rdependencies[ind] = similar(ca.rdependencies[ind])
end

#rightenv[ind] will be contracteable with the tensor on site [ind]
function rightenv(ca::FinEnv,ind,state)
    a = findfirst(i -> !(state.AR[i] === ca.rdependencies[i]), length(state):-1:(ind+1))
    a = a == nothing ? nothing : length(state)-a+1

    if a != nothing
        #we need to recalculate
        for j = a:-1:ind+1
            above = isnothing(ca.above) ? state.AR[j] : ca.above.AR[j];
            ca.rightenvs[j] = TransferMatrix(above,ca.opp[j],state.AR[j])*ca.rightenvs[j+1]
            ca.rdependencies[j] = state.AR[j]
        end
    end

    return ca.rightenvs[ind+1]
end

function leftenv(ca::FinEnv,ind,state)
    a = findfirst(i -> !(state.AL[i] === ca.ldependencies[i]), 1:(ind-1))

    if a != nothing
        #we need to recalculate
        for j = a:ind-1
            above = isnothing(ca.above) ? state.AL[j] : ca.above.AL[j];
            ca.leftenvs[j+1] = ca.leftenvs[j]*TransferMatrix(above,ca.opp[j],state.AL[j])
            ca.ldependencies[j] = state.AL[j]
        end
    end

    return ca.leftenvs[ind]
end
