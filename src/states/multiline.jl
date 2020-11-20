#this thing is useful for statmech and peps
#in principle we could let InfiniteMPS subtype from this thing
#but then we'd have to assert that numrows == 1 everywhere where it doesn't make sense ...

"
    2d extension of InfiniteMPS
"
struct MPSMultiline{A<:GenericMPSTensor,B<:MPSBondTensor}
    AL::PeriodicArray{A,2}
    AR::PeriodicArray{A,2}
    CR::PeriodicArray{B,2}
    AC::PeriodicArray{A,2}
end

Base.size(arr::MPSMultiline) = size(arr.AL)
Base.size(arr::MPSMultiline,i) = size(arr.AL,i)
Base.length(arr::MPSMultiline) = size(arr,1)
Base.eltype(arr::MPSMultiline) = eltype(arr.AL[1])
Base.lastindex(arr::MPSMultiline,i) = lastindex(arr.AL,i);
Base.similar(st::MPSMultiline) = MPSMultiline(similar(st.AL),similar(st.AR),similar(st.CR),similar(st.AC))
TensorKit.norm(st::MPSMultiline) = norm(st.AC[1]);
virtualspace(psi::MPSMultiline, a::Integer,b::Integer) = _firstspace(psi.AL[a,b])
function Base.convert(::Type{MPSMultiline},st::InfiniteMPS)
    convert(MPSMultiline,[st]);
end
function Base.convert(::Type{MPSMultiline},v::AbstractVector{T}) where T<:InfiniteMPS{A,B} where {A,B}
    ALs = PeriodicArray{A}(undef,length(v),length(v[1]));
    ARs = PeriodicArray{A}(undef,length(v),length(v[1]));
    CRs = PeriodicArray{B}(undef,length(v),length(v[1]));
    ACs = PeriodicArray{A}(undef,length(v),length(v[1]));

    for (i,row) in enumerate(v)
        ALs[i,:] = row.AL[:];
        ARs[i,:] = row.AR[:];
        CRs[i,:] = row.CR[:];
        ACs[i,:] = row.AC[:];
    end

    MPSMultiline(ALs,ARs,CRs,ACs);
end

Base.copy(m::MPSMultiline) = MPSMultiline(copy(m.AL),copy(m.AR),copy(m.CR),copy(m.AC));
function Base.copyto!(dest::Union{MPSMultiline,InfiniteMPS},src::Union{MPSMultiline,InfiniteMPS})
    copyto!(dest.AL,src.AL);
    copyto!(dest.AR,src.AR);
    copyto!(dest.CR,src.CR);
    copyto!(dest.AC,src.AC);
    dest
end


function Base.convert(::Type{InfiniteMPS},st::MPSMultiline{A,B}) where {A,B}
    @assert size(st,1) == 1 #otherwise - how would we convert?
    convert(Vector,st)[1]
end

function Base.convert(::Type{Vector},st::MPSMultiline{A,B}) where {A,B}
    map(1:size(st,1)) do row
        InfiniteMPS{A,B}(   PeriodicArray(st.AL[row,:]),
                            PeriodicArray(st.AR[row,:]),
                            PeriodicArray(st.CR[row,:]),
                            PeriodicArray(st.AC[row,:]));
    end
end

#allow users to pass in simple arrays
function MPSMultiline(A::AbstractArray{T,2}; kwargs...) where T<:GenericMPSTensor

    #we make a copy, and are therfore garantueeing no side effects for the user
    AR = PeriodicArray(A[:,:]);

    #initial guess for CR
    CR = PeriodicArray([isomorphism(Matrix{eltype(A[1])},_lastspace(v)',_lastspace(v)') for v in A]);
    AL = similar(AR);
    AC = similar(AR);

    @sync for row in 1:size(A,1)
        @Threads.spawn begin
            uniform_leftorth!(view(AL,row,:),view(CR,row,:),view(AR,row,:);kwargs...);
            uniform_rightorth!(view(AR,row,:),view(CR,row,:),view(AL,row,:);kwargs...);

            for col in 1:size(A,2)
                AC[row,col] = AL[row,col] * CR[row,col]
            end
        end
    end

    MPSMultiline(AL,AR,CR,AC);
end

function reorth!(dst::MPSMultiline;from=:AL,kwargs...)
    @assert from == :AL
    @sync for row in 1:size(dst,1)
        @Threads.spawn begin
            #dst.AL changed, dst.CR may no longer fit
            if !reduce(&,map(x->_lastspace(x[1]) == _lastspace(x[2]),zip(view(dst.CR,row,:),view(dst.AL,row,:))))
                for i in 1:length(dst)
                    dst.CR[row,i] = isomorphism(Matrix{eltype(dst.AL[row,i])},_lastspace(dst.AL[row,i])',_lastspace(dst.AL[row,i])')
                end
            end

            uniform_rightorth!(view(dst.AR,row,:),view(dst.CR,row,:),view(dst.AL,row,:);kwargs...);

            for col in 1:size(dst,2)
                dst.AC[row,col] = dst.AL[row,col]*dst.CR[row,col]
            end
        end
    end

    dst
end


l_RR(state::MPSMultiline,row,loc::Int=1) = @tensor toret[-1;-2]:=state.CR[row,loc-1][1,-2]*conj(state.CR[row,loc-1][1,-1])
l_RL(state::MPSMultiline,row,loc::Int=1) = state.CR[row,loc-1]
l_LR(state::MPSMultiline,row,loc::Int=1) = state.CR[row,loc-1]'
l_LL(state::MPSMultiline{A},row,loc::Int=1) where A= isomorphism(Matrix{eltype(A)}, space(state.AL[row,loc],1),space(state.AL[row,loc],1))

r_RR(state::MPSMultiline{A},row,loc::Int=length(state)) where A= isomorphism(Matrix{eltype(A)},domain(state.AR[row,loc]),domain(state.AR[row,loc]))
r_RL(state::MPSMultiline,row,loc::Int=length(state)) = state.CR[row,loc]'
r_LR(state::MPSMultiline,row,loc::Int=length(state)) = state.CR[row,loc]
r_LL(state::MPSMultiline,row,loc::Int=length(state))= @tensor toret[-1;-2]:=state.CR[row,loc][-1,1]*conj(state.CR[row,loc][-2,1])
