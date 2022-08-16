using Mmap


struct DiskManager
    globdat::Vector{UInt8} # memory mapped
    offsets::Vector{Tuple{UInt64,UInt64}}
    lock::SpinLock
end

DiskManager(filename::String,size) = DiskManager(Mmap.open(io->mmap(io,Vector{UInt8},(size,),shared=false),filename,"w+"),[],SpinLock())

Base.lock(dm::DiskManager) = lock(dm.lock);
Base.unlock(dm::DiskManager) = unlock(dm.lock);


function allocate!(dm::DiskManager,::Type{Array{T,N}},dims::NTuple{N,Int}) where {T,N}
    lock(dm);

    totalsize = sizeof(T)*prod(dims);

    i = 1;
    ni = 1;
    foundgap = false;

    for (a,b) in dm.offsets
        if a-i >= totalsize+1
            foundgap = true
            break;
        else
            ni+=1;
            i+=b;
        end
    end

    if !foundgap && length(dm.globdat) < i + totalsize
        unlock(dm)
        error("out of memory")
    end

    insert!(dm.offsets,ni,(i,totalsize))

    A = unsafe_wrap(Array{T,N},convert(Ptr{T}, Base.unsafe_convert(Ptr{UInt8},dm.globdat)+i-1),dims)

    finalizer(A) do x
        lock(dm)
        hit = findfirst(x->x[1] == i,dm.offsets);
        deleteat!(dm.offsets,hit);
        unlock(dm)
    end

    unlock(dm)

    A
end

mn2 = DiskManager("derp3.bin",1024*1024);

arr = allocate!(mn2,Array{ComplexF64,1},(10,));
append!(arr,[1.0])

pointer(mn2.globdat)

append!(mn2.globdat,rand(UInt8,100000))
