# Scratch space for local updates
# -------------------------------
# The local updates of an MPS algorithm allocate a fair number of intermediates, all of which die
# before the next update. Serving them from an allocator rather than from Julia's memory manager
# keeps them out of the garbage collector's way; which allocator is appropriate depends on where the
# tensors live and on whether more than one task will be sharing it.

"""
    default_allocator(x, scheduler) -> allocator

The allocator that serves the scratch space of local updates on `x`, for work scheduled with
`scheduler`.

`x` is anything with a `storagetype`, typically the state being operated on. Host memory gets a
`TensorOperations.BufferAllocator`, which serves intermediates from a reusable buffer, when a single
task owns the allocator, and a `TensorOperations.ManualAllocator`, which `malloc`s and `free`s them
one by one, when the allocator is shared between tasks - a buffer is not thread-safe, whereas a
manual allocator holds no state at all. Any other storage type falls back on
`TensorOperations.DefaultAllocator`, which allocates through the storage type itself and is
therefore correct on any device, at the cost of leaving intermediates to the garbage collector.

Extend this function to serve a storage type that MPSKit does not know about. Dedicated scratch space
can be turned off altogether with `MPSKit.Defaults.set_buffering!`.

!!! warning
    An allocator obtained for a `SerialScheduler` must not be shared between tasks. Sites that spawn
    should pass the scheduler they spawn with, so that the allocator matches the concurrency.
"""
default_allocator(x, scheduler::Scheduler) = default_allocator(storagetype(x), scheduler)

# `Memory` only exists from Julia 1.11, and `BufferAllocator`'s own storage follows suit, so the
# host-storage test is taken from TensorOperations rather than hardcoded
const HostStorage = @static isdefined(Core, :Memory) ? Union{Array, Memory} : Array

# The scheduler is what knows whether the allocator will be shared, so it - rather than the author of
# any individual sweep - decides between a buffer and a manual allocator.
if Defaults.buffering
    default_allocator(::Type{<:HostStorage}, ::SerialScheduler) = BufferAllocator()
    default_allocator(::Type{<:HostStorage}, ::Scheduler) = ManualAllocator()
end
default_allocator(::Type, ::Scheduler) = DefaultAllocator()
