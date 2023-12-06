# Holy traits to differentiate between time dependent and time independent types

struct TimeDependent end
struct NotTimeDependent end

TimeDependence(x) = NotTimeDependent()
istimed(::TimeDependent) = true
istimed(::NotTimeDependent) = false
istimed(x) = istimed(TimeDependence(x))

safe_eval(x, args...) = safe_eval(TimeDependence(x), x, args...)

# Internal use only, works always
unsafe_eval(x, args...) = x # -> this is what you should define for custom structs
