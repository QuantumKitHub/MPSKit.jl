abstract type OperatorStyle end

struct MPOStyle <: OperatorStyle end
struct HamiltonianStyle <: OperatorStyle end


abstract type IsfiniteStyle end

struct FiniteStyle <: IsfiniteStyle end
struct InfiniteStyle <: IsfiniteStyle end
