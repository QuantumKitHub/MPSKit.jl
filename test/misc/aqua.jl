println("
---------------------
|   Aqua tests       |
---------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using Aqua

@testset "Aqua" begin
    # TODO fix this
    Aqua.test_all(MPSKit; ambiguities = false, piracies = false)
end
