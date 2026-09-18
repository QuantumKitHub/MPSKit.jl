println("
---------------------
|   Braille tests    |
---------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit

@testset "braille" begin
    # Infinite Hamiltonians and MPOs
    # -------------------------------
    H = transverse_field_ising()
    buffer = IOBuffer()
    braille(buffer, H)
    output = String(take!(buffer))
    check = """
    ... 🭻⎡⠉⢈⎤🭻 ...
         ⎣⠀⢀⎦ 
    """
    @test output == check

    O = make_time_mpo(H, 1.0, TaylorCluster(3, false, false))
    braille(buffer, O)
    output = String(take!(buffer))
    check = """
    ... 🭻⎡⡏⠉⠛⠟⎤🭻 ...
         ⎣⡇⠀⠀⡂⎦ 
    """
    @test output == check

    # Finite Hamiltonians and MPOs
    # ----------------------------
    H = transverse_field_ising(; L = 4)
    braille(buffer, H)
    output = String(take!(buffer))
    check = " ⎡⠉⠈⎤🭻🭻⎡⠉⢈⎤🭻🭻⎡⠉⢈⎤🭻🭻⎡⡁⠀⎤ \n ⎣⠀⠀⎦  ⎣⠀⢀⎦  ⎣⠀⢀⎦  ⎣⡀⠀⎦ \n"
    @test output == check

    O = make_time_mpo(H, 1.0, TaylorCluster(3, false, false))
    braille(buffer, O)
    output = String(take!(buffer))
    check = " ⎡⠉⠉⠉⠉⎤🭻🭻⎡⡏⠉⠛⠟⎤🭻🭻⎡⡏⠉⠛⠟⎤🭻🭻⎡⡇⠀⎤ \n ⎣⠀⠀⠀⠀⎦  ⎣⡇⠀⠀⡂⎦  ⎣⡇⠀⠀⡂⎦  ⎣⡇⠀⎦ \n"
    @test output == check
end
