using Remez
using Test

@testset "Remez.jl" begin
    include("test_utils.jl")
    include("test_minimax.jl")
    include("verify_newton.jl")
    include("test_genericity.jl")
end
