# Verification script for Remez.jl refactor
# Adds src to load path and runs a simple minimax approximation

using LinearAlgebra
using Printf
using Test

# Include the module directly to ensure we use the local version
# But when running via Pkg.test(), Remez is already loaded.
# We should check if Remez is defined, if not, include it or use it.
if !isdefined(Main, :Remez)
    try
        using Remez
    catch
        include("../src/Remez.jl")
        using .Remez
    end
end

@testset "Newton-Raphson Verification" begin
    println("Verifying Remez.jl with Newton-Raphson... Target: exp(x) on [0, 1], 2/2")
    
    f = exp
    interval = (0.0, 1.0)
    n, d = 2, 2
    
    # Warmup
    Remez.ratfn_minimax(f, interval, n, d)

    start_time = time_ns()
    nc, dc, max_err, extrema = Remez.ratfn_minimax(f, interval, n, d)
    end_time = time_ns()
    
    println("Computation time: $((end_time - start_time)/1e6) ms")
    println("Max Error: $max_err")
    
    # Check error magnitude (should be reasonably small)
    @test max_err < 1e-5
    
    # Verify exact Alternation
    # We expect N+D+2 = 6 extrema with alternating signs of error.
    @test length(extrema) >= n + d + 2
    
    signs = [sign(y) for (x, y) in extrema]
    
    alternating = true
    for i in 1:length(signs)-1
        if signs[i] * signs[i+1] >= 0
            alternating = false
            break
        end
    end
    
    @test alternating
    if alternating
        println("✅ Equioscillation property satisfied.")
    else
        println("❌ Equioscillation FAILED.")
    end
end

