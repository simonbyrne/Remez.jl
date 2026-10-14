using Remez
using Test

@testset "Genericity Tests" begin
    @testset "Float64 Minimax" begin
        f(x) = exp(x)
        # Low degree for Float64 to ensure stability
        (nc, dc, e, x) = ratfn_minimax(f, (0.0, 1.0), 3, 0)
        @test eltype(nc) == Float64
        @test eltype(dc) == Float64
        @test typeof(e) == Float64
        @test 0 < e < 1e-3
    end

    @testset "BigFloat Minimax" begin
        f(x) = exp(x)
        (nc, dc, e, x) = ratfn_minimax(f, (big(0.0), big(1.0)), 4, 0)
        @test eltype(nc) == BigFloat
        @test eltype(dc) == BigFloat
        @test typeof(e) == BigFloat
        @test 0 < e < 1e-3  # higher precision, lower error
    end

    @testset "Poly Eval Type Stability" begin
        coeffs = [1.0, 2.0, 3.0]
        @test typeof(Remez.poly_eval(coeffs, 0.5)) == Float64
        
        coeffs_big = [big(1.0), big(2.0), big(3.0)]
        @test typeof(Remez.poly_eval(coeffs_big, big(0.5))) == BigFloat
    end

    # The whole computation follows the precision of the interval type, so
    # raising it has to buy accuracy rather than just changing the eltype.
    @testset "Precision of the interval drives accuracy" begin
        f(x) = exp(x)
        iv = (0.0, 1.0)
        _, _, e64, _ = ratfn_minimax(f, iv, 2, 2)
        e128 = setprecision(BigFloat, 128) do
            ratfn_minimax(f, (big(0.0), big(1.0)), 2, 2)[3]
        end
        e512 = setprecision(BigFloat, 512) do
            ratfn_minimax(f, (big(0.0), big(1.0)), 2, 2)[3]
        end
        # Float64 is nowhere near able to reach the 512-bit answer; more bits
        # have to make the answer monotonically better.
        @test e128 < e64
        @test e512 < e128
        # And the more precise run must not invent an error below the true one.
        @test e128 > 0
    end
end
