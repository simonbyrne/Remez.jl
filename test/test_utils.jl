using Remez
using Test

@testset "Utility Functions" begin
    @testset "Gaussian elimination" begin
        m = BigFloat[1 1 2; 3 5 8; 13 34 21]
        v = BigFloat[1, -1, 2]
        ret = m \ v

        @test ret[1] ≈ big(109)/big(26)
        @test ret[2] ≈ big(-105)/big(130)
        @test ret[3] ≈ big(-31)/big(26)
    end

    @testset "Golden section search" begin
        x, y = Remez.goldensection(sin, big"0.0", big"0.1", big"4.0")
        @test isapprox(x, asin(BigFloat(1)))
        @test isapprox(y, 1)
    end

    @testset "Extrema-winnowing algorithm" begin
        extrema = [(x, sin(20*x)*sin(197*x))  for x in big"0.0":big"0.001":big"1.0"]
        winnowed = Remez.winnow_extrema(extrema, 12)
        prevx, prevy = -1.0, 0.0
        for (x,y) in winnowed
            @test x > prevx
            @test y != 0
            @test prevy * y <= 0 # tolerates initial prevx having no sign
            @test abs(y) > 0.9
            prevx, prevy = x, y
        end
    end
    
    @testset "Poly Eval Type Stability" begin
        coeffs = [1.0, 2.0, 3.0]
        @test typeof(Remez.poly_eval(coeffs, 0.5)) == Float64
        
        coeffs_big = [big(1.0), big(2.0), big(3.0)]
        @test typeof(Remez.poly_eval(coeffs_big, big(0.5))) == BigFloat
    end

    # Not being able to find enough alternating extrema is an expected outcome
    # (the Remez iteration has to be able to start, or repair itself), so both
    # winnow routines report it instead of throwing.
    @testset "Winnow reports insufficient extrema" begin
        same_sign = [(big(0.0), big(1.0)), (big(0.5), big(2.0)), (big(1.0), big(3.0))]
        @test isempty(Remez.winnow_extrema(same_sign, 2))

        ws = RemezWorkspace{BigFloat}((0.0, 1.0), 0, 0)
        for i in 1:3
            ws.extrema_x[i] = BigFloat(i - 1) / 2
            ws.extrema_y[i] = BigFloat(i)
        end
        @test Remez.winnow_extrema!(ws, 3, 2) == 0

        alternating = [(big(0.0), big(1.0)), (big(0.5), big(-2.0)), (big(1.0), big(3.0))]
        @test length(Remez.winnow_extrema(alternating, 3)) == 3
    end

    @testset "wellcond is type generic" begin
        @test Remez.wellcond([1.0, 2.0, 3.0], 0.0, 1.0) isa Float64
        # A single coefficient is never added to a running value.
        @test Remez.wellcond([1.0], 0.0, 1.0) == 0.0
        setprecision(BigFloat, 256) do
            c = [big(1.0), big(2.0), big(3.0)]
            @test Remez.wellcond(c, 0.0, 1.0) isa BigFloat
            # Same arithmetic, just carried out in the coefficient type.
            @test Remez.wellcond(c, 0.0, 1.0) ≈ Remez.wellcond(Float64.(c), 0.0, 1.0)
        end
    end
end
