using Remez
using Logging
using Test

@testset "Minimax and Least Squares" begin
    @testset "Leastsquares test" begin
        n = 1000
        a = [i/BigFloat(n) for i in 0:n]
        (nc, dc) = ratfn_leastsquares(exp, a, 2, 2)

        for x in a
            @test isapprox(exp(x), Remez.ratfn_eval(nc, dc, x); rtol=0, atol=1e-4)
        end
    end

    @testset "Minimax polynomial" begin
        (nc, dc, e, x) = ratfn_minimax(exp, (0, 1), 4, 0)
        @test 0 < e < 1e-3
        for x = big"0.0":big"0.01":big"1.0"
            @test abs(Remez.ratfn_eval(nc, dc, x) - exp(x)) <= e * 1.0000001
        end
    end

    @testset "Minimax rational" begin
        (nc, dc, e, x) = ratfn_minimax(exp, (0, 1), 2, 2)
        @test 0 < e < 1e-3
        for x = big"0.0":big"0.01":big"1.0"
            @test abs(Remez.ratfn_eval(nc, dc, x) - exp(x)) <= e * 1.0000001
        end
    end

    @testset "Minimax weighted" begin
        @testset "Polynomial weighted" begin
            (nc, dc, e, x) = ratfn_minimax(exp, (0, 1), 4, 0, (x,y)->1/y)
            @test 0 < e < 1e-3
            for x = big"0.0":big"0.01":big"1.0"
                @test abs(Remez.ratfn_eval(nc, dc, x) - exp(x))/exp(x) <= e * 1.0000001
            end
        end

        @testset "Rational weighted" begin
            (nc, dc, e, x) = ratfn_minimax(exp, (0, 1), 2, 2, (x,y)->1/y)
            @test 0 < e < 1e-3
            for x = big"0.0":big"0.01":big"1.0"
                @test abs(Remez.ratfn_eval(nc, dc, x) - exp(x))/exp(x) <= e * 1.0000001
            end
        end
        
        @testset "Rational weighted odd degree" begin
            (nc, dc, e, x) = ratfn_minimax(exp, (0, 1), 2, 1, (x,y)->1/y)
            @test 0 < e < 1e-3
            for x = big"0.0":big"0.01":big"1.0"
                @test abs(Remez.ratfn_eval(nc, dc, x) - exp(x))/exp(x) <= e * 1.0000001
            end
        end
    end

    # `e` is documented as the maximum weighted error, so it has to bound the
    # error everywhere on the interval. Taking the max over the alternation
    # points alone can badly under-report it (a peaked target has its largest
    # error between the alternation points), so check against a dense sweep.
    @testset "Reported error bounds the error on the interval" begin
        for (f, iv, degrees) in ((exp, (0.0, 1.0), ((2, 0), (4, 0), (2, 2), (4, 2))),
                                 (sin, (0.0, Float64(pi)), ((2, 0), (4, 0), (2, 2), (4, 2))),
                                 (x -> 1/(1 + 25x^2), (-1.0, 1.0), ((2, 0), (4, 0), (6, 0))))
            for (n, d) in degrees
                (nc, dc, e, _) = with_logger(NullLogger()) do
                    ratfn_minimax(f, iv, n, d)
                end
                worst = maximum(abs(Remez.ratfn_eval(nc, dc, x) - f(x))
                                for x in range(iv[1], iv[2]; length=20001))
                # Tolerance covers the difference between sampling the interval
                # on a dense sweep and sampling it at refined extrema (about one
                # ulp); the failure this guards against under-reports by a
                # factor of 5, so this stays a tight bound.
                @test worst <= e * (1 + 1e-6)
            end
        end
    end

    @testset "Equioscillation has n + d + 2 alternating points" begin
        for (n, d) in ((2, 0), (3, 0), (2, 2))
            (nc, dc, _, ext) = with_logger(NullLogger()) do
                ratfn_minimax(exp, (0.0, 1.0), n, d)
            end
            @test length(ext) == n + d + 2
            errs = [Remez.ratfn_eval(nc, dc, x) - exp(x) for (x, _) in ext]
            @test all(errs[i] * errs[i+1] < 0 for i in 1:length(errs)-1)
        end
    end

    # A Float64 rational fit past roughly (3,3) cannot reach the convergence
    # threshold, but the approximation returned must still be near optimal.
    @testset "Best found is returned when the threshold is not reached" begin
        (nc, dc, e, _) = with_logger(NullLogger()) do
            ratfn_minimax(exp, (0.0, 1.0), 3, 3)
        end
        reference = setprecision(BigFloat, 512) do
            ratfn_minimax(exp, (big(0.0), big(1.0)), 3, 3)[3]
        end
        @test e <= reference * (1 + 1e-3)
        @test e >= reference * (1 - 1e-3)
    end

    @testset "A non-converged iteration says so" begin
        logs, _ = Test.collect_test_logs(min_level=Logging.Warn) do
            ratfn_minimax(exp, (0.0, 1.0), 3, 3)
        end
        @test any(l -> l.level >= Logging.Warn, logs)
        @test any(l -> occursin("did not reach the convergence threshold",
                                string(l.message)), logs)
    end

    # When the least-squares seed cannot produce enough alternating extrema the
    # iteration cannot start; that has to be reported as an actionable error
    # rather than an internal invariant violation.
    @testset "Unusable seed raises an actionable error" begin
        err = try
            ratfn_minimax(exp, (0.0, 1.0), 4, 3)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("(4, 3)", err.msg)
        @test occursin("alternating error extrema", err.msg)
        @test occursin("setprecision", err.msg)
    end
end
