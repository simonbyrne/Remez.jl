using Remez
using Test

import Remez: ratfn_leastsquares, ratfn_eval, ratfn_minimax, goldensection, winnow_extrema


# Test Gaussian elimination.
@testset "Gaussian test 1" begin
    m = BigFloat[1 1 2; 3 5 8; 13 34 21]
    v = BigFloat[1, -1, 2]
    ret = m \ v

    @test ret[1] ≈ big(109)/big(26)
    @test ret[2] ≈ big(-105)/big(130)
    @test ret[3] ≈ big(-31)/big(26)
end

# Test leastsquares rational functions.
@testset "Leastsquares test 1" begin
    n = 10000
    a = Array{BigFloat}(undef, n+1)
    for i = 0:1:n
        a[1+i] = i/BigFloat(n)
    end
    (nc, dc) = ratfn_leastsquares(exp, a, 2, 2)

    for x in a
        @test isapprox(exp(x), ratfn_eval(nc, dc, x); rtol=0,atol=1e-4)
    end
end

# Test golden section search.
@testset "Golden section test 1" begin
    x, y = goldensection(sin, big"0.0", big"0.1", big"4.0")
    @test isapprox(x, asin(BigFloat(1)))
    @test isapprox(y, 1)
end

# Test extrema-winnowing algorithm.
@testset "Winnow test 1" begin
    extrema = [(x, sin(20*x)*sin(197*x))  for x in big"0.0":big"0.001":big"1.0"]
    winnowed = winnow_extrema(extrema, 12)
    prevx, prevy = -1, 0
    for (x,y) in winnowed
        @test x > prevx
        @test y != 0
        @test prevy * y <= 0 # tolerates initial prevx having no sign
        @test abs(y) > 0.9
        prevx, prevy = x, y
    end
end

# Test actual minimax approximation.
@testset "Minimax test 1 (polynomial)" begin
    (nc, dc, e, x) = ratfn_minimax(exp, (0, 1), 4, 0)
    @test 0 < e < 1e-3
    for x = big"0.0":big"0.001":big"1.0"
        @test abs(ratfn_eval(nc, dc, x) - exp(x)) <= e * 1.0000001
    end
end
@testset "Minimax test 2 (rational)" begin
    (nc, dc, e, x) = ratfn_minimax(exp, (0, 1), 2, 2)
    @test 0 < e < 1e-3
    for x = big"0.0":big"0.001":big"1.0"
        @test abs(ratfn_eval(nc, dc, x) - exp(x)) <= e * 1.0000001
    end
end
@testset "Minimax test 3 (polynomial, weighted)" begin
    (nc, dc, e, x) = ratfn_minimax(exp, (0, 1), 4, 0,
                                   (x,y)->1/y)
    @test 0 < e < 1e-3
    for x = big"0.0":big"0.001":big"1.0"
        @test abs(ratfn_eval(nc, dc, x) - exp(x))/exp(x) <= e * 1.0000001
    end
end
@testset "Minimax test 4 (rational, weighted)" begin
    (nc, dc, e, x) = ratfn_minimax(exp, (0, 1), 2, 2,
                                   (x,y)->1/y)
    @test 0 < e < 1e-3
    for x = big"0.0":big"0.001":big"1.0"
        @test abs(ratfn_eval(nc, dc, x) - exp(x))/exp(x) <= e * 1.0000001
    end
end

@testset "Minimax test 5 (rational, weighted, odd degree)" begin
    
    (nc, dc, e, x) = ratfn_minimax(exp, (0, 1), 2, 1,
                                   (x,y)->1/y)
    @test 0 < e < 1e-3
    for x = big"0.0":big"0.001":big"1.0"
        @test abs(ratfn_eval(nc, dc, x) - exp(x))/exp(x) <= e * 1.0000001
    end
end
