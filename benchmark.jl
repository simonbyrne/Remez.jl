using Remez
using BenchmarkTools
using Profile
using ProfileView

function benchmark_ratfn_leastsquares()
    println("=== Benchmark ratfn_leastsquares ===")
    
    # Configuration: approximation of exp(x) in [0,1]
    n = 4  # numerator degree
    d = 2  # denominator degree
    nnodes = 16 * (n+d+1)  # same grid used in the code
    lo = big(0.0)
    hi = big(1.0)
    mid = (hi+lo)/2
    halfwid = (hi-lo)/2
    
    # Chebyshev grid (same as original code)
    grid = [mid - halfwid * cospi(big(i)/big(nnodes)) for i=0:nnodes]
    
    # Benchmark
    @btime ratfn_leastsquares(exp, $grid, $n, $d)
    
    nc, dc = ratfn_leastsquares(exp, grid, n, d)
    println("Obtained coefficients: numerator length=$(length(nc)), denominator length=$(length(dc))")
end

function benchmark_ratfn_minimax_polynomial()
    println("\n=== Benchmark ratfn_minimax (polynomial n=4, d=0) ===")
    
    @btime ratfn_minimax(exp, (0, 1), 4, 0)
    
    nc, dc, e, x = ratfn_minimax(exp, (0, 1), 4, 0)
    println("Max error: $e")
    println("Alternation points: $(length(x))")
end

function benchmark_ratfn_minimax_rational()
    println("\n=== Benchmark ratfn_minimax (rational n=2, d=2) ===")
    
    @btime ratfn_minimax(exp, (0, 1), 2, 2)
    
    nc, dc, e, x = ratfn_minimax(exp, (0, 1), 2, 2)
    println("Max error: $e")
    println("Alternation points: $(length(x))")
end

function benchmark_find_extrema()
    println("\n=== Benchmark find_extrema ===")
    
    n = 4
    d = 2
    nnodes = 16 * (n+d+1)
    lo = big(0.0)
    hi = big(1.0)
    mid = (hi+lo)/2
    halfwid = (hi-lo)/2
    grid = [mid - halfwid * cospi(big(i)/big(nnodes)) for i=0:nnodes]
    
    # Create an initial approximation
    nc, dc = ratfn_leastsquares(exp, grid, n, d)
    
    function compute_error(x)
        real_y = exp(x)
        approx_y = Remez.ratfn_eval(nc, dc, x)
        return (approx_y - real_y)
    end
    
    @btime Remez.find_extrema($compute_error, $grid)
end

function benchmark_winnow_extrema()
    println("\n=== Benchmark winnow_extrema ===")
    
    # Generate artificial extrema
    extrema = [(big(x), sin(20*x)*sin(197*x)) for x in big(0.0):big(0.001):big(1.0)]
    n = 12  # number of extrema to select
    
    @btime Remez.winnow_extrema($extrema, $n)
end

function benchmark_poly_eval()
    println("\n=== Benchmark poly_eval ===")
    
    coeffs = [BigFloat(i) for i in 1:100]
    x = big(2.5)
    
    @btime Remez.poly_eval($coeffs, $x)
    
    result = Remez.poly_eval(coeffs, x)
    println("Result: $result")
end

function run_all_benchmarks()
    println("Starting Remez.jl benchmarks")
    println("Julia version: $(VERSION)")
    println("Default BigFloat precision: $(precision(BigFloat)) bits\n")
    
    benchmark_poly_eval()
    benchmark_ratfn_leastsquares()
    benchmark_find_extrema()
    benchmark_winnow_extrema()
    benchmark_ratfn_minimax_polynomial()
    benchmark_ratfn_minimax_rational()
    
    println("\n=== All benchmarks completed ===")
end

# Optional: run profiling (comment out if not desired)
function run_profiling()
    println("\n=== Starting profiling ===")
    
    # Setup smaller test case for profiling
    n = 2
    d = 1
    nnodes = 16 * (n+d+1)
    lo = big(0.0)
    hi = big(1.0)
    mid = (hi+lo)/2
    halfwid = (hi-lo)/2
    grid = [mid - halfwid * cospi(big(i)/big(nnodes)) for i=0:nnodes]
    
    # Profile ratfn_leastsquares
    println("Profiling ratfn_leastsquares...")
    Profile.clear()
    @profile ratfn_leastsquares(exp, grid, n, d)
    println("\nTop functions by time:")
    Profile.print(format=:flat, sortedby=:time, maxdepth=10)
    
    # Profile ratfn_minimax (caso pequeno)
    println("\nProfiling ratfn_minimax (n=2, d=1)...")
    Profile.clear()
    @profile ratfn_minimax(exp, (0, 1), 2, 1)
    println("\nTop functions by time:")
    Profile.print(format=:flat, sortedby=:time, maxdepth=10)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all_benchmarks()
    run_profiling()
end