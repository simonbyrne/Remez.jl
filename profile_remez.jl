using Remez
using Profile

function profile_run()
    f(x) = exp(x)
    # Run once to compile
    ratfn_minimax(f, (0.0, 1.0), 5, 0)
    
    println("Profiling Float64 minimax (5,0)...")
    @profile for i in 1:1000
        ratfn_minimax(f, (0.0, 1.0), 5, 0)
    end
    Profile.print(format=:flat, sortedby=:count)
    
    Profile.clear()
    
    println("\nProfiling BigFloat minimax (5,0)...")
    @profile for i in 1:100
        ratfn_minimax(f, (big(0.0), big(1.0)), 5, 0)
    end
    Profile.print(format=:flat, sortedby=:count)
end

profile_run()
