# Remez.jl - Optimized & Generic Minimax Approximation

This package provides a high-performance, numerically stable implementation of the [Remez algorithm](https://en.wikipedia.org/wiki/Remez_algorithm) for computing minimax polynomial and rational approximations to functions.

## Key Features

- **Numerical Stability**: Uses an internal **Chebyshev basis** for Remez iterations, drastically improving matrix conditioning for high-degree approximations.
- **Type Genericity**: Supports any `AbstractFloat` type (`Float64`, `BigFloat`, `Double64`, etc.). Precision is automatically determined by the input interval's type.
- **Performance**:
    - `RemezWorkspace`: Pre-allocates buffers for linear solvers and basis transformations, drastically reducing GC pressure in iterative loops.
    - Optimized evaluation using `evalpoly` and in-place basis conversions (`cheb_to_power_inplace!`).
    - Efficient extrema finding with heuristic grid sizing and dynamic programming winnowing.
- **Robustness**: 
    - Safe iteration limits to prevent infinite loops.
    - Robust extrema finding with high-density grid evaluation and golden-section search.
    - Graceful fallback: returns the best approximation found if equalization isn't reached, and warns that it did so.
    - Actionable errors: when a fit cannot be started at all, an `ArgumentError` naming the degrees and the precision available, rather than an internal invariant violation.
- **Public API**:
    - `ratfn_minimax`: standard minimax (returns power basis).
    - `ratfn_minimax_cheb`: direct Chebyshev basis results.
    - `ratfn_leastsquares`: rational least-squares initialization.

## Installation

This repository is a fork of Simon Byrne's
[Remez.jl](https://github.com/simonbyrne/Remez.jl) and deliberately keeps the
upstream package UUID, so this version stays a drop-in replacement for anyone
already using it. The consequence is that `Pkg.add("Remez")` installs the
upstream release from the registry, which does **not** provide the
`RemezWorkspace` / `ratfn_minimax_cheb` API described here. Install this version
from its repository instead:

```julia
using Pkg
Pkg.add(url="https://github.com/danalec/Remez.jl")   # or Pkg.develop(path="...")
```

Registering this fork under its own name would require a new UUID, which would
break existing manifests pinned to the current one.

## Basic Usage

### Polynomial Approximation

Approximate `exp(x)` on `[0, 1]` with a degree-3 polynomial:

```julia
using Remez
# (numerator_coeffs, denominator_coeffs, max_error, alternation_points)
(nc, dc, e, x) = ratfn_minimax(exp, (0.0, 1.0), 3, 0)
```

### Rational Approximation

Approximate `exp(x)` with a (2, 2) rational function:

```julia
(nc, dc, e, x) = ratfn_minimax(exp, (0.0, 1.0), 2, 2)
```

### High-Precision with BigFloat

Simply use `big` values for the interval:

```julia
(nc, dc, e, x) = ratfn_minimax(exp, (big(0.0), big(1.0)), 10, 5)
```

The precision of the whole computation follows the interval type. To go beyond
the default 256 bits, set the precision before calling:

```julia
setprecision(BigFloat, 512)
(nc, dc, e, x) = ratfn_minimax(exp, (big(0.0), big(1.0)), 10, 5)
```

### Return values

All three entry points return the same 4-tuple:

- `nc`, `dc` — numerator and denominator coefficients, `n + 1` and `d + 1` long,
  in the power basis for `ratfn_minimax` and `ratfn_leastsquares`, in the
  Chebyshev basis for `ratfn_minimax_cheb`. `dc[1] == 1`.
- `e` — the maximum weighted error over the interval. It is measured over every
  error extremum, not just the alternation points, so it is an upper bound on
  the error of the returned approximation rather than a partial estimate.
- `x` — a vector of `n + d + 2` `(x, weighted error)` pairs, ordered by `x`,
  with alternating signs. These are the points the Chebyshev alternation
  theorem certifies.

### Error weighting

All entry points accept an optional weight function `w(x, y)`, which
equioscillates `(R(x) - f(x)) * w(x, f(x))` instead of the plain error. For
relative error on an exponential, pass `w = (x, y) -> 1/y`:

```julia
(nc, dc, e, x) = ratfn_minimax(exp, (0.0, 1.0), 4, 0, (x, y) -> 1/y)
```

## Precision and convergence

The iteration stops when the error is equioscillated to a relative tolerance
derived from the working precision. In `Float64` that tolerance is reachable
for polynomial fits and for rational fits up to roughly `(3, 3)`; beyond that,
and in `Float64` generally, the iteration may not converge. In that case the
package returns the best approximation it found and emits a warning — it does
not silently return a worse answer, and `e` still bounds the returned
approximation's error.

Some fits cannot be started at all in `Float64`: the least-squares seed's error
function may not have the `n + d + 2` alternating extrema the iteration needs
(the same limitation exists in the original Remez implementation). That is
reported as an `ArgumentError` naming the degrees and suggesting higher
precision. Use `BigFloat` for those cases.

## Advanced Usage

### Direct Chebyshev Coefficients

To avoid power-basis instability in high-degree approximations, use the direct Chebyshev API:

```julia
(nc_cheb, dc_cheb, e, x) = ratfn_minimax_cheb(exp, (0.0, 1.0), 5, 5)
```

### Using Workspaces

For repeated calls or performance-critical loops, you can use `RemezWorkspace`:

```julia
ws = RemezWorkspace{Float64}((0.0, 1.0), 3, 3)
(nc, dc, e, x) = Remez.ratfn_minimax!(ws, exp)
```

## Testing

The test suite is modularized into several focused files:
- `test_utils.jl`: Internal utility functions and math helpers.
- `test_minimax.jl`: Core minimax and least-squares verification, plus the
  reported-error bound, the non-converged fallback, and the failure modes.
- `verify_newton.jl`: Newton-Raphson solver verification (equioscillation).
- `test_genericity.jl`: Verification of type stability and precision across different numeric types.

`test/runtests.jl` is the only entry point and includes the other four files.

To run all tests:
```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

`Pkg.test()` is the command to use: it runs the suite in the environment built
from the `test` target, which is what CI runs. Running
`julia --project=. test/runtests.jl` directly also works, but it sees the
ambient environment, so it will not notice a test file using a standard library
that is missing from the `test` target in `Project.toml`.

## Benchmarking

Benchmark scripts live at the repo root. Their dependencies are in
`benchmark/Project.toml` rather than the package's own `Project.toml`, so that
installing Remez does not drag in Gtk4 (via `ProfileView`) for tooling only this
repository uses:

```bash
julia --project=benchmark -e 'using Pkg; Pkg.instantiate()'   # once
julia --project=benchmark benchmark.jl
julia --project=benchmark profile_remez.jl
```

- **[benchmark.jl](benchmark.jl)**: benchmarks the core functions with `BenchmarkTools` and profiles with `Profile`/`ProfileView`.
- **[benchmark2.jl](benchmark2.jl)**: the smaller set of benchmarks (least-squares, `find_extrema`, `winnow_extrema`, minimax).
- **[profile_remez.jl](profile_remez.jl)**: focused profile of the minimax loop.

Internal functions such as `find_extrema` and `winnow_extrema` are not
exported, so the scripts call them qualified (`Remez.find_extrema`).

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs the test suite on
Julia 1.10 (the lower bound from `Project.toml` compat) and 1.13.

## Performance

`RemezWorkspace` pre-allocates every buffer the iteration touches, so the loop
itself allocates nothing; what remains is the returned coefficient copies and
the extrema vector. Measured with `@allocated` on a warm workspace
(`Float64`, `exp` on `[0, 1]`, Julia 1.13):

| fit | workspace call | allocating call |
|---|---|---|
| 3/0 | 1 664 B | 78 995 B |
| 2/2 | 3 184 B | 109 139 B |
| 4/2 | 5 264 B | — |

The dense grid is `256 * (n + d + 1) + 1` Chebyshev nodes, which is what makes
the extrema search reliable at the cost of the evaluation budget.

## License

This package is licensed under the Apache License 2.0.
