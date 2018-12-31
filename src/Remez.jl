module Remez

using Compat

export ratfn_minimax

"""
Evaluate a polynomial.

Arguments:
   coeffs   Array of BigFloats giving the coefficients of the polynomial.
            Starts with the constant term, i.e. coeffs[i] is the
            coefficient of x^(i-1) (because Julia arrays are 1-based).
   x        Point at which to evaluate the polynomial.

Return value is a BigFloat.
"""
function poly_eval(coeffs::Array{BigFloat}, x::BigFloat)
    n = length(coeffs)
    if n == 0
        return BigFloat(0)
    end
    y = coeffs[n]
    for i = n-1:-1:1
        y = coeffs[i] + x*y
    end
    y
end

"""
Evaluate a rational function.

Arguments:
   ncoeffs  Array of BigFloats giving the coefficients of the numerator.
            Starts with the constant term, and 1-based, as above.
   dcoeffs  Array of BigFloats giving the coefficients of the denominator.
            Starts with the constant term, and 1-based, as above.
   x        Point at which to evaluate the function.

Return value is a BigFloat.
"""
function ratfn_eval(ncoeffs::Array{BigFloat}, dcoeffs::Array{BigFloat},
                    x::BigFloat)
    return poly_eval(ncoeffs, x) / poly_eval(dcoeffs, x)
end


# ----------------------------------------------------------------------
# Least-squares fitting of a rational function to a set of (x,y)
# points.
#
# We use this to get an initial starting point for the Remez
# iteration. Therefore, it doesn't really need to be particularly
# accurate; it only needs to be good enough to wiggle back and forth
# across the target function the right number of times (so as to give
# enough error extrema to start optimising from) and not have any
# poles in the target interval.
#
# Least-squares fitting of a _polynomial_ is actually a sensible thing
# to do, and minimises the rms error. Doing the following trick with a
# rational function P/Q is less sensible, because it cannot be made to
# minimise the error function (P/Q-f)^2 that you actually wanted;
# instead it minimises (P-fQ)^2. But that should be good enough to
# have the properties described above.
#
# Some theory: suppose you're trying to choose a set of parameters a_i
# so as to minimise the sum of squares of some error function E_i.
# Basic calculus says, if you do this in one variable, just
# differentiate and solve for zero. In this case, that works fine even
# with multiple variables, because you _partially_ differentiate with
# respect to each a_i, giving a system of equations, and that system
# turns out to be linear so we just solve it as a matrix.
#
# In this case, our parameters are the coefficients of P and Q; to
# avoid underdetermining the system we'll fix Q's constant term at 1,
# so that our error function (as described above) is
#
# E = \sum (p_0 + p_1 x + ... + p_n x^n - y - y q_1 x - ... - y q_d x^d)^2
#
# where the sum is over all (x,y) coordinate pairs. Setting dE/dp_j=0
# (for each j) gives an equation of the form
#
# 0 = \sum 2(p_0 + p_1 x + ... + p_n x^n - y - y q_1 x - ... - y q_d x^d) x^j
#
# and setting dE/dq_j=0 gives one of the form
#
# 0 = \sum 2(p_0 + p_1 x + ... + p_n x^n - y - y q_1 x - ... - y q_d x^d) y x^j
#
# And both of those row types, treated as multivariate linear
# equations in the p,q values, have each coefficient being a value of
# the form \sum x^i, \sum y x^i or \sum y^2 x^i, for various i. (Times
# a factor of 2, but we can throw that away.) So we can go through the
# list of input coordinates summing all of those things, and then we
# have enough information to construct our matrix and solve it
# straight off for the rational function coefficients.

# Arguments:
#    f        The function to be approximated. Maps BigFloat -> BigFloat.
#    xvals    Array of BigFloats, giving the list of x-coordinates at which
#             to evaluate f.
#    n        Degree of the numerator polynomial of the desired rational
#             function.
#    d        Degree of the denominator polynomial of the desired rational
#             function.
#    w        Error-weighting function. Takes two BigFloat arguments x,y
#             and returns a scaling factor for the error at that location.
#             A larger value indicates that the error should be given
#             greater weight in the square sum we try to minimise.
#             If unspecified, defaults to giving everything the same weight.
#
# Return values: a pair of arrays of BigFloats (N,D) giving the
# coefficients of the returned rational function. N has size n+1; D
# has size d+1. Both start with the constant term, i.e. N[i] is the
# coefficient of x^(i-1) (because Julia arrays are 1-based). D[1] will
# be 1.
function ratfn_leastsquares(f::Function, xvals::Array{BigFloat}, n, d,
                            w = (x,y)->BigFloat(1))
    # Accumulate sums of x^i y^j, for j={0,1,2} and a range of x.
    # Again because Julia arrays are 1-based, we'll have sums[i,j]
    # being the sum of x^(i-1) y^(j-1).
    maxpow = max(n,d) * 2 + 1
    sums = zeros(BigFloat, maxpow, 3)
    for x = xvals
        y = f(x)
        weight = w(x,y)
        for i = 1:1:maxpow
            for j = 1:1:3
                sums[i,j] += x^(i-1) * y^(j-1) * weight
            end
        end
    end

    # Build the matrix. We're solving n+d+1 equations in n+d+1
    # unknowns. (We actually have to return n+d+2 coefficients, but
    # one of them is hardwired to 1.)
    matrix = Array{BigFloat}(undef, n+d+1, n+d+1)
    vector = Array{BigFloat}(undef, n+d+1)
    for i = 0:1:n
        # Equation obtained by differentiating with respect to p_i,
        # i.e. the numerator coefficient of x^i.
        row = 1+i
        for j = 0:1:n
            matrix[row, 1+j] = sums[1+i+j, 1]
        end
        for j = 1:1:d
            matrix[row, 1+n+j] = -sums[1+i+j, 2]
        end
        vector[row] = sums[1+i, 2]
    end
    for i = 1:1:d
        # Equation obtained by differentiating with respect to q_i,
        # i.e. the denominator coefficient of x^i.
        row = 1+n+i
        for j = 0:1:n
            matrix[row, 1+j] = sums[1+i+j, 2]
        end
        for j = 1:1:d
            matrix[row, 1+n+j] = -sums[1+i+j, 3]
        end
        vector[row] = sums[1+i, 3]
    end

    # Solve the matrix equation.
    all_coeffs = matrix \ vector

    # And marshal the results into two separate polynomial vectors to
    # return.
    ncoeffs = all_coeffs[1:n+1]
    dcoeffs = vcat([1], all_coeffs[n+2:n+d+1])
    return (ncoeffs, dcoeffs)
end


# ----------------------------------------------------------------------
# Golden-section search to find a maximum of a function.

# Arguments:
#    f        Function to be maximised/minimised. Maps BigFloat -> BigFloat.
#    a,b,c    BigFloats bracketing a maximum of the function.
#
# Expects:
#    a,b,c are in order (either a<=b<=c or c<=b<=a)
#    a != c             (but b can equal one or the other if it wants to)
#    f(a) <= f(b) >= f(c)
#
# Return value is an (x,y) pair of BigFloats giving the extremal input
# and output. (That is, y=f(x).)
function goldensection(f::Function, a::BigFloat, b::BigFloat, c::BigFloat)
    # Decide on a 'good enough' threshold.
    epsbits = precision(BigFloat)
    threshold = abs(c-a) * 2^(-epsbits/2)

    # We'll need the golden ratio phi, of course. Or rather, in this
    # case, we need 1/phi = 0.618...
    one_over_phi = 2 / (1 + sqrt(BigFloat(5)))

    # Flip round the interval endpoints so that the interval [a,b] is
    # at least as large as [b,c]. (Then we can always pick our new
    # point in [a,b] without having to handle lots of special cases.)
    if abs(b-a) < abs(c-a)
        a,  c  = c,  a
    end

    # Evaluate the function at the initial points.
    fa = f(a)
    fb = f(b)
    fc = f(c)

    while abs(c-a) > threshold

        # Check invariants.
        @assert(a <= b <= c || c <= b <= a)
        @assert(fa <= fb >= fc)

        # Subdivide the larger of the intervals [a,b] and [b,c]. We've
        # arranged that this is always [a,b], for simplicity.
        d = a + (b-a) * one_over_phi

        # Now we have an interval looking like this (possibly
        # reversed):
        #
        #    a            d       b            c
        #
        # and we know f(b) is bigger than either f(a) or f(c). We have
        # two cases: either f(d) > f(b), or vice versa. In either
        # case, we can narrow to an interval of 1/phi the size, and
        # still satisfy all our invariants (three ordered points,
        # [a,b] at least the width of [b,c], f(a)<=f(b)>=f(c)).
        fd = f(d)
        if fd > fb
            a,  b,  c  = a,  d,  b
            fa, fb, fc = fa, fd, fb
        else
            a,  b,  c  = c,  b,  d
            fa, fb, fc = fc, fb, fd
        end
    end

    return (b, fb)
end


# ----------------------------------------------------------------------
# Find the extrema of a function within a given interval.

# Arguments:
#    f         The function to be approximated. Maps BigFloat -> BigFloat.
#    grid      A set of points at which to evaluate f. Must be high enough
#              resolution to make extrema obvious.
#
# Returns an array of (x,y) pairs of BigFloats, with each x,y giving
# the extremum location and its value (i.e. y=f(x)).
function find_extrema(f::Function, grid::Array{BigFloat})
    len = length(grid)
    extrema = Tuple{BigFloat, BigFloat}[]
    for i = 1:1:len
        # We have to provide goldensection() with three points
        # bracketing the extremum. If the extremum is at one end of
        # the interval, then the only way we can do that is to set two
        # of the points equal (which goldensection() will cope with).
        prev = max(1, i-1)
        next = min(i+1, len)

        # Find our three pairs of (x,y) coordinates.
        xp, xi, xn = grid[prev], grid[i], grid[next]
        yp, yi, yn = f(xp), f(xi), f(xn)

        # See if they look like an extremum, and if so, ask
        # goldensection() to give a more exact location for it.
        if yp <= yi >= yn
            push!(extrema, goldensection(f, xp, xi, xn))
        elseif yp >= yi <= yn
            x, y = goldensection(x->-f(x), xp, xi, xn)
            push!(extrema, (x, -y))
        end
    end
    return extrema
end


# ----------------------------------------------------------------------
# Winnow a list of a function's extrema to give a subsequence of a
# specified length, with the extrema in the subsequence alternating
# signs, and with the smallest absolute value of an extremum in the
# subsequence as large as possible.
#
# We do this using a dynamic-programming approach. We work along the
# provided array of extrema, and at all times, we track the best set
# of extrema we have so far seen for each possible (length, sign of
# last extremum) pair. Each new extremum is evaluated to see whether
# it can be added to any previously seen best subsequence to make a
# new subsequence that beats the previous record holder in its slot.

# Arguments:
#    extrema   An array of (x,y) pairs of BigFloats giving the input extrema.
#    n         Number of extrema required as output.
#
# Returns a new array of (x,y) pairs which is a subsequence of the
# original sequence. (So, in particular, if the input was sorted by x
# then so will the output be.)
function winnow_extrema(extrema::Array{Tuple{BigFloat,BigFloat}}, n)
    # best[i,j] gives the best sequence so far of length i and with
    # sign j (where signs are coded as 1=positive, 2=negative), in the
    # form of a tuple (cost, actual array of x,y pairs).
    best = fill((BigFloat(0), Tuple{BigFloat,BigFloat}[]), n, 2)

    for (x,y) = extrema
        if y > 0
            sign = 1
        elseif y < 0
            sign = 2
        else
            # A zero-valued extremum cannot possibly contribute to any
            # optimal sequence, so we simply ignore it!
            continue
        end

        for i = 1:1:n
            # See if we can create a new entry for best[i,sign] by
            # appending our current (x,y) to some previous thing.
            if i == 1
                # Special case: we don't store a best zero-length
                # sequence :-)
                candidate = (abs(y), [(x,y)])
            else
                othersign = 3-sign # map 1->2 and 2->1
                oldscore, oldlist = best[i-1, othersign]
                newscore = min(abs(y), oldscore)
                newlist = vcat(oldlist, [(x,y)])
                candidate = (newscore, newlist)
            end
            # If our new candidate improves on the previous value of
            # best[i,sign], then replace it.
            if candidate[1] > best[i,sign][1]
                best[i,sign] = candidate
            end
        end
    end

    # Our ultimate return value has to be either best[n,1] or
    # best[n,2], but it could be either. See which one has the higher
    # score.
    if best[n,1][1] > best[n,2][1]
        ret = best[n,1][2]
    else
        ret = best[n,2][2]
    end
    # Make sure we did actually _find_ a good answer.
    @assert(length(ret) == n)
    return ret
end

# ----------------------------------------------------------------------
# Construct a rational-function approximation with equal and
# alternating weighted deviation at a specific set of x-coordinates.

# Arguments:
#    f         The function to be approximated. Maps BigFloat -> BigFloat.
#    coords    An array of BigFloats giving the x-coordinates. There should
#              be n+d+2 of them.
#    n, d      The degrees of the numerator and denominator of the desired
#              approximation.
#    prev_err  A plausible value for the alternating weighted deviation.
#              (Required to kickstart a binary search in the nonlinear case;
#              see comments below.)
#    w         Error-weighting function. Takes two BigFloat arguments x,y
#              and returns a scaling factor for the error at that location.
#              The returned approximation R should have the minimum possible
#              maximum value of abs((f(x)-R(x)) * w(x,f(x))). Optional
#              parameter, defaulting to the always-return-1 function.
#
# Return values: a pair of arrays of BigFloats (N,D) giving the
# coefficients of the returned rational function. N has size n+1; D
# has size d+1. Both start with the constant term, i.e. N[i] is the
# coefficient of x^(i-1) (because Julia arrays are 1-based). D[1] will
# be 1.
function ratfn_equal_deviation(f::Function, coords::Array{BigFloat},
                               n, d, prev_err::BigFloat,
                               w = (x,y)->BigFloat(1))
    @assert(length(coords) == n+d+2)

    if d == 0
        # Special case: we're after a polynomial. In this case, we
        # have the particularly easy job of just constructing and
        # solving a system of n+2 linear equations, to find the n+1
        # coefficients of the polynomial and also the amount of
        # deviation at the specified coordinates. Each equation is of
        # the form
        #
        #   p_0 x^0 + p_1 x^1 + ... + p_n x^n ± e/w(x) = f(x)
        #
        # in which the p_i and e are the variables, and the powers of
        # x and calls to w and f are the coefficients.

        matrix = Array{BigFloat}(undef, n+2, n+2)
        vector = Array{BigFloat}(undef, n+2)
        currsign = +1
        for i = 1:1:n+2
            x = coords[i]
            for j = 0:1:n
                matrix[i,1+j] = x^j
            end
            y = f(x)
            vector[i] = y
            matrix[i, n+2] = currsign / w(x,y)
            currsign = -currsign
        end

        outvector = matrix \ vector

        ncoeffs = outvector[1:n+1]
        dcoeffs = [BigFloat(1)]
        return ncoeffs, dcoeffs
    else
        # For a nontrivial rational function, the system of equations
        # we need to solve becomes nonlinear, because each equation
        # now takes the form
        #
        #   p_0 x^0 + p_1 x^1 + ... + p_n x^n
        #   --------------------------------- ± e/w(x) = f(x)
        #     x^0 + q_1 x^1 + ... + q_d x^d
        #
        # and multiplying up by the denominator gives you a lot of
        # terms containing e × q_i. So we can't do this the really
        # easy way using a matrix equation as above.
        #
        # Fortunately, this is a fairly easy kind of nonlinear system.
        # The equations all become linear if you switch to treating e
        # as a constant, so a reasonably sensible approach is to pick
        # a candidate value of e, solve all but one of the equations
        # for the remaining unknowns, and then see what the error
        # turns out to be in the final equation. The Chebyshev
        # alternation theorem guarantees that that error in the last
        # equation will be anti-monotonic in the input e, so we can
        # just binary-search until we get the two as close to equal as
        # we need them.

        function try_e(e)
            # Try a given value of e, derive the coefficients of the
            # resulting rational function by setting up equations
            # based on the first n+d+1 of the n+d+2 coordinates, and
            # see what the error turns out to be at the final
            # coordinate.
            matrix = Array{BigFloat}(undef, n+d+1, n+d+1)
            vector = Array{BigFloat}(undef, n+d+1)
            currsign = +1
            for i = 1:1:n+d+1
                x = coords[i]
                y = f(x)
                y_adj = y - currsign * e / w(x,y)
                for j = 0:1:n
                    matrix[i,1+j] = x^j
                end
                for j = 1:1:d
                    matrix[i,1+n+j] = -x^j * y_adj
                end
                vector[i] = y_adj
                currsign = -currsign
            end

            outvector = matrix \ vector

            ncoeffs = outvector[1:n+1]
            dcoeffs = vcat([BigFloat(1)], outvector[n+2:n+d+1])

            x = coords[n+d+2]
            y = f(x)
            last_e = (ratfn_eval(ncoeffs, dcoeffs, x) - y) * w(x,y) * -currsign

            return ncoeffs, dcoeffs, last_e
        end

        epsbits = precision(BigFloat)
        threshold = 2^(-epsbits/2) # convergence threshold

        # Start by trying our previous iteration's error value. This
        # value (e0) will be one end of our binary-search interval,
        # and whatever it caused the last point's error to be, that
        # (e1) will be the other end.
        e0 = prev_err
        nc, dc, e1 = try_e(e0)
        if abs(e1-e0) <= threshold
            # If we're _really_ lucky, we hit the error right on the
            # nose just by doing that!
            return nc, dc
        end
        s = sign(e1-e0)

        # Verify by assertion that trying our other interval endpoint
        # e1 gives a value that's wrong in the other direction.
        # (Otherwise our binary search won't get a sensible answer at
        # all.)
        nc, dc, e2 = try_e(e1)
        @assert(sign(e2-e1) == -s)

        # Now binary-search until our two endpoints narrow enough.
        local emid
        while abs(e1-e0) > threshold
            emid = (e1+e0)/2
            nc, dc, enew = try_e(emid)
            if sign(enew-emid) == s
                e0 = emid
            else
                e1 = emid
            end
        end

        return nc, dc
    end
end


"""
    N,D,E,X = ratfn_minimax(f, interval, n, d, w)

Top-level function to find a minimax rational-function approximation.

Arguments:

 * f         The function to be approximated. Maps BigFloat -> BigFloat.
 * interval  A tuple giving the endpoints of the interval
             (in either order) on which to approximate f.
 * n, d      The degrees of the numerator and denominator of the desired
             approximation.
 * w         Error-weighting function. Takes two BigFloat arguments x,y
             and returns a scaling factor for the error at that location.
             The returned approximation R should have the minimum possible
             maximum value of abs((f(x)-R(x)) * w(x,f(x))). Optional
             parameter, defaulting to the always-return-1 function.

Return values: a tuple (N,D,E,X), where

 * N,D       A pair of arrays of BigFloats giving the coefficients
             of the returned rational function. N has size n+1; D
             has size d+1. Both start with the constant term, i.e.
             N[i] is the coefficient of x^(i-1) (because Julia
             arrays are 1-based). D[1] will be 1.
 * E         The maximum weighted error (BigFloat).
 * X         An array of pairs of BigFloats giving the locations of n+2
             points and the weighted error at each of those points. The
             weighted error values will have alternating signs, which
             means that the Chebyshev alternation theorem guarantees
             that any other function of the same degree must exceed
             the error of this one at at least one of those points.
"""
function ratfn_minimax(f, interval, n, d,
                       w = (x,y)->BigFloat(1))
    # We start off by finding a least-squares approximation. This
    # doesn't need to be perfect, but if we can get it reasonably good
    # then it'll save iterations in the refining stage.
    #
    # Least-squares approximations tend to look nicer in a minimax
    # sense if you evaluate the function at a big pile of Chebyshev
    # nodes rather than uniformly spaced points. These values will
    # also make a good grid to use for the initial search for error
    # extrema, so we'll keep them around for that reason too.

    # Construct the grid.
    lo = BigFloat(minimum(interval))
    hi = BigFloat(maximum(interval))

    local grid
    let
        mid = (hi+lo)/2
        halfwid = (hi-lo)/2
        nnodes = 16 * (n+d+1)
        grid = [ mid - halfwid * cospi(big(i)/big(nnodes)) for i=0:nnodes ]
    end

    # Find the initial least-squares approximation.
    (nc, dc) = ratfn_leastsquares(f, grid, n, d, w)

    # Threshold of convergence. We stop when the relative difference
    # between the min and max (winnowed) error extrema is less than
    # this.
    #
    # This is set to the cube root of machine epsilon on a more or
    # less empirical basis, because the rational-function case will
    # not converge reliably if you set it to only the square root.
    # (Repeatable by using the --test mode.) On the assumption that
    # input and output error in each iteration can be expected to be
    # related by a simple power law (because it'll just be down to how
    # many leading terms of a Taylor series are zero), the cube root
    # was the next thing to try.
    epsbits = precision(BigFloat)
    threshold = 2^(-epsbits/3)

    # Main loop.
    while true
        # Find all the error extrema we can.
        function compute_error(x)
            real_y = f(x)
            approx_y = ratfn_eval(nc, dc, x)
            return (approx_y - real_y) * w(x, real_y)
        end
        extrema = find_extrema(compute_error, grid)

        # Winnow the extrema down to the right number, and ensure they
        # have alternating sign.
        extrema = winnow_extrema(extrema, n+d+2)

        # See if we've finished.
        min_err = minimum([abs(y) for (x,y) = extrema])
        max_err = maximum([abs(y) for (x,y) = extrema])
        variation = (max_err - min_err) / max_err
        if variation < threshold
            return nc, dc, max_err, extrema
        end

        # If not, refine our function by equalising the error at the
        # extrema points, and go round again.
        (nc, dc) = ratfn_equal_deviation(f, map(x->x[1], extrema),
                                         n, d, max_err, w)
    end
end

# ----------------------------------------------------------------------
# Check if a polynomial is well-conditioned for accurate evaluation in
# a given interval by Horner's rule.
#
# This is true if at every step where Horner's rule computes
# (coefficient + x*value_so_far), the constant coefficient you're
# adding on is of larger magnitude than the x*value_so_far operand.
# And this has to be true for every x in the interval.
#
# Arguments:
#    coeffs    The coefficients of the polynomial under test. Starts with
#              the constant term, i.e. coeffs[i] is the coefficient of
#              x^(i-1) (because Julia arrays are 1-based).
#    lo, hi    The bounds of the interval.
#
# Return value: the largest ratio (x*value_so_far / coefficient), at
# any step of evaluation, for any x in the interval. If this is less
# than 1, the polynomial is at least somewhat well-conditioned;
# ideally you want it to be more like 1/8 or 1/16 or so, so that the
# relative rounding error accumulated at each step are reduced by
# several factors of 2 when the next coefficient is added on.

function wellcond(coeffs, lo, hi)
    x = max(abs(lo), abs(hi))
    worst = 0
    so_far = 0
    for i = length(coeffs):-1:1
        coeff = abs(coeffs[i])
        so_far *= x
        if coeff != 0
            thisval = so_far / coeff
            worst = max(worst, thisval)
            so_far += coeff
        end
    end
    return worst
end


end # module
