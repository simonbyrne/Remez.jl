# Remez.jl

[![Travis build status](https://travis-ci.org/simonbyrne/Remez.jl.svg?branch=master)](https://travis-ci.org/simonbyrne/Remez.jl)

This is an implementation of the [Remez algorithm](https://en.wikipedia.org/wiki/Remez_algorithm) for computing minimax polynomial approximations to functions.

It is largely based on [code by ARM](https://github.com/ARM-software/optimized-routines/blob/da55ef9510a53822b5706c61ad97795828999c80/auxiliary/remez.jl), but updated for newer Julia versions and built into a package.

The main function is `ratfn_minimax`, see help for more details.

