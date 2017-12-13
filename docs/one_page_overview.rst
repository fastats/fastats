# Fastats

Fastats is designed to allow extremely high performance numerical code
that only requires knowledge of python.

It achieves this through:

- Passing functions as arguments into [numba](http://numba.pydata.org).
- A core library of numerical functions that compose easily.

[Numba](http://numba.pydata.org) is an excellent library which JIT compiles
numerical python functions down to native code. It also makes concurrency
easy, and therefore is amazingly useful for numerical programming.

With fastats, functions can be wrapped with the `@fs` decorator which transforms
the function to allow any keyword arguments to be passed; these keyword arguments
specify the functions to be replaced within the original code.

When you specify a keyword argument, the code is transformed into AST form,
and is then re-built with the specified functions replaced. This leads to
extremely fast execution times (due to `numba`), and extremely fast development
times (due to AST replacement of functions).

# How do the AST Transforms work?

Imagine a nested function such as a very simple Newton-Raphson solver:

::

    def newton_raphson(x0, delta, root=fs_func, deriv=deriv):
        last_x = x0
        next_x = last_x + 10 * delta
        while abs(last_x - next_x) > delta:
            new_y = root(next_x)
            last_x = next_x
            next_x = last_x - new_y / deriv(last_x, delta)
        return next_x


In the code above we actually want to replace the `root` function with one of our own
choosing, without having to re-write the entire newton_raphson wrapper function.

In `fastats` this is performed by changing the semantics of positional and keyword
arguments; `numba` does not allow us to pass functions as arguments, but even if it
did, we still need the ability to arbitrarily modify deeply nested functions (use
cases discussed below).

The Newton-Raphson code shown above is in the `fastats` standard library, and allows us
to do the following:

::

    from fastats.optimise.root_finding.newton_raphson import newton_raphson

    def my_func(x, y):
        return x**2 + y**2

    my_solver = newton_raphson(, 0.001, root=my_func)

    my_solver(3, 4)

`my_func` is the function for which we would like to find the roots. It takes two arguments,
`x` and `y`, one of which will be kept constant whilst the other is varied to find the root.

When `newton_raphson` is called, it takes the `root=my_func` kwarg, and inspects the signature
of `my_func`. It finds that my_func takes 2 arguments `x` and `y`, and expects the first argument
(`x` in this case) to be the parameter that is modified by the algorithm to find the root.

As a result, we need to be able to pass `y` from the top level caller all the way down to the
places where 'root' is called. For example, line 5 in the example above needs to be changed to

::

    new_y = my_func(next_x, y)

In order to do this, the `newton_raphson` function needs its signature modified to take `y`
as an argument. This is what `fastats` performs: at any level in the AST, `fastats` will modify
the function signatures and ensure that the correct arguments are passed to all functions,
in order to allow any function to be modified by passing it as a keyword argument at the
top-level.

In this example, `deriv` will numerically calculate the derivative at each point, however we
can trivially calculate the analytical derivative by hand, and replace the `deriv` function
like this:

::
    def my_deriv(x, y):
        return 2 * x + 2 * y

    my_solve = newton_raphson(, 0.001, root=my_func, deriv=my_deriv)

    my_solve(3, 4)

Which allows us to **optionally** pass an optimised function, but fall back on a non-optimised
version for experimenting/research.

This is not-limited to specific functions - if you are happy with lower-precision in certain
calculations, you can pass faster (lower-precision) versions of any mathematical functions,
and `fastats` will replace them throughout the entire AST before sending onto numba, without
requiring you to modify any code.

As an example, some calculations require the complementary error function `erfc` to be calculated.
The accuracy (precision) of `erfc` depends partially on how many terms are in the multiplication. By
reducing the number of terms we can speed up calculations at the expense of accuracy.

If you are happy with 8 decimal places, you can speed up calculations by passing `erfc8`:

::
    from fastats.core.erfc import erfc8

    my_solve = newton_raphson(, 0.000, root=my_func, erfc=erfc8)

To increate precision (at the expense of calculation time), you could use `erfc16`:

::
    from fastats.core.erfc import erfc16

    my_solve = newton_raphson(, 0.001, root=my_func, deriv=my_deriv, erfc=erfc16)

These `ast-modification` semantics therefore allow you to use any pure python numerical
code, regardless of whether the original author allowed arbitrary functions to be passed in.

# Why are we re-writing functions that already exist in numpy/scipy/etc?

One of the major advantages of numba is that the JIT compilation will be
optimized for the specific hardware you are running on. The traditional
python system of writing a C or Cython routine and then pre-compiling it
will not be optimised for specific hardware.

Over the last few years, the SIMD registers in CPUs have grown to 512-bits,
which numba/fastats code will be able to take advantage of. Older builds of
numpy/scipy will not.

This library therefore attempts to move as much logic into small reusable
python functions which can be compiled as required using numba. We do not
use pre-compiled code.

This system also allows us to have lower or higher precision variants of
functions if we want to control the runtime and/or accuracy. With compiled
code this adds a huge overhead to the codebase.
