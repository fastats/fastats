# fastats
[![Build Status](https://travis-ci.org/fastats/fastats.svg?branch=master)](https://travis-ci.org/fastats/fastats)
[![Build Status (windows)](https://ci.appveyor.com/api/projects/status/9ufvyclit358sfb8/branch/master?svg=true)](https://ci.appveyor.com/project/pawroman/fastats/branch/master)
[![codecov](https://codecov.io/gh/fastats/fastats/branch/master/graph/badge.svg)](https://codecov.io/gh/fastats/fastats)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/2199521147834d58b9f0e8e155c97309)](https://www.codacy.com/app/dave.willmer/fastats?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=fastats/fastats&amp;utm_campaign=Badge_Grade)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A pure Python library for benchmarked, scalable numerics, built using [numba](https://numba.pydata.org/).

[Fastats mailing list](https://groups.google.com/forum/#!forum/fastats)


---

### Latest Release: 2019.1, get it using ``pip install fastats``

## Aims/Reasoning


Current state-of-the-art in numerics / algorithmics / machine learning has many big problems, two of which are:

1. The data is getting bigger and more complex, and code is having trouble scaling to these levels.
2. The code is getting bigger and more complex, and developers are having trouble scaling to these levels.

To fix (1) we need better algorithms, code which vectorises to SIMD instructions, and code which parallelises across CPU cores.

To fix (2) we need to focus on simpler code which is easier to debug.

``fastats`` (ie, fast-stats) tries to help with both of these by; using Linear Algebra for performance optimizations in common functions,
using [numba](https://numba.pydata.org/)
from [Anaconda](https://www.anaconda.com/) to JIT compile the optimized Python code to
vectorised native code, whilst being trivial to run in pure Python mode for debugging.

## Usage

Finding the roots of an equation is central to much of data science and machine learning. For monotonic functions we can use a Newton-Raphson solver to find the root:

```python
from fastats import newton_raphson

def my_func(x):
    return x**3 - x - 1

result = newton_raphson(0.025, 1e-6, root=my_func)
```

This uses [numba](https://numba.pydata.org/) under-the-hood to JIT compile the python code to native code, and uses fastats transforms to call ``my_func`` where required.

However, we usually wish to take a fast function and apply it to a large data set, so ``fastats`` allows you to get the optimized function back as a callable:

```python
newton_opt = newton_raphson(0.025, 1e-6, root=my_func, return_callable=True)

result = newton_opt(0.03, 1e-6)
```

If you profile this you will find it's extremely fast (from a 2015 Macbook Pro):

```bash
>>> %timeit newton_opt(0.03, 1e-6)
785 ns ± 8.2 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
```

compared with SciPy 1.0.1:

 ```bash
 >>> import scipy
 >>> scipy.__version__
 >>> from scipy.optimize import newton
 >>> %timeit newton(my_func, x0=0.03, tol=1e-6)
25.6 µs ± 954 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
 ```


#### What does this show?

Most high-level languages like Python/Lua/Ruby have a formal C-API which allows us to 'drop' down to native code easily (such as SciPy shown above). However, not only is this time-consuming, error-prone and off-putting to many developers, but as you can see from the example above, the specialised C extensions do not automatically scale to larger data.

Through the use of [numba](https://numba.pydata.org/) to JIT-compile the entire function down to native code, we can quickly scale to much larger data sizes without leaving the simplicity of Python.

#### What does fastats actually do?

The secret is in the handling of the function arguments.

When we write C-extensions to high-level languages, we are usually trying to speed up a certain algorithm which is taking too long. This works well for specialised libraries, however in this world of `big` data, the next step is usually `now I want to apply that function to this array of 10 million items`. This is where the C-extension / native library technique falls down.

C-extensions to high-level languages are necessarily limited by the defined API - ie, you can write a C function to take 3 floats, or 3 arrays of floats, but it's very difficult to deal with arbitrary inputs.

``fastats`` allows you to pass functions as arguments into ``numba``, and therefore abstract away the specific looping or concurrency constructs, resulting in faster, cleaner development time, as well as faster execution time.

#### Requirements

Python >= 3.5 only.  Python 3.6 or newer is strongly recommended.

See [setup.py](setup.py) - `install_requires` for installation requirements.

The [contribution guide](.github/CONTRIBUTING.md) contains information on how to install
development requirements.

##### Test requirements

For test requirements, take a look at [.travis.yml](.travis.yml) or [.appveyor.yml](.appveyor.yml).

#### Contributing

Please make sure you've read the contribution guide: [CONTRIBUTING.md](.github/CONTRIBUTING.md)

In short, we use PRs for everything.


#### Sponsors

<img src="http://pico-software.com/images/picosoftware.png" width="300" alt="Pico Software" title="Pico Software"/>
