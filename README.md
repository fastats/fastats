# fastats
[![Build Status](https://travis-ci.org/fastats/fastats.svg?branch=master)](https://travis-ci.org/fastats/fastats)
[![codecov](https://codecov.io/gh/fastats/fastats/branch/master/graph/badge.svg)](https://codecov.io/gh/fastats/fastats)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/2199521147834d58b9f0e8e155c97309)](https://www.codacy.com/app/dave.willmer/fastats?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=fastats/fastats&amp;utm_campaign=Badge_Grade)


A pure python library for benchmarked, scalable numerics, built using [numba](http://numba.pydata.org/).

---

### **This is pre-release software, there are no packages published yet.**

## Aims/Reasoning


Current state-of-the-art in numerics / algorithmics / machine learning has many big problems, two of which are:

1. The data is getting bigger and more complex, and code is having trouble scaling to these levels.
2. The code is getting bigger and more complex, and developers are having trouble scaling to these levels.

To fix (1) we need better algorithms, code which vectorises to SIMD instructions, and code which parallelises across CPU cores.

To fix (2) we need to focus on simpler code which is easier to debug.

fastats (ie, fast-stats) tries to help with both of these by using [numba](http://numba.pydata.org/) from [Continuum Analytics](https://www.continuum.io/) to JIT compile pure python code to vectorised native code, whilst being trivial to run in pure python mode for debugging.

## Usage

Finding the roots of an equation is central to much of machine learning. For monotonic functions we can use a Newton-Raphson solver to find the root:

```python
from fastats.api import newton_raphson

def my_func(x):
    return x**3 - x - 1

result = newton_raphson(0.025, 1e-6, root=my_func)
```


```bash
>>> %timeit newton_raphson(0.025, 1e-6, root=my_func)

```

compared with scipy 0.12 ...

 ```bash
 >>> import scipy
 >>> scipy.__version__
 >>> from scipy.optimize import newton
 >>> %timeit newton(my_function, x0=0.025)

 ```


#### What does this show?

Most high-level languages like Python/Lua/Ruby have a formal C-API which allows us to 'drop' down to native code easily (such as scipy shown above). However, not only is this time-consuming, error-prone and off-putting to many developers, but as you can see from the example above, the specialised C extensions do not automatically scale to larger data.

Through the use of [numba](http://numba.pydata.org/) to JIT-compile the entire function down to native code, we can quickly scale to much larger data sizes without leaving the simplicity of python.

#### What does fastats actually do?

The secret is in the handling of the function arguments.

When we write C-extensions to high-level languages, we are usually trying to speed up a certain algorithm which is taking too long. This works well for specialised libraries, however in this world of `big` data, the next step is usually `now i want to apply that function to this array of 10 million items`. This is where the C-extension / native library technique falls down.


C-extensions to high-level languages are necessarily limited by the defined API - ie, you can write a C function to take 3 floats, or 3 arrays of floats, but it's very difficult to deal with arbitrary inputs.

#### Requirements

- Python >= 3.5
- Numba >= 0.33


#### Contributing

All contributions are welcome :)

If you would like to contribute anything, open a PR (issues are turned off):

- To report a bug, open a PR with a unittest that fails.
- To request an API change/new functionality, open a PR with a failing unittest showing your preferred API.
- To submit a fix, open a PR with passing unittests + doctests. Doctests should be minimal, and serve as API docs for the most common use cases, unittests should be exhaustive.

Simples :)

#### Sponsors

![Pico Software](http://pico-software.com/images/picosoftware.png | width=300)
