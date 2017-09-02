
from numba import jit
from numba.targets.registry import CPUDispatcher


def convert_to_jit(func):
    if isinstance(func, CPUDispatcher):
        return func

    _jit = jit(nopython=True, nogil=True)
    return _jit(func)


if __name__ == '__main__':
    import pytest
    pytest.main()
