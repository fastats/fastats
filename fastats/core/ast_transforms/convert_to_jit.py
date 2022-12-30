
from inspect import isbuiltin, isfunction
from types import MappingProxyType

from numba import jit
from numba.core.target_extension import CPUDispatcher


JIT_KWARGS = MappingProxyType({
    'nopython': True, 'nogil': True
})


def convert_to_jit(func):
    if isinstance(func, CPUDispatcher) or isbuiltin(func):
        return func

    if not isfunction(func):
        raise TypeError("Can't JIT a non-function object: {}".format(func))

    _jit = jit(**JIT_KWARGS)

    return _jit(func)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
