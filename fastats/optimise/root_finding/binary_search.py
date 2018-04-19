
from fastats import fastfunc
from fastats.basic import identity


@fastfunc
def binary_search(a, b, delta, root=identity):
    x = (a + b) / 2

    while (b - a) / 2 > delta:
        res = root(x)

        if res == 0.0:
            return x
        # TODO : try faster method of
        # comparing signs using xor.
        elif root(a) * res < 0:
            b = x
        else:
            a = x

        x = (a + b) / 2

    return x


if __name__ == '__main__':
    import pytest
    pytest.main([__file__])
