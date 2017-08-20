

def clip(x):
    """
    Performs a `clip`, or `cap and floor` of
    the input data.

    If the value is outside the range of [-x, x],
    then either -x or x is returned, capping and
    flooring the data.

    :param x:
    :return:
    """
    def _inner(val):
        if val > x:
            return x
        if val < -x:
            return -x
        return val
    return _inner
