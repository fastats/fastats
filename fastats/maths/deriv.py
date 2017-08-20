
def root(x):
    return x


def deriv(x, delta):
    first = root(x + delta)
    second = root(x - delta)
    return (first - second) / (2 * delta)
