

def is_2d(n):
    return len(n.shape) == 2


def is_square(n):
    square = n.shape[0] == n.shape[1]
    return is_2d(n) and square
