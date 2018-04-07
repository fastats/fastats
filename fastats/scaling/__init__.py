
from fastats.scaling.scaling import (standard, min_max, rank, scale, demean, standard_parallel, min_max_parallel,
                                     demean_parallel, shrink_off_diagonals)


__all__ = [
    'standard',
    'standard_parallel',
    'min_max',
    'min_max_parallel',
    'rank',
    'scale',
    'demean',
    'demean_parallel',
    'shrink_off_diagonals',
]
