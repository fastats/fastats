
from fastats.core.decorator import fs
from fastats.core.single_pass import single_pass
from fastats.core.windowed_pass import windowed_pass, windowed_pass_2d
from fastats.core.windowed_stateful_pass import windowed_stateful_pass
from fastats.optimise.newton_raphson import newton_raphson
from ._version import VERSION


__all__ = [
    'fs',
    'single_pass',
    'windowed_pass',
    'windowed_pass_2d',
    'windowed_stateful_pass',
    'newton_raphson',
]


__version__ = VERSION
