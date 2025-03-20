import os
__all__ = [
        'Simulator',
        'Sampler',
        'PROJECT_ROOT',
        ]
from .sampler import Sampler
from .simulator import Simulator

# get the root of the project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
