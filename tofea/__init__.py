__version__ = "0.1.0"

import jax

jax.config.update("jax_enable_x64", True)

DEFAULT_SOLVER = "SuperLU"
