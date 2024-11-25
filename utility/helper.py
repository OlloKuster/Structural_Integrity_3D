import jax.numpy as jnp

def split_int(a):
    """
    Splits a into two parts as close as possible to the middle.
    :param a: Value to be split in the middle.
    :return: Tuple of both halves of a.
    """
    return a // 2, a // 2 + a % 2


def softplus(x, beta=50):
    """
    Softplus function to balance out the optimisation.
    :param x: Array of figure of merits which are evaluated.
    :param beta: Steepness of the curve.
    :return: Array of softplus of all figure of merits.
    """
    mask = x * beta > 20
    return jnp.where(mask, x, 1 / beta * jnp.log(1 + jnp.exp(jnp.where(mask, 0, x * beta))))