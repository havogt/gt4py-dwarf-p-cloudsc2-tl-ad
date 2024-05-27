import gt4py.next as gtx
from gt4py.next import common

I2 = gtx.Dimension("I")
J2 = gtx.Dimension("J")
K2 = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)


def as_field(*dims: gtx.Dimension, use_jax=False):
    from jax import numpy as jnp

    def impl(arr) -> gtx.Field:
        domain = common.Domain(
            dims=dims, ranges=[common.unit_range(s) for s in arr.shape]
        )
        if use_jax:
            return common._field(jnp.asarray(arr), domain=domain)
        else:
            return common._field(arr, domain=domain)

    return impl
