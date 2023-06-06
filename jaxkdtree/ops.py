import numpy as np
import jaxlib.mlir.ir as ir
from jaxlib.hlo_helpers import custom_call
from functools import partial
from jax.core import Primitive
from jax.interpreters import xla
from jax.interpreters import mlir
import jax
from jax.interpreters import ad

from typing import Tuple

def kNN(positions):
    return kNN_p.bind(
        positions
    )

def kNN_abstract_eval(x):
  return x.update(shape=x.shape, dtype=x.dtype)


def kNN_lowering(ctx, x):
  (x_aval,) = ctx.avals_in
  x_type = ir.RankedTensorType(x.type)
  n = len(x_type.shape)
  layout = tuple(range(n - 1, -1, -1))
  return [
      custom_call(
          "kNN",
          [x_type],
          operands=[x],
          operand_layouts=[layout],
          result_layouts=[layout],
          has_side_effect=False,
          operand_output_aliases={0: 0}
      )
  ]

kNN_p = Primitive("kNN")
kNN_p.def_impl(partial(xla.apply_primitive, kNN_p))
kNN_p.def_abstract_eval(kNN_abstract_eval)
mlir.register_lowering(kNN_p, kNN_lowering, platform="gpu")
