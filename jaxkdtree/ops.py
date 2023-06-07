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
  return x.update(shape=x.shape, dtype=np.int32)


def kNN_lowering(ctx, x):
  (x_aval,) = ctx.avals_in
  x_type = ir.RankedTensorType(x.type)
  x_shape = x_type.shape
  out_type = ir.RankedTensorType.get(x_shape, ir.IntegerType.get_signless(32))
  n = len(out_type.shape)
  layout = tuple(range(n - 1, -1, -1))
  return [
      custom_call(
          "kNN",
          [out_type],
          operands=[x, x],
          operand_layouts=[layout, layout],
          result_layouts=[layout]
      )
  ]

kNN_p = Primitive("kNN")
kNN_p.def_impl(partial(xla.apply_primitive, kNN_p))
kNN_p.def_abstract_eval(kNN_abstract_eval)
mlir.register_lowering(kNN_p, kNN_lowering, platform="gpu")
