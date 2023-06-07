import numpy as np
import jax.numpy as jnp
import jaxlib.mlir.ir as ir
from jaxlib.hlo_helpers import custom_call
from jax.core import Primitive
from jax.interpreters import xla
from jax.interpreters import mlir
import jax
from jax.interpreters import ad
from jax import dtypes

from jaxkdtree import create_kNN_descriptor

from functools import partial

def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]

def kNN(x, k=8, max_radius=1.):
    """
    x: tensor of shape [N, 3]
    k: int giving you the number of k nearest neighbors to search for. Can be one of [4, 8, 16, 32]
    radius: float with the maximum radius to consider in the NN search
    """
    return kNN_p.bind(
        x, k=k, max_radius=max_radius
    )

def kNN_abstract_eval(operand, *, k, max_radius):
  if k not in [ 8, ]:
    raise ValueError(f'k must be in set of predefined values, got {k}')
  
  if not dtypes.issubdtype(operand.dtype, np.floating):
    raise ValueError('operand must be a floating type')

  # Replacing the last dimension which used to be space by the k nearest neighbors
  dims = list(operand.shape)
  assert len(dims) == 2 
  dims[-1] = k
  return operand.update(shape=dims, dtype=np.int32)


def kNN_lowering(ctx, x, *, k, max_radius):
  (x_aval,) = ctx.avals_in
  x_type = ir.RankedTensorType(x.type)
  x_shape = x_type.shape
  out_shape = [x_shape[0], k]
  out_type = ir.RankedTensorType.get(out_shape, ir.IntegerType.get_signless(32))

  opaque = create_kNN_descriptor(x_shape[0], k, max_radius)
  
  return [
      custom_call(
          "kNN",
          [out_type],
          operands=[x],
          backend_config=opaque,
          operand_layouts=default_layouts(x_shape),
          result_layouts=default_layouts(out_shape)
      )
  ]

kNN_p = Primitive("kNN")
kNN_p.def_impl(partial(xla.apply_primitive, kNN_p))
kNN_p.def_abstract_eval(kNN_abstract_eval)
mlir.register_lowering(kNN_p, kNN_lowering, platform="gpu")
