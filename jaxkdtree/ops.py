import numpy as np
import jax.numpy as jnp
import jaxlib.mlir.ir as ir
from jaxlib.hlo_helpers import custom_call
from jax.core import Primitive
from jax.interpreters import xla
from jax.interpreters import mlir
from jax.interpreters import batching
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
  if k not in [8, 16, 32, ]:
    raise ValueError(f'k must be in set of predefined values, got {k}')
  
  if not dtypes.issubdtype(operand.dtype, np.floating):
    raise ValueError('operand must be a floating type')

  # Replacing the last dimension which used to be space by the k nearest neighbors
  dims = list(operand.shape)
  dims[-1] = k
  return operand.update(shape=dims, dtype=np.int32)


def kNN_lowering(ctx, x, *, k, max_radius):
  (x_aval,) = ctx.avals_in
  x_type = ir.RankedTensorType(x.type)
  x_shape = x_type.shape
  if len(x_shape) == 2: # No batch dimension
    out_shape = [x_shape[0], k]
    numBatches = 1
    num_points = x_shape[0]
  elif len(x_shape) == 3: # With batch dimension
    out_shape = [x_shape[0], x_shape[1], k]
    numBatches = x_shape[0]
    num_points = x_shape[1]
  else:
    raise ValueError('Input tensor rank should be 2 or 3')

  out_type = ir.RankedTensorType.get(out_shape, ir.IntegerType.get_signless(32))

  opaque = create_kNN_descriptor(num_points, k, max_radius, numBatches)
    
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

def kNN_batching_rule(batched_args, batch_dims, k, max_radius):
  x, = batched_args
  bd, = batch_dims

  # Less naive batching
  x = batching.moveaxis(x, bd, 0)
  return kNN_p.bind(x, k=k, max_radius=max_radius), 0

  # # Naive batching
  # x = batching.moveaxis(x, bd, 0)
  # batched = [kNN_p.bind(x_slice, k=k, max_radius=max_radius) for x_slice in x]
  # return jnp.stack(batched), 0

kNN_p = Primitive("kNN")
kNN_p.def_impl(partial(xla.apply_primitive, kNN_p))
kNN_p.def_abstract_eval(kNN_abstract_eval)
mlir.register_lowering(kNN_p, kNN_lowering, platform="gpu")
batching.primitive_batchers[kNN_p] = kNN_batching_rule