from jax.lib import xla_client
from . import _jaxkdtree

# Registering ops for XLA
for name, fn in _jaxkdtree.registrations().items():
  xla_client.register_custom_call_target(name, fn, platform="gpu")

create_kNN_descriptor = _jaxkdtree.create_kNN_descriptor

from .ops import kNN