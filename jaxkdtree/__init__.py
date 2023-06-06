from jax.lib import xla_client
from . import _jaxkdtree

# Registering ops for XLA
for name, fn in _jaxkdtree.registrations().items():
  xla_client.register_custom_call_target(name, fn, platform="gpu")

from .ops import kNN