"""Duplex HTTP API 端点说明。

端点:
  GET  /health
  POST /v1/stream/omni_init
  POST /v1/stream/prefill
  POST /v1/stream/decode
  POST /v1/stream/break
  POST /v1/stream/update_session_config  (sampling)
"""

from .duplex import DuplexSession

__all__ = ["DuplexSession"]
