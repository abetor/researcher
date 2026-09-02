"""A deep-research engine that turns a topic into claims and evidence.

The package contains the harness port, phase orchestrator, source adapters, durable
checkpoint, and CLI. It is stateless: all authoritative state lives in the topic
directory and follows the llmwiki contract. Vendored support code lives in shared/.
"""
__all__ = ["adapters", "checkpoint", "orchestrator", "sources"]
