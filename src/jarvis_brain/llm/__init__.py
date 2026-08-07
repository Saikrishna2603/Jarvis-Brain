"""LLM provider layer for Jarvis Brain.

The LLM layer is reasoning-only in v1. It cannot execute tools directly and
must pass through SecretGuard and routing controls before use.
"""
