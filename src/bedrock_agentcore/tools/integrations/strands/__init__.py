"""Strands Agents integrations for AgentCore built-in tools.

Requires the ``strands-agents`` extra: ``pip install 'bedrock-agentcore[strands-agents]'``.
"""

from .code_interpreter_sandbox import AgentCoreCodeInterpreterSandbox

__all__ = ["AgentCoreCodeInterpreterSandbox"]
