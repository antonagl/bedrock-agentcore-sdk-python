"""Integration tests for the Strands AgentCoreCodeInterpreterSandbox.

Run with:
    uv run pytest tests_integ/tools/test_code_interpreter_sandbox.py -xvs
"""

import os

import pytest
from strands.sandbox import ExecutionResult, StreamChunk
from strands.sandbox.errors import SandboxPathNotFoundError

from bedrock_agentcore.tools.integrations.strands import AgentCoreCodeInterpreterSandbox

PAYLOAD = b"\x89PNG\r\n\x1a\n" + bytes(range(59))


@pytest.fixture
async def sandbox():
    region = os.environ.get("BEDROCK_TEST_REGION", "us-west-2")
    async with AgentCoreCodeInterpreterSandbox(region=region) as sb:
        yield sb


@pytest.mark.integration
class TestCodeInterpreterSandbox:
    async def test_execute(self, sandbox):
        result = await sandbox.execute("echo hello && echo oops >&2 && exit 3")
        assert result.exit_code == 3
        assert "hello" in result.stdout
        assert "oops" in result.stderr

    async def test_execute_streaming_ends_with_result(self, sandbox):
        chunks = [c async for c in sandbox.execute_streaming("echo streamed")]
        assert isinstance(chunks[-1], ExecutionResult)
        assert any(isinstance(c, StreamChunk) and "streamed" in c.data for c in chunks)

    async def test_execute_cwd_and_env(self, sandbox):
        result = await sandbox.execute('echo "$GREETING" && pwd', cwd="/tmp", env={"GREETING": "hi there"})
        assert result.exit_code == 0
        assert "hi there" in result.stdout
        assert "/tmp" in result.stdout

    async def test_execute_code_native_keeps_context(self, sandbox):
        await sandbox.execute_code("x = 41", "python3")
        result = await sandbox.execute_code("print(x + 1)", "python3")
        assert "42" in result.stdout

    async def test_execute_code_shell_fallback(self, sandbox):
        result = await sandbox.execute_code("echo from-bash", "bash")
        assert result.exit_code == 0
        assert "from-bash" in result.stdout

    async def test_file_round_trip(self, sandbox):
        await sandbox.write_file("sandbox_integ/data.bin", PAYLOAD)
        assert await sandbox.read_file("sandbox_integ/data.bin") == PAYLOAD

        entries = {e.name: e for e in await sandbox.list_files("sandbox_integ")}
        assert "data.bin" in entries
        assert entries["data.bin"].is_dir is False

        parent = {e.name: e for e in await sandbox.list_files("")}
        assert parent["sandbox_integ"].is_dir is True

        await sandbox.remove_file("sandbox_integ/data.bin")
        assert "data.bin" not in {e.name for e in await sandbox.list_files("sandbox_integ")}

    async def test_read_missing_file(self, sandbox):
        with pytest.raises(FileNotFoundError):
            await sandbox.read_file("does/not/exist.txt")

    async def test_list_missing_directory(self, sandbox):
        with pytest.raises(SandboxPathNotFoundError):
            await sandbox.list_files("does/not/exist")
