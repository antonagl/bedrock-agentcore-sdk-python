"""Unit tests for AgentCoreCodeInterpreterSandbox."""

import base64
import time
from unittest.mock import MagicMock, patch

import pytest
from strands.sandbox import ExecutionResult, FileInfo, Sandbox, StreamChunk
from strands.sandbox.errors import SandboxPathNotFoundError, SandboxTimeoutError

from bedrock_agentcore.tools.integrations.strands import AgentCoreCodeInterpreterSandbox
from bedrock_agentcore.tools.integrations.strands.code_interpreter_sandbox import (
    CodeInterpreterInvocationError,
)


def _response(*results, exception=None):
    events = [{"result": r} for r in results]
    if exception:
        events.append(exception)
    return {"sessionId": "sess-1", "stream": events}


def _exec(stdout="", stderr="", exit_code=0, is_error=False, content=None):
    return {
        "content": content or [],
        "structuredContent": {"stdout": stdout, "stderr": stderr, "exitCode": exit_code},
        "isError": is_error,
    }


@pytest.fixture
def client():
    c = MagicMock()
    c.session_id = None
    c.identifier = None

    def _start(**kwargs):
        c.session_id = "sess-1"
        c.identifier = kwargs.get("identifier")
        return "sess-1"

    def _stop():
        c.session_id = None
        c.identifier = None
        return True

    c.start.side_effect = _start
    c.stop.side_effect = _stop
    return c


@pytest.fixture
def sandbox(client):
    return AgentCoreCodeInterpreterSandbox(client=client, session_name="test")


class TestConstruction:
    def test_is_strands_sandbox(self, sandbox):
        assert isinstance(sandbox, Sandbox)

    @patch("bedrock_agentcore.tools.integrations.strands.code_interpreter_sandbox.CodeInterpreter")
    def test_creates_client_with_region(self, mock_ci):
        sb = AgentCoreCodeInterpreterSandbox("us-west-2")
        mock_ci.assert_called_once_with("us-west-2", session=None, integration_source="strands")
        assert sb.client is mock_ci.return_value

    @patch("bedrock_agentcore.tools.integrations.strands.code_interpreter_sandbox.CodeInterpreter")
    def test_region_from_boto_session(self, mock_ci):
        session = MagicMock(region_name="eu-west-1")
        AgentCoreCodeInterpreterSandbox(session=session)
        mock_ci.assert_called_once_with("eu-west-1", session=session, integration_source="strands")

    @patch("bedrock_agentcore.tools.integrations.strands.code_interpreter_sandbox.boto3.Session")
    def test_missing_region_raises(self, mock_session):
        mock_session.return_value.region_name = None
        with pytest.raises(ValueError, match="region"):
            AgentCoreCodeInterpreterSandbox()


class TestLifecycle:
    async def test_start_is_lazy_and_idempotent(self, sandbox, client):
        client.invoke.return_value = _response(_exec(stdout="hi"))
        assert sandbox.session_id is None
        await sandbox.execute("echo hi")
        await sandbox.execute("echo hi")
        client.start.assert_called_once_with(
            identifier="aws.codeinterpreter.v1", name="test", session_timeout_seconds=900
        )
        assert sandbox.session_id == "sess-1"

    async def test_context_manager_starts_and_stops(self, sandbox, client):
        async with sandbox as sb:
            assert sb.session_id == "sess-1"
        client.stop.assert_called_once()
        assert sandbox.session_id is None

    async def test_stop_does_not_terminate_borrowed_session(self, client):
        client.session_id = "existing"
        client.identifier = "custom-ci"
        sb = AgentCoreCodeInterpreterSandbox(client=client)
        await sb.start()
        await sb.stop()
        client.start.assert_not_called()
        client.stop.assert_not_called()

    async def test_stop_without_session_is_noop(self, sandbox, client):
        await sandbox.stop()
        client.stop.assert_not_called()


class TestExecute:
    async def test_execute_maps_to_execute_command(self, sandbox, client):
        client.invoke.return_value = _response(_exec(stdout="out", stderr="err", exit_code=3))
        result = await sandbox.execute("ls -la")
        client.invoke.assert_called_once_with("executeCommand", {"command": "ls -la"})
        assert result == ExecutionResult(exit_code=3, stdout="out", stderr="err")

    async def test_streaming_yields_chunks_then_result(self, sandbox, client):
        client.invoke.return_value = _response(_exec(stdout="a"), _exec(stdout="b", stderr="e", exit_code=0))
        chunks = [c async for c in sandbox.execute_streaming("cmd")]
        assert chunks[:3] == [
            StreamChunk("a", "stdout"),
            StreamChunk("b", "stdout"),
            StreamChunk("e", "stderr"),
        ]
        assert chunks[-1] == ExecutionResult(exit_code=0, stdout="ab", stderr="e")

    async def test_cwd_and_env_are_applied(self, sandbox, client):
        client.invoke.return_value = _response(_exec())
        await sandbox.execute("pwd", cwd="/tmp/my dir", env={"FOO": "a b"})
        command = client.invoke.call_args.args[1]["command"]
        assert command == "cd '/tmp/my dir' && export FOO='a b' && pwd"

    async def test_working_dir_default(self, client):
        sb = AgentCoreCodeInterpreterSandbox(client=client, working_dir="/work")
        client.invoke.return_value = _response(_exec())
        await sb.execute("pwd")
        assert client.invoke.call_args.args[1]["command"] == "cd /work && pwd"

    async def test_invalid_env_key_rejected(self, sandbox, client):
        with pytest.raises(ValueError, match="Invalid environment variable"):
            await sandbox.execute("pwd", env={"BAD KEY": "x"})
        client.invoke.assert_not_called()

    async def test_text_content_fallback_without_structured_content(self, sandbox, client):
        client.invoke.return_value = _response({"content": [{"type": "text", "text": "boom"}], "isError": True})
        result = await sandbox.execute("bad")
        assert result == ExecutionResult(exit_code=1, stdout="", stderr="boom")

    async def test_text_content_success_fallback(self, sandbox, client):
        client.invoke.return_value = _response({"content": [{"type": "text", "text": "ok"}]})
        result = await sandbox.execute("good")
        assert result == ExecutionResult(exit_code=0, stdout="ok", stderr="")

    async def test_stream_exception_raises(self, sandbox, client):
        client.invoke.return_value = _response(exception={"throttlingException": {"message": "slow down"}})
        with pytest.raises(CodeInterpreterInvocationError, match="throttlingException: slow down") as exc_info:
            await sandbox.execute("ls")
        assert exc_info.value.error_type == "throttlingException"

    async def test_timeout_raises_sandbox_timeout(self, sandbox, client):
        def slow(*args):
            time.sleep(0.5)
            return _response(_exec(stdout="late"))

        client.invoke.side_effect = slow
        with pytest.raises(SandboxTimeoutError):
            await sandbox.execute("sleep 10", timeout=0.05)

    async def test_timeout_preserves_partial_output(self, sandbox, client):
        def events():
            yield {"result": _exec(stdout="partial")}
            time.sleep(0.5)
            yield {"result": _exec(stdout="late")}

        client.invoke.return_value = {"stream": events()}
        with pytest.raises(SandboxTimeoutError) as exc_info:
            await sandbox.execute("cmd", timeout=0.2)
        assert exc_info.value.stdout == "partial"

    async def test_expired_deadline(self, sandbox, client):
        client.invoke.return_value = _response(_exec())
        fake_time = MagicMock()
        fake_time.monotonic.side_effect = [0.0, 100.0]
        with patch("bedrock_agentcore.tools.integrations.strands.code_interpreter_sandbox.time", fake_time):
            with pytest.raises(SandboxTimeoutError):
                await sandbox.execute("cmd", timeout=1)
        client.invoke.assert_not_called()


class TestExecuteCode:
    @pytest.mark.parametrize(
        "language,expected",
        [("python", "python"), ("python3", "python"), ("node", "javascript"), ("TypeScript", "typescript")],
    )
    async def test_native_languages(self, sandbox, client, language, expected):
        client.invoke.return_value = _response(_exec(stdout="1"))
        result = await sandbox.execute_code("print(1)", language)
        client.invoke.assert_called_once_with(
            "executeCode", {"code": "print(1)", "language": expected, "clearContext": False}
        )
        assert result.stdout == "1"

    async def test_clear_context(self, sandbox, client):
        client.invoke.return_value = _response(_exec())
        await sandbox.execute_code("x = 1", "python", clear_context=True)
        assert client.invoke.call_args.args[1]["clearContext"] is True

    async def test_images_become_output_files(self, sandbox, client):
        png = b"\x89PNG..."
        client.invoke.return_value = _response(
            _exec(content=[{"type": "image", "data": png, "mimeType": "image/png"}, {"type": "text", "text": "x"}])
        )
        result = await sandbox.execute_code("plot()", "python")
        assert len(result.output_files) == 1
        assert result.output_files[0].content == png
        assert result.output_files[0].mime_type == "image/png"
        assert result.output_files[0].name == "output_0.png"

    @pytest.mark.parametrize(
        "kwargs",
        [{"cwd": "/tmp"}, {"env": {"A": "1"}}],
    )
    async def test_cwd_or_env_falls_back_to_shell(self, sandbox, client, kwargs):
        client.invoke.return_value = _response(_exec(stdout="1"))
        await sandbox.execute_code("print(1)", "python3", **kwargs)
        method, args = client.invoke.call_args.args
        assert method == "executeCommand"
        assert "| python3" in args["command"]
        assert base64.b64encode(b"print(1)").decode() in args["command"]

    async def test_unsupported_language_falls_back_to_shell(self, sandbox, client):
        client.invoke.return_value = _response(_exec(stdout="hi"))
        result = await sandbox.execute_code("echo hi", "bash")
        method, args = client.invoke.call_args.args
        assert method == "executeCommand"
        assert "| bash" in args["command"]
        assert result.stdout == "hi"

    async def test_working_dir_falls_back_to_shell(self, client):
        sb = AgentCoreCodeInterpreterSandbox(client=client, working_dir="/work")
        client.invoke.return_value = _response(_exec())
        await sb.execute_code("print(1)", "python3")
        method, args = client.invoke.call_args.args
        assert method == "executeCommand"
        assert args["command"].startswith("cd /work && base64 -d")

    async def test_invalid_language_rejected(self, sandbox, client):
        with pytest.raises(ValueError, match="invalid characters"):
            await sandbox.execute_code("x", "python; rm -rf /")
        client.invoke.assert_not_called()


class TestFiles:
    async def test_read_file_blob(self, sandbox, client):
        client.invoke.return_value = _response(
            {"content": [{"type": "resource", "resource": {"uri": "file:///a.bin", "blob": b"\x00\x01"}}]}
        )
        assert await sandbox.read_file("a.bin") == b"\x00\x01"
        client.invoke.assert_called_once_with("readFiles", {"paths": ["a.bin"]})

    async def test_read_file_text(self, sandbox, client):
        client.invoke.return_value = _response(
            {"content": [{"type": "text", "text": "ignored"}, {"type": "resource", "resource": {"text": "héllo"}}]}
        )
        assert await sandbox.read_text("a.txt") == "héllo"

    async def test_read_file_error(self, sandbox, client):
        client.invoke.return_value = _response({"content": [{"type": "text", "text": "No such file"}], "isError": True})
        with pytest.raises(FileNotFoundError, match="No such file"):
            await sandbox.read_file("missing")

    async def test_read_file_empty_response(self, sandbox, client):
        client.invoke.return_value = _response({"content": []})
        with pytest.raises(FileNotFoundError, match="missing"):
            await sandbox.read_file("missing")

    async def test_write_file(self, sandbox, client):
        client.invoke.return_value = _response({"content": [{"type": "text", "text": "ok"}]})
        await sandbox.write_text("dir/a.txt", "hi")
        client.invoke.assert_called_once_with("writeFiles", {"content": [{"path": "dir/a.txt", "blob": b"hi"}]})

    async def test_write_file_error(self, sandbox, client):
        client.invoke.return_value = _response({"content": [], "isError": True})
        with pytest.raises(OSError, match="Failed to write file: a.txt"):
            await sandbox.write_file("a.txt", b"")

    async def test_remove_file(self, sandbox, client):
        client.invoke.return_value = _response({"content": []})
        await sandbox.remove_file("a.txt")
        client.invoke.assert_called_once_with("removeFiles", {"paths": ["a.txt"]})

    async def test_remove_file_error(self, sandbox, client):
        client.invoke.return_value = _response({"content": [{"type": "text", "text": "gone"}], "isError": True})
        with pytest.raises(FileNotFoundError, match="gone"):
            await sandbox.remove_file("a.txt")

    async def test_list_files(self, sandbox, client):
        client.invoke.return_value = _response(
            {
                "content": [
                    {"type": "resource_link", "uri": "file:///data", "name": "data", "description": "Directory"},
                    {
                        "type": "resource_link",
                        "uri": "file:///a.csv",
                        "name": "a.csv",
                        "mimeType": "text/csv",
                        "size": 12,
                    },
                    {"type": "resource_link", "uri": "file:///sub/"},
                    {"type": "resource_link", "uri": "file:///unknown"},
                    {"type": "resource_link", "uri": "file:///"},
                    {"type": "text", "text": "summary"},
                ]
            }
        )
        entries = await sandbox.list_files("/")
        client.invoke.assert_called_once_with("listFiles", {"directoryPath": "/"})
        assert entries == [
            FileInfo(name="data", is_dir=True, size=None),
            FileInfo(name="a.csv", is_dir=False, size=12),
            FileInfo(name="sub", is_dir=True, size=None),
            FileInfo(name="unknown", is_dir=None, size=None),
        ]

    async def test_list_files_not_found(self, sandbox, client):
        client.invoke.return_value = _response(
            {"content": [{"type": "text", "text": "Directory does not exist"}], "isError": True}
        )
        with pytest.raises(SandboxPathNotFoundError):
            await sandbox.list_files("/nope")

    async def test_list_files_other_error(self, sandbox, client):
        client.invoke.return_value = _response({"content": [{"type": "text", "text": "denied"}], "isError": True})
        with pytest.raises(OSError, match="denied") as exc_info:
            await sandbox.list_files("/root")
        assert not isinstance(exc_info.value, SandboxPathNotFoundError)


class TestTools:
    def test_get_tools(self, client):
        sb = AgentCoreCodeInterpreterSandbox(client=client, working_dir="/work")
        tools = {t.tool_name: t for t in sb.get_tools()}
        assert set(tools) == {"sandbox_shell", "sandbox_file_editor"}
        assert "Code Interpreter" in tools["sandbox_shell"].tool_spec["description"]
        assert "/work" in tools["sandbox_shell"].tool_spec["description"]

    def test_agent_accepts_sandbox(self, sandbox):
        from strands import Agent

        agent = Agent(sandbox=sandbox, model=MagicMock())
        assert agent.sandbox is sandbox
        assert {"sandbox_shell", "sandbox_file_editor"} <= set(agent.tool_names)
