"""Strands ``Sandbox`` backed by an AgentCore Code Interpreter session.

Maps the :class:`strands.sandbox.Sandbox` interface onto Code Interpreter tool actions:

=============================  ==========================
``Sandbox`` method             Code Interpreter action
=============================  ==========================
``execute_streaming``          ``executeCommand``
``execute_code_streaming``     ``executeCode``
``read_file``                  ``readFiles``
``write_file``                 ``writeFiles``
``list_files``                 ``listFiles``
``remove_file``                ``removeFiles``
=============================  ==========================

Example:
    >>> from strands import Agent
    >>> from bedrock_agentcore.tools.integrations.strands import AgentCoreCodeInterpreterSandbox
    >>>
    >>> sandbox = AgentCoreCodeInterpreterSandbox(region="us-west-2")
    >>> agent = Agent(sandbox=sandbox)
    >>> agent("Write a CSV with 10 random numbers and compute their mean.")
    >>> await sandbox.stop()
"""

import asyncio
import logging
import mimetypes
import posixpath
import shlex
import time
from collections.abc import AsyncGenerator, Iterator
from types import TracebackType
from typing import Any, Callable, Optional, TypeVar

import boto3

try:
    from strands.sandbox import ExecutionResult, FileInfo, OutputFile, PosixShellSandbox, StreamChunk
    from strands.sandbox.errors import SandboxPathNotFoundError, SandboxTimeoutError
    from strands.sandbox.posix_shell import build_shell_env_prefix
    from strands.types.tools import AgentTool
    from strands.vended_tools.file_editor import make_file_editor
    from strands.vended_tools.file_editor.file_editor import DEFAULT_FILE_EDITOR_DESCRIPTION
    from strands.vended_tools.shell import make_shell
    from strands.vended_tools.shell.types import SANDBOX_SHELL_DESCRIPTION
except ImportError as e:  # pragma: no cover - exercised only without the extra installed
    raise ImportError(
        "AgentCoreCodeInterpreterSandbox requires strands-agents>=1.56.0. "
        "Install it with: pip install 'bedrock-agentcore[strands-agents]'"
    ) from e

from ...code_interpreter_client import DEFAULT_IDENTIFIER, DEFAULT_TIMEOUT, CodeInterpreter

logger = logging.getLogger(__name__)

T = TypeVar("T")

#: Maps interpreter names accepted by ``execute_code`` to Code Interpreter ``executeCode`` languages.
LANGUAGE_ALIASES: dict[str, str] = {
    "python": "python",
    "python3": "python",
    "py": "python",
    "javascript": "javascript",
    "js": "javascript",
    "node": "javascript",
    "nodejs": "javascript",
    "typescript": "typescript",
    "ts": "typescript",
}

_STREAM_EXCEPTION_KEYS = (
    "accessDeniedException",
    "conflictException",
    "internalServerException",
    "resourceNotFoundException",
    "serviceQuotaExceededException",
    "throttlingException",
    "validationException",
)

_DIRECTORY_MIME_TYPES = {"inode/directory", "application/x-directory", "directory"}
_NOT_FOUND_MARKERS = ("not found", "no such file", "does not exist", "not exist", "enoent")
_SENTINEL = object()


class CodeInterpreterInvocationError(RuntimeError):
    """Raised when the Code Interpreter stream returns a service exception event."""

    def __init__(self, error_type: str, message: str) -> None:
        """Initialize the error.

        Args:
            error_type: The stream exception member name (e.g. ``"throttlingException"``).
            message: The service-provided error message.
        """
        super().__init__(f"{error_type}: {message}")
        self.error_type = error_type


class AgentCoreCodeInterpreterSandbox(PosixShellSandbox):
    """A Strands sandbox that runs commands, code, and file I/O in an AgentCore Code Interpreter session.

    The Code Interpreter session is started lazily on first use (or explicitly via :meth:`start`)
    and is shared by every tool of the agent, so files written by one tool are visible to others.
    Call :meth:`stop` (or use ``async with``) to release the session; otherwise it expires after
    ``session_timeout_seconds``.

    Behavior notes:

    - ``execute_code`` with a Python/JavaScript/TypeScript interpreter name (see
      :data:`LANGUAGE_ALIASES`) runs natively via ``executeCode``. The Code Interpreter keeps a
      persistent execution context across calls, and generated images are returned as
      :attr:`ExecutionResult.output_files`. Pass ``clear_context=True`` to reset it.
    - ``executeCode`` does not support a working directory or environment variables, so when
      ``cwd`` or ``env`` is given — or the language is not natively supported (e.g. ``"bash"``) —
      the code is instead piped to the interpreter through ``executeCommand``.
    - ``timeout`` is enforced client-side; a timed-out call raises
      :class:`~strands.sandbox.errors.SandboxTimeoutError`.
    - Output is delivered as the service streams it back; for most actions that is a single
      chunk once execution completes.
    """

    def __init__(
        self,
        region: Optional[str] = None,
        *,
        identifier: str = DEFAULT_IDENTIFIER,
        session: Optional[boto3.Session] = None,
        session_name: Optional[str] = None,
        session_timeout_seconds: int = DEFAULT_TIMEOUT,
        working_dir: Optional[str] = None,
        client: Optional[CodeInterpreter] = None,
    ) -> None:
        """Initialize the sandbox.

        Args:
            region: AWS region of the Code Interpreter. Defaults to the boto3 session's region.
                Ignored when ``client`` is given.
            identifier: Code Interpreter identifier, either the system interpreter
                (``aws.codeinterpreter.v1``) or a custom interpreter ID.
            session: Optional boto3 session used to create the client. Ignored when ``client`` is given.
            session_name: Optional name for the Code Interpreter session.
            session_timeout_seconds: Session timeout in seconds (default 900).
            working_dir: Default working directory for ``execute``/``execute_code`` calls.
                ``None`` uses the Code Interpreter's default working directory.
            client: An existing :class:`CodeInterpreter` client. If it already has an active
                session, the sandbox reuses it and :meth:`stop` will not terminate it.

        Raises:
            ValueError: If no region is given and none can be resolved from the boto3 session.
        """
        if client is None:
            resolved_region = region or (session or boto3.Session()).region_name
            if not resolved_region:
                raise ValueError("region must be provided or configured for the boto3 session")
            client = CodeInterpreter(resolved_region, session=session, integration_source="strands")

        self.client = client
        self.identifier = identifier
        self.session_name = session_name
        self.session_timeout_seconds = session_timeout_seconds
        self.working_dir = working_dir
        self._owns_session = not (client.session_id and client.identifier)
        self._start_lock = asyncio.Lock()

    # ---- Session lifecycle ----

    @property
    def session_id(self) -> Optional[str]:
        """The active Code Interpreter session ID, or ``None`` if no session is active."""
        return self.client.session_id

    async def start(self) -> str:
        """Start the Code Interpreter session if one is not already active.

        Returns:
            The active session ID.
        """
        async with self._start_lock:
            if not (self.client.session_id and self.client.identifier):
                await asyncio.to_thread(
                    self.client.start,
                    identifier=self.identifier,
                    name=self.session_name,
                    session_timeout_seconds=self.session_timeout_seconds,
                )
                self._owns_session = True
            return str(self.client.session_id)

    async def stop(self) -> None:
        """Stop the Code Interpreter session if this sandbox started it."""
        async with self._start_lock:
            if self._owns_session and self.client.session_id:
                await asyncio.to_thread(self.client.stop)

    async def __aenter__(self) -> "AgentCoreCodeInterpreterSandbox":
        """Start the session on entering an ``async with`` block."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        """Stop the session on leaving an ``async with`` block."""
        await self.stop()

    # ---- Invocation helpers ----

    async def _call(self, func: Callable[..., T], *args: Any, deadline: Optional[float] = None) -> T:
        """Run a blocking call in a worker thread, bounded by an optional monotonic ``deadline``."""
        if deadline is None:
            return await asyncio.to_thread(func, *args)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise asyncio.TimeoutError
        return await asyncio.wait_for(asyncio.to_thread(func, *args), timeout=remaining)

    async def _invoke_stream(
        self, method: str, arguments: dict[str, Any], *, deadline: Optional[float] = None
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Invoke a Code Interpreter action and yield each ``result`` payload from its event stream.

        Raises:
            CodeInterpreterInvocationError: If the stream carries a service exception event.
            asyncio.TimeoutError: If ``deadline`` passes.
        """
        await self.start()
        response = await self._call(self.client.invoke, method, arguments, deadline=deadline)
        events: Iterator[dict[str, Any]] = iter(response.get("stream") or [])
        while True:
            event = await self._call(next, events, _SENTINEL, deadline=deadline)
            if event is _SENTINEL:
                return
            for key in _STREAM_EXCEPTION_KEYS:
                if key in event:
                    raise CodeInterpreterInvocationError(key, (event[key] or {}).get("message", ""))
            if "result" in event:
                yield event["result"]

    async def _invoke(self, method: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """Invoke a Code Interpreter action and collect all ``result`` payloads."""
        return [result async for result in self._invoke_stream(method, arguments)]

    async def _stream_execution(
        self, method: str, arguments: dict[str, Any], timeout: Optional[float]
    ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
        """Stream an ``executeCommand``/``executeCode`` invocation as sandbox chunks and a final result."""
        deadline = time.monotonic() + timeout if timeout is not None else None
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        output_files: list[OutputFile] = []
        exit_code: Optional[int] = None
        is_error = False
        try:
            async for result in self._invoke_stream(method, arguments, deadline=deadline):
                stdout, stderr, result_exit_code, files = _parse_execution_result(result)
                is_error = is_error or bool(result.get("isError"))
                if result_exit_code is not None:
                    exit_code = result_exit_code
                output_files.extend(files)
                if stdout:
                    stdout_parts.append(stdout)
                    yield StreamChunk(data=stdout, stream_type="stdout")
                if stderr:
                    stderr_parts.append(stderr)
                    yield StreamChunk(data=stderr, stream_type="stderr")
        except asyncio.TimeoutError:
            raise SandboxTimeoutError(timeout, "".join(stdout_parts), "".join(stderr_parts)) from None

        if exit_code is None:
            exit_code = 1 if is_error else 0
        yield ExecutionResult(
            exit_code=exit_code,
            stdout="".join(stdout_parts),
            stderr="".join(stderr_parts),
            output_files=output_files,
        )

    # ---- Execution ----

    async def execute_streaming(
        self,
        command: str,
        *,
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
        """Execute a shell command in the Code Interpreter session via ``executeCommand``.

        Args:
            command: The shell command to execute.
            timeout: Maximum execution time in seconds. ``None`` means no timeout.
            cwd: Working directory for this command, overriding ``working_dir``.
            env: Environment variables to export for this command.
            **kwargs: Additional keyword arguments for forward compatibility.

        Yields:
            :class:`StreamChunk` objects for output, then a final :class:`ExecutionResult`.

        Raises:
            ValueError: If an environment variable name is invalid.
            SandboxTimeoutError: If execution exceeds ``timeout`` seconds.
            CodeInterpreterInvocationError: If the service returns an error event.
        """
        prefix = build_shell_env_prefix(env)
        effective_cwd = cwd if cwd is not None else self.working_dir
        if effective_cwd is not None:
            prefix = f"cd {shlex.quote(effective_cwd)} && {prefix}"
        async for chunk in self._stream_execution("executeCommand", {"command": prefix + command}, timeout):
            yield chunk

    async def execute_code_streaming(
        self,
        code: str,
        language: str,
        *,
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
        """Execute source code in the Code Interpreter session.

        Natively supported languages (see :data:`LANGUAGE_ALIASES`) run via ``executeCode`` in the
        interpreter's persistent context. Other languages, or calls that set ``cwd``/``env`` (or a
        default ``working_dir``), pipe the code to ``language`` through ``executeCommand``.

        Args:
            code: The source code to execute.
            language: The interpreter to use (e.g. ``"python3"``, ``"node"``, ``"bash"``).
            timeout: Maximum execution time in seconds. ``None`` means no timeout.
            cwd: Working directory for execution.
            env: Environment variables to set for this execution.
            **kwargs: Additional keyword arguments. ``clear_context=True`` resets the native
                execution context before running.

        Yields:
            :class:`StreamChunk` objects for output, then a final :class:`ExecutionResult`.

        Raises:
            ValueError: If ``language`` contains invalid characters.
            SandboxTimeoutError: If execution exceeds ``timeout`` seconds.
            CodeInterpreterInvocationError: If the service returns an error event.
        """
        native_language = LANGUAGE_ALIASES.get(language.lower())
        if native_language is None or cwd is not None or env or self.working_dir is not None:
            async for chunk in super().execute_code_streaming(code, language, timeout=timeout, cwd=cwd, env=env):
                yield chunk
            return

        arguments = {
            "code": code,
            "language": native_language,
            "clearContext": bool(kwargs.get("clear_context", False)),
        }
        async for chunk in self._stream_execution("executeCode", arguments, timeout):
            yield chunk

    # ---- File operations ----

    async def read_file(self, path: str, **kwargs: Any) -> bytes:
        """Read a file from the session via ``readFiles``.

        Args:
            path: Path to the file to read.
            **kwargs: Additional keyword arguments for forward compatibility.

        Returns:
            The file contents as raw bytes.

        Raises:
            FileNotFoundError: If the file does not exist or cannot be read.
        """
        for result in await self._invoke("readFiles", {"paths": [path]}):
            if result.get("isError"):
                raise FileNotFoundError(_result_text(result) or f"Failed to read file: {path}")
            for item in result.get("content") or []:
                resource = item.get("resource") if item.get("type") == "resource" else None
                if not resource:
                    continue
                if resource.get("blob") is not None:
                    return bytes(resource["blob"])
                if resource.get("text") is not None:
                    return str(resource["text"]).encode("utf-8")
        raise FileNotFoundError(f"Failed to read file: {path}")

    async def write_file(self, path: str, content: bytes, **kwargs: Any) -> None:
        """Write raw bytes to a file in the session via ``writeFiles``.

        Args:
            path: Path to the file to write.
            content: The content to write.
            **kwargs: Additional keyword arguments for forward compatibility.

        Raises:
            OSError: If the file cannot be written.
        """
        for result in await self._invoke("writeFiles", {"content": [{"path": path, "blob": content}]}):
            if result.get("isError"):
                raise OSError(_result_text(result) or f"Failed to write file: {path}")

    async def remove_file(self, path: str, **kwargs: Any) -> None:
        """Remove a file from the session via ``removeFiles``.

        Args:
            path: Path to the file to remove.
            **kwargs: Additional keyword arguments for forward compatibility.

        Raises:
            FileNotFoundError: If the file does not exist or cannot be removed.
        """
        for result in await self._invoke("removeFiles", {"paths": [path]}):
            if result.get("isError"):
                raise FileNotFoundError(_result_text(result) or f"Failed to remove file: {path}")

    async def list_files(self, path: str, **kwargs: Any) -> list[FileInfo]:
        """List directory contents in the session via ``listFiles``.

        Args:
            path: Path to the directory to list.
            **kwargs: Additional keyword arguments for forward compatibility.

        Returns:
            :class:`FileInfo` entries for the directory contents.

        Raises:
            SandboxPathNotFoundError: If the directory does not exist.
            OSError: If the listing fails for another reason.
        """
        entries: list[FileInfo] = []
        for result in await self._invoke("listFiles", {"directoryPath": path}):
            if result.get("isError"):
                message = _result_text(result)
                if any(marker in message.lower() for marker in _NOT_FOUND_MARKERS):
                    raise SandboxPathNotFoundError(path)
                raise OSError(message or f"Failed to list directory: {path}")
            for item in result.get("content") or []:
                entry = _to_file_info(item)
                if entry is not None:
                    entries.append(entry)
        return entries

    # ---- Tool vending ----

    def get_tools(self) -> list[AgentTool]:
        """Default sandbox-compatible tools auto-registered with this sandbox.

        Returns:
            Shell and file editor tools bound to this sandbox.
        """
        where = " Runs in an Amazon Bedrock AgentCore Code Interpreter session."
        cwd = f" Working directory: {self.working_dir}." if self.working_dir else ""
        return [
            make_file_editor(
                sandbox=self,
                name="sandbox_file_editor",
                description=f"{DEFAULT_FILE_EDITOR_DESCRIPTION}{where}",
            ),
            make_shell(
                sandbox=self,
                name="sandbox_shell",
                description=f"{SANDBOX_SHELL_DESCRIPTION}{where}{cwd}",
            ),
        ]


# ---- Response parsing ----


def _result_text(result: dict[str, Any]) -> str:
    """Join the ``text`` content blocks of a Code Interpreter result."""
    return "".join(
        str(item.get("text", "")) for item in result.get("content") or [] if item.get("type") == "text"
    ).strip()


def _parse_execution_result(
    result: dict[str, Any],
) -> tuple[str, str, Optional[int], list[OutputFile]]:
    """Extract ``(stdout, stderr, exit_code, output_files)`` from an execution result payload."""
    structured = result.get("structuredContent") or {}
    stdout = structured.get("stdout")
    stderr = structured.get("stderr")
    if stdout is None and stderr is None:
        text = _result_text(result)
        stdout, stderr = ("", text) if result.get("isError") else (text, "")

    output_files: list[OutputFile] = []
    for index, item in enumerate(result.get("content") or []):
        if item.get("type") != "image" or item.get("data") is None:
            continue
        mime_type = item.get("mimeType") or "application/octet-stream"
        name = item.get("name") or f"output_{index}{mimetypes.guess_extension(mime_type) or ''}"
        output_files.append(OutputFile(name=name, content=bytes(item["data"]), mime_type=mime_type))

    return stdout or "", stderr or "", structured.get("exitCode"), output_files


def _to_file_info(item: dict[str, Any]) -> Optional[FileInfo]:
    """Convert a ``listFiles`` content block to a :class:`FileInfo`, or ``None`` if it is not an entry."""
    if item.get("type") not in ("resource_link", "resource"):
        return None
    uri = str(item.get("uri") or "")
    name = item.get("name") or posixpath.basename(uri.removeprefix("file://").rstrip("/"))
    if not name or name in (".", ".."):
        return None

    description = str(item.get("description") or "").strip().lower()
    mime_type = str(item.get("mimeType") or "").lower()
    is_dir: Optional[bool]
    if description in ("directory", "dir", "folder") or mime_type in _DIRECTORY_MIME_TYPES or uri.endswith("/"):
        is_dir = True
    elif description == "file" or mime_type or item.get("size") is not None:
        is_dir = False
    else:
        is_dir = None
    return FileInfo(name=name.rstrip("/"), is_dir=is_dir, size=item.get("size") if is_dir is False else None)
