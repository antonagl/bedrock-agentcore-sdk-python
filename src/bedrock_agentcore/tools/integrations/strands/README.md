# AgentCore Code Interpreter Sandbox for Strands

`AgentCoreCodeInterpreterSandbox` implements the Strands
[`Sandbox`](https://strandsagents.com) interface (`strands.sandbox.Sandbox`, `strands-agents>=1.56.0`)
on top of an Amazon Bedrock AgentCore Code Interpreter session. Commands, code, and file
operations run in the managed Code Interpreter instead of on the agent's host.

## Installation

```bash
pip install 'bedrock-agentcore[strands-agents]'
```

## Usage

```python
from strands import Agent
from bedrock_agentcore.tools.integrations.strands import AgentCoreCodeInterpreterSandbox

async with AgentCoreCodeInterpreterSandbox(region="us-west-2") as sandbox:
    agent = Agent(sandbox=sandbox)
    await agent.invoke_async("Create data.csv with 10 random numbers and print their mean.")
```

The sandbox automatically adds the `sandbox_shell` and `sandbox_file_editor` tools to the agent.
Other sandbox-aware tools, such as `strands.vended_tools.shell`, use it through `agent.sandbox`.

You can also call the sandbox directly:

```python
sandbox = AgentCoreCodeInterpreterSandbox(region="us-west-2")
await sandbox.write_text("data/input.txt", "hello")
result = await sandbox.execute("wc -c data/input.txt")
result = await sandbox.execute_code("print(open('data/input.txt').read())", "python3")
await sandbox.stop()
```

The session starts lazily on first use. Call `stop()`, or use `async with`, to release it.
Otherwise the session expires after `session_timeout_seconds` (900 by default).

### Options

| Argument | Description |
|---|---|
| `region` | AWS region. Defaults to the boto3 session's region. |
| `identifier` | Code Interpreter ID. Defaults to the system interpreter `aws.codeinterpreter.v1`. Pass a custom interpreter ID for VPC or other custom configurations. |
| `session` | Optional `boto3.Session`. |
| `session_name` / `session_timeout_seconds` | Settings for the Code Interpreter session. |
| `working_dir` | Default working directory for `execute`/`execute_code`. |
| `client` | Existing `CodeInterpreter` client. If it already has a session, the sandbox reuses it and does not stop it. |

## Method mapping

| `Sandbox` method | Code Interpreter action |
|---|---|
| `execute_streaming` | `executeCommand` |
| `execute_code_streaming` | `executeCode` |
| `read_file` | `readFiles` |
| `write_file` | `writeFiles` |
| `list_files` | `listFiles` |
| `remove_file` | `removeFiles` |

## Behavior notes

- **Native code execution.** `execute_code` runs Python (`python`, `python3`), JavaScript (`node`,
  `javascript`), and TypeScript (`typescript`, `ts`) natively through `executeCode`. That context
  persists across calls. Pass `clear_context=True` to reset it. Generated images are returned in
  `ExecutionResult.output_files`.
- **Shell fallback.** `executeCode` cannot set a working directory or environment variables.
  When `cwd`, `env`, or `working_dir` is set, or the language is not supported natively
  (for example `bash`), the code is piped to the interpreter through `executeCommand`.
- **Timeouts.** `timeout` is enforced client-side and raises `SandboxTimeoutError`. The remote
  process is not killed; it keeps running until the session ends.
