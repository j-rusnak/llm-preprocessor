# LLM Preprocessor — VS Code extension

Thin shim that registers the bundled `preprocessor_app --mcp` binary as a
local MCP server with VS Code's Language Model host (VS Code 1.99+).

## Build

```powershell
cd vscode-extension
npm install
npm run compile
```

Then load the folder via `Developer: Install Extension from Location...` or
press `F5` from a VS Code launched against this folder.

## Settings

- `llmPreprocessor.binaryPath` — path to `preprocessor_app` (default: PATH lookup).
- `llmPreprocessor.configPath` — optional path to `config.json`.
- `llmPreprocessor.cwd` — working directory (default: first workspace folder).
- `llmPreprocessor.serverLabel` — display name for the MCP server.

## Manual MCP registration

If you would rather configure the MCP server directly (e.g. for Continue,
Claude Desktop, or `~/.vscode/mcp.json`), point your client at:

```json
{
  "mcpServers": {
    "llm-preprocessor": {
      "command": "preprocessor_app",
      "args": ["--mcp", "config.json"]
    }
  }
}
```
