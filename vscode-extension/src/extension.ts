// LLM Preprocessor — minimal VS Code extension.
//
// Registers the bundled `preprocessor_app --mcp` binary as a stdio MCP
// server with VS Code's Language Model / MCP host (VS Code 1.99+).
//
// This file is intentionally small: it just teaches the editor how to
// spawn the C++ binary. All RAG logic lives in the binary itself.

import * as vscode from 'vscode';

const SERVER_ID = 'llm-preprocessor';

interface McpServerDefinition {
    label: string;
    command: string;
    args?: string[];
    cwd?: string;
    env?: Record<string, string>;
}

// `vscode.lm.registerMcpServerDefinitionProvider` lands in the proposed
// API for newer VS Code builds. We probe at runtime so the extension
// degrades gracefully on older hosts.
type LmExtras = {
    registerMcpServerDefinitionProvider?: (
        id: string,
        provider: { provideMcpServerDefinitions: () => Thenable<McpServerDefinition[]> | McpServerDefinition[] }
    ) => vscode.Disposable;
};

function buildDefinition(): McpServerDefinition {
    const cfg = vscode.workspace.getConfiguration('llmPreprocessor');
    const binary = cfg.get<string>('binaryPath', 'preprocessor_app');
    const configPath = cfg.get<string>('configPath', '');
    const cwdOverride = cfg.get<string>('cwd', '');
    const label = cfg.get<string>('serverLabel', SERVER_ID);

    const args = ['--mcp'];
    if (configPath) {
        args.push(configPath);
    }

    const folders = vscode.workspace.workspaceFolders;
    const cwd = cwdOverride || (folders && folders.length > 0 ? folders[0].uri.fsPath : process.cwd());

    return { label, command: binary, args, cwd };
}

export function activate(context: vscode.ExtensionContext) {
    const lm = vscode.lm as unknown as LmExtras;

    const register = () => {
        if (typeof lm.registerMcpServerDefinitionProvider !== 'function') {
            vscode.window.showWarningMessage(
                'LLM Preprocessor: this VS Code build does not expose the MCP server API. ' +
                'Update VS Code (1.99+) and ensure the chat.mcp.enabled setting is on.'
            );
            return undefined;
        }
        return lm.registerMcpServerDefinitionProvider(SERVER_ID, {
            provideMcpServerDefinitions: () => [buildDefinition()]
        });
    };

    let disposable = register();
    if (disposable) {
        context.subscriptions.push(disposable);
    }

    context.subscriptions.push(
        vscode.commands.registerCommand('llmPreprocessor.restart', () => {
            if (disposable) {
                disposable.dispose();
            }
            disposable = register();
            if (disposable) {
                context.subscriptions.push(disposable);
                vscode.window.showInformationMessage('LLM Preprocessor MCP server re-registered.');
            }
        })
    );
}

export function deactivate(): void {
    // Disposables are released via context.subscriptions.
}
