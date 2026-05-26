export type ContextWindow = {
  retrievalQuery: string;
  includedChunkIds: string[];
  omittedChunkIds: string[];
  budgetChars: number;
};

export class AgentContextStore {
  private windows: ContextWindow[] = [];

  recordWindow(window: ContextWindow) {
    this.windows.push(window);
    if (this.windows.length > 50) this.windows.shift();
  }

  latestForRetrieval(query: string) {
    return [...this.windows].reverse().find((window) => window.retrievalQuery === query);
  }
}
