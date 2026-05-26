export type ContextWindow = {
  retrievalQuery: string;
  keptChunkIds: string[];
  skippedChunkIds: string[];
  budgetChars: number;
  persistedAt?: string;
};

export type ContextSnapshot = {
  version: 1;
  savedAt: string;
  items: ContextWindow[];
};

export interface ContextPersistence {
  load(key: string): string | null;
  save(key: string, value: string): void;
  remove(key: string): void;
}

export interface SnapshotCodec {
  encode(snapshot: ContextSnapshot): string;
  decode(raw: string): Partial<ContextSnapshot> | null;
}

export class LocalStorageContextPersistence implements ContextPersistence {
  constructor(private readonly storage: Pick<Storage, "getItem" | "setItem" | "removeItem">) {}
  load(key: string) { return this.storage.getItem(key); }
  save(key: string, value: string) { this.storage.setItem(key, value); }
  remove(key: string) { this.storage.removeItem(key); }
}

export class AgentContextStore {
  private static readonly cacheNamespace = "agent-context-window-cache";
  private windows: ContextWindow[] = [];

  constructor(
    private readonly persistence: ContextPersistence,
    private readonly codec: SnapshotCodec,
    private readonly key = AgentContextStore.cacheNamespace,
  ) {}

  captureWindow(window: ContextWindow): ContextWindow {
    const normalized = {
      ...window,
      budgetChars: Math.max(0, Math.trunc(window.budgetChars)),
      persistedAt: new Date().toISOString(),
    };
    this.windows.push(normalized);
    if (this.windows.length > 50) this.windows.shift();
    this.saveSnapshot();
    return normalized;
  }

  latestForRetrieval(query: string): ContextWindow | undefined {
    return [...this.windows].reverse().find((window) => window.retrievalQuery === query);
  }

  loadSnapshot(): ContextSnapshot {
    try {
      const raw = this.persistence.load(this.key);
      const parsed = raw ? this.codec.decode(raw) : null;
      this.windows = parsed?.version === 1 && Array.isArray(parsed.items) ? parsed.items.slice(-50) : [];
    } catch {
      this.persistence.remove(this.key);
      this.windows = [];
    }
    return this.snapshot();
  }

  saveSnapshot(): ContextSnapshot {
    const snapshot = this.snapshot();
    this.persistence.save(this.key, this.codec.encode(snapshot));
    return snapshot;
  }

  snapshot(): ContextSnapshot {
    return { version: 1, savedAt: new Date().toISOString(), items: this.windows.map((window) => ({ ...window })) };
  }
}
