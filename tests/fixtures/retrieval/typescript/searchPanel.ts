type SearchResult = {
  id: string;
  title: string;
  snippet: string;
};

export class SearchPanel {
  private abortController?: AbortController;
  private timer?: number;

  constructor(private readonly input: HTMLInputElement) {}

  bind(onResults: (results: SearchResult[]) => void) {
    this.input.addEventListener("input", () => {
      window.clearTimeout(this.timer);
      this.timer = window.setTimeout(() => {
        this.runDebouncedSearch(this.input.value, onResults);
      }, 150);
    });
  }

  private async runDebouncedSearch(
    query: string,
    onResults: (results: SearchResult[]) => void,
  ) {
    this.abortController?.abort();
    this.abortController = new AbortController();
    const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`, {
      signal: this.abortController.signal,
    });
    if (!response.ok) throw new Error("search failed");
    const payload = await response.json() as { results: SearchResult[] };
    onResults(payload.results);
  }
}
