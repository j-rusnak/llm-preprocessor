type RetrievalDiagnostic = {
  language: string;
  query: string;
  expectedPath: string;
  expectedRank: number | null;
  topK: string[];
  graphExpanded: boolean;
};

export function buildContextGraphRows(items: RetrievalDiagnostic[]) {
  return items.map((item) => ({
    label: `${item.language}: ${item.expectedPath}`,
    rank: item.expectedRank ?? "miss",
    nearMiss: item.expectedRank === null || item.expectedRank > 3,
    graphLift: item.graphExpanded ? "graph expansion" : "base retrieval",
    topKPreview: item.topK.slice(0, 3).join(" -> "),
  }));
}
