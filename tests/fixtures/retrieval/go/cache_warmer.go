package retrieval

import "context"

type EmbeddingCache interface {
	Prefetch(ctx context.Context, repoPath string, chunkIDs []string) error
}

func WarmRepositoryVectors(ctx context.Context, cache EmbeddingCache, repoPath string, chunkIDs []string) error {
	if len(chunkIDs) == 0 {
		return nil
	}
	return cache.Prefetch(ctx, repoPath, chunkIDs)
}
