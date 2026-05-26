package dev.preprocessor.routing;

public final class ModelRoutingPolicy {
    public String chooseTier(String promptBucket, int requestChars) {
        if ("CodeEdit".equals(promptBucket) && requestChars < 2000) {
            return "cheap";
        }
        if ("CodeGenerate".equals(promptBucket) && requestChars > 8000) {
            return "frontier";
        }
        return "fallback";
    }
}
