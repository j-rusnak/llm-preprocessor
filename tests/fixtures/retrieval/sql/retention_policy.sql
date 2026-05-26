CREATE TABLE dashboard_history_retention (
    id INTEGER PRIMARY KEY,
    snapshot_kind TEXT NOT NULL,
    baseline_name TEXT NOT NULL,
    captured_at_ms INTEGER NOT NULL,
    retained_until_ms INTEGER NOT NULL
);

CREATE INDEX idx_dashboard_history_retention_baseline
    ON dashboard_history_retention (baseline_name, captured_at_ms DESC);
