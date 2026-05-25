import json
import time
from collections.abc import Iterable


def read_jsonl_records(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def ingest_with_retry(client, path: str, batch_size: int = 100) -> int:
    inserted = 0
    batch: list[dict] = []
    for record in read_jsonl_records(path):
        batch.append(record)
        if len(batch) >= batch_size:
            inserted += _write_batch_with_backoff(client, batch)
            batch.clear()
    if batch:
        inserted += _write_batch_with_backoff(client, batch)
    return inserted


def _write_batch_with_backoff(client, records: list[dict]) -> int:
    delay = 0.25
    for attempt in range(4):
        try:
            client.insert_many(records)
            return len(records)
        except TimeoutError:
            if attempt == 3:
                raise
            time.sleep(delay)
            delay *= 2
    return 0
