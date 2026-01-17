import json
import os
from pathlib import Path

import pytest

from backend.db.sqlite_store import SQLiteStore
from scripts.migrate_memory_to_db import migrate_memory_stores_to_db


def make_jsonl(path: Path, entries):
    with path.open('w', encoding='utf-8') as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def test_migrate_jsonl_and_text(tmp_path):
    # Setup a fake memory_stores layout
    ms = tmp_path / 'memory_stores'
    katz = ms / 'katz'
    dogz = ms / 'dogz'
    katz.mkdir(parents=True)
    dogz.mkdir(parents=True)

    # JSONL file for katz
    katz_file = katz / 'katz_mem.jsonl'
    entries = [
        {"agent_id": "katz", "content": "Katz memory one", "source": "legacy"},
        {"agent_id": "katz", "content": "Katz memory two", "source": "legacy"},
    ]
    make_jsonl(katz_file, entries)

    # Plain text file for dogz (single-blob)
    dogz_file = dogz / 'note.txt'
    dogz_file.write_text('DogZ single note\n')

    # Create a temporary SQLite DB
    dbpath = tmp_path / 'panel.db'
    store = SQLiteStore(str(dbpath))

    migrated = migrate_memory_stores_to_db(store, src_dir=str(ms))
    assert migrated == 3

    # Ensure contents present
    mems_katz = [m['content'] for m in store.get_memories('katz')]
    assert 'Katz memory one' in mems_katz
    assert 'Katz memory two' in mems_katz

    mems_dogz = [m['content'] for m in store.get_memories('dogz')]
    assert 'DogZ single note' in mems_dogz

    # Re-run migration -> should be idempotent
    migrated2 = migrate_memory_stores_to_db(store, src_dir=str(ms))
    assert migrated2 == 0
