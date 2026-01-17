import os
from pathlib import Path

from backend.db.sqlite_store import SQLiteStore
from scripts.migrate_memory_to_db import migrate_memory_stores_to_db


def test_migrate_dry_run_and_archive(tmp_path):
    ms = tmp_path / 'memory_stores'
    a = ms / 'agentA'
    a.mkdir(parents=True)
    f = a / 'notes.txt'
    f.write_text('note one\n')

    dbpath = tmp_path / 'panel.db'
    store = SQLiteStore(str(dbpath))

    # dry-run should return count but not persist nor move files
    count = migrate_memory_stores_to_db(store, src_dir=str(ms), dry_run=True, archive=str(tmp_path / 'archive'))
    assert count == 1
    assert f.exists()

    # real run with archive should move file
    count2 = migrate_memory_stores_to_db(store, src_dir=str(ms), dry_run=False, archive=str(tmp_path / 'archive'))
    assert count2 == 1
    # original file moved
    assert not f.exists()
    archived = tmp_path / 'archive' / 'agentA' / 'notes.txt'
    assert archived.exists()
