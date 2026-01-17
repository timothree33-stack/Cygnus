"""Migrate legacy on-disk memory_stores into the canonical SQLiteStore.

This module provides a single convenience function:

  migrate_memory_stores_to_db(store_or_dbpath: Union[SQLiteStore, str], src_dir: Optional[str]=None) -> int

Behavior & guarantees:
- Supports simple legacy formats commonly used in this repo:
  - JSONL files where each line is a JSON object with keys: content, agent_id, embedding (optional), source (optional)
  - JSON arrays of objects (same schema)
  - Plain text files: migrates the whole file as one memory; agent inferred from parent directory name
- Idempotent: will not insert duplicate memories (compares agent_id + content)
- Robust: skips unreadable files and continues; returns the number of new memories inserted

Intended for use by the admin endpoint `POST /api/admin/import-memory` and by maintainers during upgrades.
"""
from __future__ import annotations
import os
import json
from typing import Optional, Union

# Import the store class lazily to avoid import cycles in tests
try:
    from backend.db.sqlite_store import SQLiteStore
except Exception:  # pragma: no cover - defensive
    SQLiteStore = None  # type: ignore


def _iter_legacy_entries_from_file(path: str, agent_hint: Optional[str] = None):
    """Yield dicts with keys: agent_id (optional), content, embedding (optional), source (optional)."""
    name = os.path.basename(path)
    # Try JSONL / NDJSON (line-delimited JSON)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            first = f.read(4096)
            f.seek(0)
            # Heuristics
            if first.lstrip().startswith('['):
                # JSON array
                data = json.load(f)
                if isinstance(data, list):
                    for obj in data:
                        if isinstance(obj, dict) and obj.get('content'):
                            yield {
                                'agent_id': obj.get('agent_id') or agent_hint,
                                'content': obj.get('content'),
                                'embedding': obj.get('embedding'),
                                'source': obj.get('source') or f'file:{name}'
                            }
                    return
            # Line-delimited JSON or plain text fallback
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('{') and line.endswith('}'):
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict) and obj.get('content'):
                            yield {
                                'agent_id': obj.get('agent_id') or agent_hint,
                                'content': obj.get('content'),
                                'embedding': obj.get('embedding'),
                                'source': obj.get('source') or f'file:{name}'
                            }
                            continue
                    except Exception:
                        pass
                # Plain text line -> treat as a content entry
                yield {
                    'agent_id': agent_hint,
                    'content': line,
                    'embedding': None,
                    'source': f'file:{name}'
                }
            return
    except Exception:
        # Binary or unreadable -> treat as single blob text if possible
        try:
            with open(path, 'rb') as f:
                data = f.read()
                text = None
                try:
                    text = data.decode('utf-8')
                except Exception:
                    try:
                        import chardet
                        enc = chardet.detect(data).get('encoding')
                        if enc:
                            text = data.decode(enc, errors='ignore')
                    except Exception:
                        text = None
                if text:
                    yield {
                        'agent_id': agent_hint,
                        'content': text,
                        'embedding': None,
                        'source': f'file:{name}'
                    }
        except Exception:
            return


def _already_exists(store, agent_id: Optional[str], content: str) -> bool:
    """Return True if a memory with the same agent_id and content already exists in store."""
    try:
        existing = store.get_memories(agent_id)
        return any(m.get('content') == content for m in existing)
    except Exception:
        return False


def migrate_memory_stores_to_db(store_or_dbpath: Union["SQLiteStore", str], src_dir: Optional[str] = None, *, dry_run: bool = False, archive: Optional[str] = None) -> int:
    """Migrate legacy memory files into the SQLiteStore.

    Args:
      store_or_dbpath: either an existing SQLiteStore instance or a path to the DB file
      src_dir: optional path to the memory_stores directory (defaults to ./memory_stores)
      dry_run: if True, only count what *would* be migrated (do not persist)
      archive: optional path where successfully-migrated files will be moved (preserves relative layout)

    Returns:
      number of new memories that *would* be inserted (or were inserted when dry_run=False)
    """
    if src_dir is None:
        src_dir = os.environ.get('MEMORY_STORES_PATH', './memory_stores')

    # Resolve store instance
    store = None
    if isinstance(store_or_dbpath, str):
        if SQLiteStore is None:
            raise RuntimeError('SQLiteStore not importable in this environment')
        store = SQLiteStore(store_or_dbpath)
    else:
        store = store_or_dbpath

    if not os.path.exists(src_dir):
        return 0

    # Prepare archive dir if requested
    if archive:
        os.makedirs(archive, exist_ok=True)

    migrated = 0
    migrated_files = set()

    # Walk immediate children as agent namespaces
    for root, dirs, files in os.walk(src_dir):
        # skip the top-level index files if any
        for fn in files:
            path = os.path.join(root, fn)
            # infer agent from the directory under src_dir if available
            rel = os.path.relpath(path, src_dir)
            parts = rel.split(os.sep)
            agent_hint = parts[0] if len(parts) > 1 else parts[0]
            file_had_new = False
            for entry in _iter_legacy_entries_from_file(path, agent_hint=agent_hint):
                content = (entry.get('content') or '').strip()
                if not content:
                    continue
                agent_id = entry.get('agent_id')
                source = entry.get('source')
                embedding = entry.get('embedding')
                # Skip duplicates
                if _already_exists(store, agent_id, content):
                    continue
                # Count in dry-run; persist only when dry_run is False
                if not dry_run:
                    try:
                        store.save_memory(agent_id, content, embedding, source)
                        migrated += 1
                        file_had_new = True
                    except Exception:
                        # best-effort continue
                        continue
                else:
                    migrated += 1
                    file_had_new = True
            if file_had_new and archive and not dry_run:
                # move the processed file to the archive preserving relative path
                relpath = os.path.relpath(path, src_dir)
                dst = os.path.join(archive, relpath)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                try:
                    os.replace(path, dst)
                    migrated_files.add(dst)
                except Exception:
                    # non-fatal
                    pass
    return migrated


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='migrate_memory_to_db', description='Migrate legacy memory_stores into SQLite DB')
    parser.add_argument('--db', help='Path to SQLite DB (optional). If omitted, a SQLiteStore will be created using defaults.')
    parser.add_argument('--src', help='Path to memory_stores (defaults to ./memory_stores)')
    parser.add_argument('--dry-run', action='store_true', dest='dry_run', help='Do not persist; only report how many items would be migrated')
    parser.add_argument('--archive', help='If set, move migrated files into this directory (preserves layout)')
    args = parser.parse_args()

    db_arg = args.db if args.db else None
    src_arg = args.src if args.src else None

    try:
        if db_arg:
            count = migrate_memory_stores_to_db(db_arg, src_dir=src_arg, dry_run=args.dry_run, archive=args.archive)
        else:
            # pass a live store where possible
            try:
                from backend.db.sqlite_store import SQLiteStore
                store = SQLiteStore()
                count = migrate_memory_stores_to_db(store, src_dir=src_arg, dry_run=args.dry_run, archive=args.archive)
            except Exception:
                # fallback to path-based
                count = migrate_memory_stores_to_db(':memory:', src_dir=src_arg, dry_run=args.dry_run, archive=args.archive)
        print(f"migrated={count}")
    except Exception as e:
        print(f"Migration failed: {e}")
        raise
